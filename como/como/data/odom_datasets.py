import os
import re
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from como.geometry.camera import resize_intrinsics


# Assuming one by one loading
def odom_collate_fn(batch):
    assert len(batch) == 1

    timestamp, rgb, depth, pose = batch[0]
    rgb = rgb.unsqueeze(0)

    if depth is not None:
        depth = depth.unsqueeze(0)

    if pose is not None:
        pose = pose.unsqueeze(0)

    return (timestamp, rgb, depth, pose)


class OdometryDataset(Dataset):
    def __init__(self, img_size):
        self.is_live = False
        self.img_size = img_size
        self.has_depth = False
        self.has_pose = False

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        timestamp = self.load_timestamp(idx)
        rgb = self.load_rgb(idx)

        if getattr(self, "has_depth", False):
            depth = self.load_depth(idx)
        else:
            depth = None

        if getattr(self, "has_pose", False):
            pose = self.load_pose(idx)
        else:
            pose = None

        return timestamp, rgb, depth, pose


class TumOdometryDataset(OdometryDataset):
    def __init__(self, seq_path, img_size, gt_tolerance=0.03):
        super().__init__(img_size)

        self.seq_path = seq_path
        self.gt_tolerance = gt_tolerance

        tmp = self.seq_path.rstrip("/").rsplit("/", 3)
        self.save_traj_name = tmp[1] + "_" + tmp[2]

        rgb_index_path = os.path.join(seq_path, "matched_rgb.txt")
        if not os.path.exists(rgb_index_path):
            rgb_index_path = os.path.join(seq_path, "rgb.txt")

        depth_index_path = os.path.join(seq_path, "matched_depth.txt")
        if not os.path.exists(depth_index_path):
            fallback_depth = os.path.join(seq_path, "depth.txt")
            depth_index_path = fallback_depth if os.path.exists(fallback_depth) else None

        self.ts_list = []
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []

        self.has_depth = depth_index_path is not None
        self.has_pose = False

        with open(rgb_index_path, "r") as rgb_file:
            lines = rgb_file.readlines()
        for i in range(3, len(lines)):
            line_list = lines[i].split()
            self.ts_list.append(float(line_list[0]))
            self.rgb_list.append(os.path.join(seq_path, line_list[1]))

        if self.has_depth:
            depth_ts_list = []
            with open(depth_index_path, "r") as depth_file:
                lines = depth_file.readlines()
            for i in range(3, len(lines)):
                line_list = lines[i].split()
                depth_ts_list.append(float(line_list[0]))
                self.depth_list.append(os.path.join(seq_path, line_list[1]))

            assert len(self.rgb_list) == len(self.depth_list), (
                f"RGB/depth length mismatch: {len(self.rgb_list)} vs {len(self.depth_list)}"
            )

            for rgb_ts, depth_ts in zip(self.ts_list, depth_ts_list):
                assert abs(rgb_ts - depth_ts) < 1e-4, (
                    f"RGB/depth timestamp mismatch: {rgb_ts} vs {depth_ts}"
                )

        gt_path = os.path.join(seq_path, "groundtruth.txt")
        if os.path.exists(gt_path):
            self.pose_list = self.load_and_match_groundtruth(gt_path, self.ts_list)
            self.has_pose = True
            assert len(self.pose_list) == len(self.ts_list)

        self.data_len = len(self.rgb_list)

        intrinsics_path = Path(seq_path) / "intrinsics.txt"
        match = re.search(r"freiburg(\d+)", seq_path)

        if intrinsics_path.exists():
            self.setup_camera_vars_from_intrinsics_file(intrinsics_path)
        elif match:
            dataset_ind = int(match.group(1))
            self.setup_camera_vars(dataset_ind)
        else:
            size_orig = torch.tensor([480, 640], dtype=torch.float32)
            image_scale_factors = torch.tensor(self.img_size, dtype=torch.float32) / size_orig
            intrinsics_orig = torch.tensor(
                [
                    [615.0, 0.0, 320.0],
                    [0.0, 615.0, 240.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
            self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)
            self.distortion = None
            self.map1, self.map2 = None, None

        self.USE_BRIGHTNESS_AUG = False

        print(f"[TumOdometryDataset] has_depth={self.has_depth}")
        print(f"[TumOdometryDataset] has_pose={self.has_pose}")
        print(
            f"[TumOdometryDataset] num_rgb={len(self.rgb_list)}, "
            f"num_depth={len(self.depth_list) if self.has_depth else 0}, "
            f"num_pose={len(self.pose_list) if self.has_pose else 0}"
        )

    def generate_brightness_curve(self, total_frames):
        curve = np.ones(total_frames)

        start1, peak1, end1 = 100, 175, 250
        if total_frames > end1:
            x_up = np.linspace(0, np.pi, peak1 - start1)
            curve[start1:peak1] = 1.0 + 0.5 * (1 - np.cos(x_up)) / 2
            x_down = np.linspace(0, np.pi, end1 - peak1)
            curve[peak1:end1] = 1.0 + 0.5 * (1 + np.cos(x_down)) / 2

        start2, peak2, end2 = 400, 475, 550
        if total_frames > end2:
            x_up = np.linspace(0, np.pi, peak2 - start2)
            curve[start2:peak2] = 1.0 - 0.4 * (1 - np.cos(x_up)) / 2
            x_down = np.linspace(0, np.pi, end2 - peak2)
            curve[peak2:end2] = 1.0 - 0.4 * (1 + np.cos(x_down)) / 2

        return curve

    def setup_camera_vars_from_intrinsics_file(self, intrinsics_path):
        numeric_lines = []

        with open(intrinsics_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                try:
                    values = [float(x) for x in parts]
                    numeric_lines.append(values)
                except ValueError:
                    continue

        if len(numeric_lines) == 0:
            raise ValueError(f"No numeric intrinsics found in {intrinsics_path}")

        first = numeric_lines[0]

        if len(first) >= 6:
            width = float(first[0])
            height = float(first[1])
            fx = float(first[2])
            fy = float(first[3])
            cx = float(first[4])
            cy = float(first[5])
        elif len(first) == 4:
            fx = float(first[0])
            fy = float(first[1])
            cx = float(first[2])
            cy = float(first[3])
            width = 640.0
            height = 480.0
        else:
            raise ValueError(
                f"intrinsics.txt format invalid: expected 4 or at least 6 numeric values, got {len(first)}"
            )

        size_orig = torch.tensor([height, width], dtype=torch.float32)
        image_scale_factors = torch.tensor(self.img_size, dtype=torch.float32) / size_orig

        intrinsics_orig = torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)
        self.distortion = None
        self.map1, self.map2 = None, None

    def setup_camera_vars(self, dataset_ind):
        size_orig = torch.tensor([480, 640])
        image_scale_factors = torch.tensor(self.img_size) / size_orig

        if dataset_ind == 1:
            intrinsics_orig = torch.tensor(
                [[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]]
            )
            distortion = np.array([0.2624, -0.9531, -0.0054, 0.0026, 1.1633])
        elif dataset_ind == 2:
            intrinsics_orig = torch.tensor(
                [[520.9, 0.0, 325.1], [0.0, 521.0, 249.7], [0.0, 0.0, 1.0]]
            )
            distortion = np.array([0.2312, -0.7849, -0.0033, -0.0001, 0.9172])
        elif dataset_ind == 3:
            intrinsics_orig = torch.tensor(
                [[535.4, 0.0, 320.1], [0.0, 539.2, 247.6], [0.0, 0.0, 1.0]]
            )
            distortion = None
        else:
            raise ValueError(
                "TumOdometryDataset with dataset ind "
                + str(dataset_ind)
                + " is not a valid dataset."
            )

        if distortion is not None:
            orig_img_size = [size_orig[1].item(), size_orig[0].item()]
            K = intrinsics_orig.numpy()
            K_u, _ = cv2.getOptimalNewCameraMatrix(
                K, distortion, orig_img_size, alpha=0, newImgSize=orig_img_size
            )
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                K, distortion, None, K_u, orig_img_size, cv2.CV_32FC1
            )
            intrinsics_orig = torch.from_numpy(K_u)
        else:
            self.map1, self.map2 = None, None

        self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)

    def tum_tq_to_pose(self, tx, ty, tz, qx, qy, qz, qw):
        pose = np.eye(4, dtype=np.float32)
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        return torch.from_numpy(pose).to(torch.get_default_dtype())

    def load_and_match_groundtruth(self, gt_path, rgb_ts_list):
        gt_entries = []

        with open(gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 8:
                    continue

                ts = float(parts[0])
                tx, ty, tz = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:8])

                pose = self.tum_tq_to_pose(tx, ty, tz, qx, qy, qz, qw)
                gt_entries.append((ts, pose))

        if len(gt_entries) == 0:
            raise ValueError(f"No valid groundtruth poses found in {gt_path}")

        gt_ts = np.array([x[0] for x in gt_entries], dtype=np.float64)
        gt_poses = [x[1] for x in gt_entries]

        pose_list = []
        for ts in rgb_ts_list:
            idx = np.searchsorted(gt_ts, ts)

            candidates = []
            if idx < len(gt_ts):
                candidates.append(idx)
            if idx > 0:
                candidates.append(idx - 1)

            best_idx = None
            best_dt = float("inf")
            for cand in candidates:
                dt = abs(gt_ts[cand] - ts)
                if dt < best_dt:
                    best_dt = dt
                    best_idx = cand

            if best_idx is None or best_dt > self.gt_tolerance:
                raise ValueError(
                    f"No matched groundtruth pose for rgb timestamp {ts:.6f} "
                    f"within tolerance {self.gt_tolerance:.3f}s"
                )

            pose_list.append(gt_poses[best_idx])

        return pose_list

    def load_rgb(self, idx):
        bgr_np = cv2.imread(self.rgb_list[idx])
        rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)

        if hasattr(self, "map1") and self.map1 is not None:
            rgb_np_u = cv2.remap(rgb_np, self.map1, self.map2, cv2.INTER_LINEAR)
        else:
            rgb_np_u = rgb_np

        new_img_size = [self.img_size[1], self.img_size[0]]
        rgb_np_resized = cv2.resize(
            rgb_np_u, new_img_size, interpolation=cv2.INTER_LINEAR
        )

        rgb = TF.to_tensor(rgb_np_resized)

        if not hasattr(self, "brightness_curve"):
            self.brightness_curve = self.generate_brightness_curve(self.data_len)

        multiplier = self.brightness_curve[idx]

        if self.USE_BRIGHTNESS_AUG and multiplier != 1.0:
            rgb = torch.clamp(rgb * multiplier, 0.0, 1.0)

        return rgb

    def load_depth(self, idx):
        depth_np = cv2.imread(self.depth_list[idx], cv2.IMREAD_ANYDEPTH)
        depth_np = depth_np.astype(np.float32) / 5000.0
        depth = torch.from_numpy(depth_np)
        depth = depth.unsqueeze(0)
        depth_r = TF.resize(
            depth,
            self.img_size,
            interpolation=TF.InterpolationMode.NEAREST,
            antialias=False,
        )
        depth_r = depth_r.to(torch.get_default_dtype())
        return depth_r

    def load_pose(self, idx):
        return self.pose_list[idx]

    def load_timestamp(self, idx):
        return self.ts_list[idx]


class ScanNetOdometryDataset(OdometryDataset):
    def __init__(self, seq_path, img_size, crop_size):
        self.has_depth = True
        super().__init__(img_size)

        self.seq_path = seq_path
        self.crop_size = crop_size

        tmp = self.seq_path.rsplit("/", 4)
        scene_id = tmp[-2]
        self.save_traj_name = tmp[1] + "_" + scene_id

        rgb_path = seq_path + "color/"
        rgb_list = []
        for file_name in os.listdir(rgb_path):
            if file_name.endswith(".jpg"):
                rgb_list.append(os.path.join(rgb_path, file_name))

        self.rgb_list = sorted(
            rgb_list, key=lambda x: int(re.findall(r"\d+", x.rsplit("/", 1)[-1])[0])
        )

        info_file = open(seq_path + scene_id + ".txt")
        lines = info_file.readlines()

        if re.match(r"appVersionId", lines[0]):
            line_ind = 0
        else:
            line_ind = -1

        color_width = self.line_to_np(lines[3 + line_ind])
        color_height = self.line_to_np(lines[1 + line_ind])
        size_orig = torch.tensor([color_height[0], color_width[0]])

        fx = self.line_to_np(lines[6 + line_ind])
        fy = self.line_to_np(lines[8 + line_ind])
        cx = self.line_to_np(lines[10 + line_ind])
        cy = self.line_to_np(lines[12 + line_ind])

        intrinsics_orig = torch.tensor(
            [[fx[0], 0.0, cx[0]], [0.0, fy[0], cy[0]], [0.0, 0.0, 1.0]]
        )

        image_scale_factors = torch.tensor([480, 640]) / size_orig
        self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)
        self.intrinsics[0, 2] -= self.crop_size
        self.intrinsics[1, 2] -= self.crop_size
        image_scale_factors = torch.tensor(self.img_size) / torch.tensor(
            [480 - 2 * crop_size, 640 - 2 * crop_size]
        )
        self.intrinsics = resize_intrinsics(self.intrinsics, image_scale_factors)

        self.data_len = len(self.rgb_list)

    def line_to_np(self, line):
        return np.fromstring(line.split(" = ")[1], sep=" ")

    def load_rgb(self, idx):
        bgr_np = cv2.imread(self.rgb_list[idx])
        rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
        rgb = TF.to_tensor(rgb_np)
        h, w = rgb.shape[-2:]
        rgb_crop = rgb[
            ...,
            self.crop_size : (h - self.crop_size),
            self.crop_size : (w - self.crop_size),
        ]
        rgb_r = TF.resize(
            rgb_crop,
            self.img_size,
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        return rgb_r

    def load_depth(self, idx):
        depth_np = cv2.imread(self.depth_list[idx], cv2.IMREAD_ANYDEPTH)
        depth_np = depth_np.astype(np.float32) / 1000.0
        depth = TF.to_tensor(depth_np)
        h, w = depth.shape[-2:]
        depth_crop = depth[
            ...,
            self.crop_size : (h - self.crop_size),
            self.crop_size : (w - self.crop_size),
        ]
        depth_r = TF.resize(
            depth_crop,
            self.img_size,
            interpolation=TF.InterpolationMode.NEAREST,
            antialias=False,
        )
        depth_r = depth_r.to(torch.get_default_dtype())
        return depth_r

    def load_pose(self, idx):
        pose_np = np.loadtxt(self.pose_list[idx])
        pose_mat = torch.from_numpy(pose_np)
        pose_mat = pose_mat.to(torch.get_default_dtype())
        return pose_mat

    def load_timestamp(self, idx):
        return idx / 30.0


class ReplicaDataset(OdometryDataset):
    def __init__(self, seq_path, img_size):
        self.has_depth = True
        super().__init__(img_size)

        self.seq_path = seq_path

        tmp = self.seq_path.rsplit("/", 4)
        scene_id = tmp[-2]
        self.save_traj_name = tmp[1] + "_" + scene_id

        self.rgb_list = sorted(glob.glob(os.path.join(seq_path, "results/*.jpg")))

        self.data_len = len(self.rgb_list)

        self.setup_camera_vars()

    def setup_camera_vars(self):
        size_orig = torch.tensor([680, 1200])

        intrinsics_orig = torch.tensor(
            [[600.0, 0.0, 599.5], [0.0, 600.0, 339.5], [0.0, 0.0, 1.0]]
        )

        image_scale_factors = torch.tensor(self.img_size) / size_orig
        self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)

    def load_rgb(self, idx):
        bgr_np = cv2.imread(self.rgb_list[idx])
        rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)

        new_img_size = [self.img_size[1], self.img_size[0]]
        rgb_np_resized = cv2.resize(
            rgb_np, new_img_size, interpolation=cv2.INTER_LINEAR
        )

        rgb = TF.to_tensor(rgb_np_resized)
        return rgb

    def load_timestamp(self, idx):
        return idx / 30.0


class EurocOdometryDataset(OdometryDataset):
    """
    Loader for TUM-VI / EuRoC format datasets.
    Expected directory structure:
        dataset_dir/
            mav0/
                cam0/
                    data/
                    data.csv
                mocap0/
                    data.csv
    """

    def __init__(self, seq_path, img_size):
        self.has_depth = False
        super().__init__(img_size)

        self.seq_path = seq_path

        tmp = seq_path.rstrip("/").rsplit("/", 1)
        self.save_traj_name = tmp[-1]

        cam0_csv = os.path.join(seq_path, "mav0", "cam0", "data.csv")
        self.ts_list = []
        self.rgb_list = []

        with open(cam0_csv, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                parts = line.split(",")
                ts_ns = int(parts[0])
                fname = parts[1].strip()
                self.ts_list.append(ts_ns / 1e9)
                self.rgb_list.append(
                    os.path.join(seq_path, "mav0", "cam0", "data", fname)
                )

        self.data_len = len(self.rgb_list)
        self.setup_camera_vars()

    def setup_camera_vars(self):
        size_orig = torch.tensor([512, 512])
        intrinsics_orig = torch.tensor(
            [
                [190.97847715128717, 0.0, 254.93170605935475],
                [0.0, 190.9733070521226, 256.8974428996504],
                [0.0, 0.0, 1.0],
            ]
        )
        image_scale_factors = torch.tensor(self.img_size) / size_orig
        self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)
        self.map1, self.map2 = None, None

    def load_rgb(self, idx):
        bgr_np = cv2.imread(self.rgb_list[idx])
        rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
        new_img_size = [self.img_size[1], self.img_size[0]]
        rgb_np_resized = cv2.resize(
            rgb_np, new_img_size, interpolation=cv2.INTER_LINEAR
        )
        rgb = TF.to_tensor(rgb_np_resized)
        return rgb

    def load_timestamp(self, idx):
        return self.ts_list[idx]
