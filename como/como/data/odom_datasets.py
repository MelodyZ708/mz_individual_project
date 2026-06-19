import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np
import cv2

import os
import re
import glob

from como.geometry.camera import resize_intrinsics


# Assuming one by one loading
def odom_collate_fn(batch):
    assert len(batch) == 1
    return (batch[0][0], batch[0][1].unsqueeze(0))


class OdometryDataset(Dataset):
    def __init__(self, img_size):
        self.is_live = False
        self.img_size = img_size

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        timestamp = self.load_timestamp(idx)
        rgb = self.load_rgb(idx)
        return timestamp, rgb


class TumOdometryDataset(OdometryDataset):
    def __init__(self, seq_path, img_size):
        super().__init__(img_size)

        self.seq_path = seq_path

        tmp = self.seq_path.rsplit("/", 3)
        self.save_traj_name = tmp[1] + "_" + tmp[2]

        # RGB only
        rgb_file = open(seq_path + "rgb.txt")
        lines = rgb_file.readlines()
        self.ts_list = []
        self.rgb_list = []
        for i in range(3, len(lines)):  # Skip info from first 3 lines
            line_list = lines[i].split()
            self.ts_list.append(float(line_list[0]))
            self.rgb_list.append(os.path.join(seq_path, line_list[1]))

        self.data_len = len(self.rgb_list)

        match = re.search("freiburg(\d+)", seq_path)
        if match:
            dataset_ind = int(match.group(1))
            self.setup_camera_vars(dataset_ind)
        else:
            # fallback for self-recorded data (e.g. RealSense D435i 640x480)
            from como.geometry.camera import resize_intrinsics
            size_orig = torch.tensor([480, 640])
            image_scale_factors = torch.tensor(self.img_size) / size_orig
            intrinsics_orig = torch.tensor([
                [615.0,   0.0, 320.0],
                [  0.0, 615.0, 240.0],
                [  0.0,   0.0,   1.0]
            ])
            self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)
            self.distortion = None
        self.USE_BRIGHTNESS_AUG = False

    def generate_brightness_curve(self, total_frames):
        # 模拟符合实际的渐进式光照变化（例如经过窗户或阴影）
        curve = np.ones(total_frames)
        
        # 1. 模拟经过亮区（如窗户）：亮度逐渐上升到1.5倍，然后缓慢回落
        start1, peak1, end1 = 100, 175, 250
        if total_frames > end1:
            x_up = np.linspace(0, np.pi, peak1 - start1)
            curve[start1:peak1] = 1.0 + 0.5 * (1 - np.cos(x_up)) / 2
            x_down = np.linspace(0, np.pi, end1 - peak1)
            curve[peak1:end1] = 1.0 + 0.5 * (1 + np.cos(x_down)) / 2

        # 2. 模拟经过暗区（如阴影）：亮度逐渐下降到0.6倍，然后缓慢回升
        start2, peak2, end2 = 400, 475, 550
        if total_frames > end2:
            x_up = np.linspace(0, np.pi, peak2 - start2)
            curve[start2:peak2] = 1.0 - 0.4 * (1 - np.cos(x_up)) / 2
            x_down = np.linspace(0, np.pi, end2 - peak2)
            curve[peak2:end2] = 1.0 - 0.4 * (1 + np.cos(x_down)) / 2
            
        return curve


    def setup_camera_vars(self, dataset_ind):
        size_orig = torch.tensor([480, 640])
        image_scale_factors = torch.tensor(self.img_size) / size_orig

        ## ROS Default
        # intrinsics_orig = torch.tensor([ [ 525.0,    0.0,  319.5],
        #                                   [   0.0,  525.0,  239.5],
        #                                   [   0.0,    0.0,    1.0]] )
        # distortion = None

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
                + dataset_ind
                + " is not a valid dataset."
            )

        ## NOTE: With 0 distortion, getOptimalNewCameraMatrix gives different K,
        # and initUndistortRectifyMap will have a map with values at the borders...

        # Setup distortion
        if distortion is not None:
            orig_img_size = [size_orig[1].item(), size_orig[0].item()]
            K = intrinsics_orig.numpy()
            # alpha = 0.0 means invalid pixels are cropped, while 1.0 means all original pixels are present in new image
            K_u, validPixROI = cv2.getOptimalNewCameraMatrix(
                K, distortion, orig_img_size, alpha=0, newImgSize=orig_img_size
            )
            # TODO: What type to use for maps?
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                K, distortion, None, K_u, orig_img_size, cv2.CV_32FC1
            )
            intrinsics_orig = torch.from_numpy(K_u)
        else:
            self.map1, self.map2 = None, None
            intrinsics_orig = intrinsics_orig

        self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)

    def load_rgb(self, idx):
        bgr_np = cv2.imread(self.rgb_list[idx])
        rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)

        # Undistort/resize
        # Use precalculated initUndistortRectifyMap for faster dataloading
        if hasattr(self, "map1") and self.map1 is not None:
            rgb_np_u = cv2.remap(rgb_np, self.map1, self.map2, cv2.INTER_LINEAR)
        else:
            rgb_np_u = rgb_np

        new_img_size = [self.img_size[1], self.img_size[0]]
        rgb_np_resized = cv2.resize(
            rgb_np_u, new_img_size, interpolation=cv2.INTER_LINEAR
        )

        rgb = TF.to_tensor(rgb_np_resized)
        
        if not hasattr(self, 'brightness_curve'):
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
        h, w = depth.shape[-2:]
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
            rgb_list, key=lambda x: int(re.findall("\d+", x.rsplit("/", 1)[-1])[0])
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

        image_scale_factors = (
            torch.tensor([480, 640]) / size_orig
        )  # Images saved as this size
        self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)
        self.intrinsics[0, 2] -= self.crop_size
        self.intrinsics[1, 2] -= self.crop_size
        print(self.intrinsics)
        image_scale_factors = torch.tensor(self.img_size) / torch.tensor(
            [480 - 2 * crop_size, 640 - 2 * crop_size]
        )
        self.intrinsics = resize_intrinsics(self.intrinsics, image_scale_factors)

        print(intrinsics_orig)
        print(self.intrinsics)

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

    # TODO: Is ScanNet always 30 FPS?
    def load_timestamp(self, idx):
        return idx / 30.0


class ReplicaDataset(OdometryDataset):
    def __init__(self, seq_path, img_size):
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

        # Resize - different aspect ratio but keeps all image content
        image_scale_factors = torch.tensor(self.img_size) / size_orig
        self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)

        print(self.intrinsics)

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


# update!
class EurocOdometryDataset(OdometryDataset):
    """
    Loader for TUM-VI / EuRoC format datasets.
    Expected directory structure:
        dataset_dir/
            cam0/
                data/          # images named as <timestamp_ns>.png
                data.csv       # timestamp_ns, filename
            mocap0/
                data.csv       # timestamp_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z
            imu0/
                data.csv       # (optional, not used)
    """
    def __init__(self, seq_path, img_size):
        super().__init__(img_size)

        self.seq_path = seq_path

        # Save trajectory name from path
        tmp = seq_path.rstrip("/").rsplit("/", 1)
        self.save_traj_name = tmp[-1]

        # Load image list from cam0/data.csv
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
                self.ts_list.append(ts_ns / 1e9)  # nanoseconds -> seconds
                self.rgb_list.append(os.path.join(seq_path, "mav0", "cam0", "data", fname))

        self.data_len = len(self.rgb_list)
        self.setup_camera_vars()

    def setup_camera_vars(self):
        # TUM-VI cam0 intrinsics (512x512, fisheye undistorted to pinhole)
        # From official calibration: https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset
        size_orig = torch.tensor([512, 512] )
        intrinsics_orig = torch.tensor(
            [[190.97847715128717, 0.0, 254.93170605935475],
             [0.0, 190.9733070521226, 256.8974428996504],
             [0.0, 0.0, 1.0]]
        )
        image_scale_factors = torch.tensor(self.img_size) / size_orig
        self.intrinsics = resize_intrinsics(intrinsics_orig, image_scale_factors)
        self.map1, self.map2 = None, None

    def load_rgb(self, idx):
        bgr_np = cv2.imread(self.rgb_list[idx])
        rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
        new_img_size = [self.img_size[1], self.img_size[0]]
        rgb_np_resized = cv2.resize(rgb_np, new_img_size, interpolation=cv2.INTER_LINEAR)
        rgb = TF.to_tensor(rgb_np_resized)
        return rgb

    def load_timestamp(self, idx):
        return self.ts_list[idx]
