import torch
from torch.utils.data import IterableDataset
import torchvision.transforms.functional as TF

import cv2
import pyrealsense2 as rs
import numpy as np
import os
import time

from como.geometry.camera import resize_intrinsics


class RealsenseDataset(IterableDataset):
    def __init__(self, img_size, cfg):
        super().__init__()
        self.is_live = True
        self.img_size = img_size
        self.cfg = cfg

        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.save_traj_name = "realsense_" + timestr

        # 录制相关初始化
        self.save_recording = cfg.get("save_recording", False)
        if self.save_recording:
            base_dir = cfg.get("save_dir", "recordings")
            self.record_dir = os.path.join(base_dir, timestr)
            self.rgb_dir = os.path.join(self.record_dir, "rgb")
            self.depth_dir = os.path.join(self.record_dir, "depth")
            os.makedirs(self.rgb_dir, exist_ok=True)
            os.makedirs(self.depth_dir, exist_ok=True)
            self.rgb_txt = open(os.path.join(self.record_dir, "rgb.txt"), "w")
            self.depth_txt = open(os.path.join(self.record_dir, "depth.txt"), "w")
            self.rgb_txt.write("# timestamp filename\n")
            self.depth_txt.write("# timestamp filename\n")
            self._frame_idx = 0
            print(f"[RealSense] Recording to: {self.record_dir}")

        self.start()

    def start(self):
        config = rs.config()
        config.enable_stream(
            stream_type=rs.stream.color,
            width=self.cfg["width"],
            height=self.cfg["height"],
            framerate=self.cfg["fps"],
        )
        # 新增：开启 depth 流（和 RGB 同分辨率）
        config.enable_stream(
            stream_type=rs.stream.depth,
            width=self.cfg["width"],
            height=self.cfg["height"],
            framerate=self.cfg["fps"],
        )

        self.pipeline = rs.pipeline()
        # 新增：对齐 depth 到 RGB 坐标系
        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.start(config)

        rgb_sensor = profile.get_device().query_sensors()[1]
        rgb_sensor.set_option(rs.option.enable_auto_exposure, True)
        rgb_sensor.set_option(rs.option.enable_auto_white_balance, True)
        rgb_sensor.set_option(rs.option.exposure, 100)

        rgb_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        rgb_intrinsics = rgb_profile.get_intrinsics()

        size_orig = torch.tensor([rgb_intrinsics.height, rgb_intrinsics.width])
        image_scale_factors = torch.tensor(self.img_size) / size_orig

        intrinsics_orig = torch.tensor(
            [
                [rgb_intrinsics.fx, 0.0, rgb_intrinsics.ppx],
                [0.0, rgb_intrinsics.fy, rgb_intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ]
        )
        distortion = np.asarray(rgb_intrinsics.coeffs)

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

    def shutdown(self):
        self.pipeline.stop()
        # 新增：关闭录制文件
        if self.save_recording:
            self.rgb_txt.close()
            self.depth_txt.close()
            print(f"[RealSense] Recording saved to: {self.record_dir}")

    def __len__(self):
        return 1.0e10

    def __iter__(self):
        return self

    def __next__(self):
        frameset = self.pipeline.wait_for_frames()
        # 新增：对齐 depth 到 RGB
        aligned = self.align.process(frameset)

        timestamp = frameset.get_timestamp()
        timestamp /= 1000.0  # original in ms

        # 改为从 aligned 取 RGB 帧
        rgb_frame = aligned.get_color_frame()
        rgb_np = np.asanyarray(rgb_frame.get_data())

        # 新增：取 depth 帧（uint16，单位 mm）
        depth_frame = aligned.get_depth_frame()
        depth_np = np.asanyarray(depth_frame.get_data())

        # Undistort
        if self.map1 is not None:
            rgb_np_u = cv2.remap(rgb_np, self.map1, self.map2, cv2.INTER_LINEAR)
        else:
            rgb_np_u = rgb_np
        new_img_size = [self.img_size[1], self.img_size[0]]
        rgb_np_resized = cv2.resize(
            rgb_np_u, new_img_size, interpolation=cv2.INTER_LINEAR
        )
        # 新增：depth resize 用 NEAREST 避免深度值插值失真
        depth_np_resized = cv2.resize(
            depth_np, new_img_size, interpolation=cv2.INTER_NEAREST
        )

        # 新增：保存到磁盘（TUM 格式）
        if self.save_recording:
            ts_str = f"{timestamp:.6f}"
            rgb_fname = f"rgb/{ts_str}.png"
            depth_fname = f"depth/{ts_str}.png"
            cv2.imwrite(
                os.path.join(self.record_dir, rgb_fname),
                rgb_np_resized[:, :, ::-1]  # RGB → BGR for cv2
            )
            cv2.imwrite(
                os.path.join(self.record_dir, depth_fname),
                depth_np_resized  # uint16 PNG，cv2 直接支持
            )
            self.rgb_txt.write(f"{ts_str} {rgb_fname}\n")
            self.depth_txt.write(f"{ts_str} {depth_fname}\n")
            self._frame_idx += 1

        rgb = TF.to_tensor(rgb_np_resized)

        return timestamp, rgb
