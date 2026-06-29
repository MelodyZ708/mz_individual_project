import open3d.visualization.gui as gui
import numpy as np

from como.gui.GuiWindow import GuiWindow
from como.utils.multiprocessing import transfer_data

from como.odom.sequential.TrackingSeq import TrackingSeq
from como.odom.sequential.MappingSeq import MappingSeq
from como.utils.o3d import rgb_depth_to_pcd
import torch


class ComoSeq(GuiWindow):
    def __init__(self, viz_cfg, slam_cfg, dataset):
        super().__init__(viz_cfg, slam_cfg, dataset)

    def setup_slam_processes(self, slam_cfg):
        # Setup SLAM processes
        intrinsics = self.get_intrinsics()
        img_size = self.get_img_size()

        self.sensor_tracking_only = slam_cfg["tracking"].get(
            "sensor_tracking_only", False
        )
        self.mapping_use_sensor_depth = slam_cfg["mapping"].get(
            "use_sensor_depth", False
        )

        self.tracking = TrackingSeq(slam_cfg["tracking"], intrinsics, img_size)
        self.tracking.setup()

        self.mapping = None
        if not self.sensor_tracking_only:
            self.mapping = MappingSeq(slam_cfg["mapping"], intrinsics)
            self.mapping.setup()

        # 如果你有这段额外逻辑，也要只在 mapping 存在时执行
        if (
            self.mapping is not None
            and slam_cfg["tracking"].get("color") == "unet"
        ):
            self.tracking.set_unet(self.mapping.model.gaussian_cov_net)

    def start_slam_processes(self):
        self.tracking_done = False
        self.mapping_done = False

    def shutdown_slam_processes(self):
        print("Done.")

    def signal_slam_end(self):
        self.tracking_done = True
        self.mapping_done = True

    def load_data(self, it):
        timestamp, rgb, depth = next(it)
        return timestamp, rgb, depth

    def iter(self, timestamp, rgb, depth):
        if self.sensor_tracking_only:
            self.iter_sensor_tracking_only(timestamp, rgb, depth)
            return
        
        if self.mapping_use_sensor_depth:
            self.iter_with_sensor_depth_mapping(timestamp, rgb, depth)
            return
        
        # Send input data to tracking and visualization
        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.update_curr_image_render(rgb)
        )
        # Track if init, otherwise send raw data to mapping for initialization
        if self.mapping.is_init:
            track_data_in = (timestamp, rgb.clone())
            track_data_in = transfer_data(
                track_data_in, self.tracking.device, self.tracking.dtype
            )
            track_data_viz, track_data_map = self.tracking.track(track_data_in)
            # Handle tracking viz data
            track_data_viz = transfer_data(track_data_viz, self.device, self.dtype)
            tracked_timestamp, tracked_pose = track_data_viz
            # Record data
            self.timestamps.append(tracked_timestamp)
            self.est_poses = np.concatenate((self.est_poses, tracked_pose))
            # Visualize tracked pose
            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self.update_pose_render(tracked_pose)
            )
            # Visualize background using shaders if using
            if self.render_val == "Phong":
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self.render_o3d_image()
                )
        else:
            track_data_map = ("init", timestamp, rgb.clone())

        ## Handle tracking map data
        kf_viz_data, kf_ref_data = self.mapping.map(track_data_map)
        # Update tracking kf ref
        if kf_ref_data is not None:
            kf_ref_data = transfer_data(
                kf_ref_data, self.tracking.device, self.tracking.dtype
            )
            self.tracking.update_kf_reference(kf_ref_data)
        # Visualization and bookkeeping
        if kf_viz_data is not None:
            kf_viz_data = transfer_data(kf_viz_data, self.device, self.dtype)
            (
                kf_timestamps,
                kf_rgbs,
                kf_poses,
                kf_depths,
                kf_sparse_coords,
                P_sparse,
                obs_ref_mask,
                one_way_poses,
                kf_pairs,
                one_way_pairs,
            ) = kf_viz_data
            # Storing variables to save later
            self.update_kf_vars(kf_timestamps, kf_rgbs, kf_depths, kf_poses, P_sparse)

            pcd = None
            kf_normals = None
            if self.render_val == "Point Cloud":
                pcd, kf_normals = rgb_depth_to_pcd(
                    kf_rgbs,
                    kf_depths,
                    kf_poses,
                    self.get_intrinsics(),
                    self.cfg["cos_thresh"],
                )

            gui.Application.instance.post_to_main_thread(
                self.window,
                lambda: self.update_keyframe_render(
                    kf_timestamps,
                    kf_rgbs,
                    kf_poses,
                    kf_depths,
                    kf_sparse_coords,
                    P_sparse,
                    obs_ref_mask,
                    one_way_poses,
                    kf_pairs,
                    one_way_pairs,
                    pcd,
                    kf_normals,
                ),
            )

        return
    
    def iter_with_sensor_depth_mapping(self, timestamp, rgb, depth):
        # Send input data to tracking and visualization
        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.update_curr_image_render(rgb)
        )

        # Track if init, otherwise send raw RGB-D data to mapping for initialization
        if self.mapping.is_init:
            track_data_in = (timestamp, rgb.clone())
            track_data_in = transfer_data(
                track_data_in, self.tracking.device, self.tracking.dtype
            )
            track_data_viz, track_data_map = self.tracking.track(track_data_in)

            # Handle tracking viz data
            track_data_viz = transfer_data(track_data_viz, self.device, self.dtype)
            tracked_timestamp, tracked_pose = track_data_viz

            # Record data
            self.timestamps.append(tracked_timestamp)
            self.est_poses = np.concatenate((self.est_poses, tracked_pose))

            # Visualize tracked pose
            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self.update_pose_render(tracked_pose)
            )

            # Visualize background using shaders if using
            if self.render_val == "Phong":
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self.render_o3d_image()
                )

            # Repack keyframe data so mapping can store sensor depth
            if track_data_map is not None and track_data_map[0] == "keyframe":
                _, rgb_kf, pose_curr_kf, aff_curr_kf, kf_timestamp, curr_timestamp = track_data_map
                track_data_map = (
                    "keyframe_sensor_depth",
                    rgb_kf,
                    depth.clone(),
                    pose_curr_kf,
                    aff_curr_kf,
                    kf_timestamp,
                    curr_timestamp,
                )
        else:
            track_data_map = ("init_sensor_depth", timestamp, rgb.clone(), depth.clone())

        # Handle tracking->mapping data with sensor-depth mapping path
        kf_viz_data, kf_ref_data = self.mapping.map_sensor_depth(track_data_map)

        # Update tracking keyframe reference
        if kf_ref_data is not None:
            kf_ref_data = transfer_data(
                kf_ref_data, self.tracking.device, self.tracking.dtype
            )
            self.tracking.update_kf_reference(kf_ref_data)

        # Visualization and bookkeeping
        if kf_viz_data is not None:
            kf_viz_data = transfer_data(kf_viz_data, self.device, self.dtype)
            (
                kf_timestamps,
                kf_rgbs,
                kf_poses,
                kf_depths,
                kf_sparse_coords,
                P_sparse,
                obs_ref_mask,
                one_way_poses,
                kf_pairs,
                one_way_pairs,
            ) = kf_viz_data

            self.update_kf_vars(kf_timestamps, kf_rgbs, kf_depths, kf_poses, P_sparse)

            pcd = None
            kf_normals = None
            if self.render_val == "Point Cloud":
                pcd, kf_normals = rgb_depth_to_pcd(
                    kf_rgbs,
                    kf_depths,
                    kf_poses,
                    self.get_intrinsics(),
                    self.cfg["cos_thresh"],
                )

            gui.Application.instance.post_to_main_thread(
                self.window,
                lambda: self.update_keyframe_render(
                    kf_timestamps,
                    kf_rgbs,
                    kf_poses,
                    kf_depths,
                    kf_sparse_coords,
                    P_sparse,
                    obs_ref_mask,
                    one_way_poses,
                    kf_pairs,
                    one_way_pairs,
                    pcd,
                    kf_normals,
                ),
            )

        return

    def push_sensor_reference(self, timestamp, rgb, depth, pose_w):
        print(f"[push_sensor_reference] rgb type={type(rgb)}, depth type={type(depth)}, pose type={type(pose_w)}")

        if isinstance(rgb, (tuple, list)):
            assert len(rgb) == 1
            rgb = rgb[0]

        if isinstance(depth, (tuple, list)):
            assert len(depth) == 1
            depth = depth[0]

        if isinstance(pose_w, (tuple, list)):
            assert len(pose_w) == 1
            pose_w = pose_w[0]

        timestamps = [timestamp]

        kf_rgb = rgb.clone()
        kf_pose = pose_w.clone()

        # CNN tracking 下 affine 可以先全零
        kf_aff = torch.zeros((1, 2, 1), device=pose_w.device, dtype=pose_w.dtype)

        kf_depth = depth.clone()

        # 这里再做 debug 保存，顺序才对
        if not hasattr(self, "saved_first_ref_depth"):
            self.saved_first_ref_depth = True
            torch.save(kf_depth.cpu(), "debug_first_ref_depth.pt")
            torch.save(kf_rgb.cpu(), "debug_first_ref_rgb.pt")
            print("[DEBUG] saved first reference rgb/depth")

        depth_min = torch.min(kf_depth).item()
        depth_max = torch.max(kf_depth).item()
        depth_mean = torch.mean(kf_depth).item()
        depth_nonzero = torch.count_nonzero(kf_depth > 0).item()

        print(
            "[DEBUG depth stats] "
            f"min={depth_min:.4f}, max={depth_max:.4f}, "
            f"mean={depth_mean:.4f}, nonzero={depth_nonzero}"
        )

        kf_ref_data = (timestamps, kf_rgb, kf_pose, kf_aff, kf_depth)

        kf_ref_data = transfer_data(
            kf_ref_data, self.tracking.device, self.tracking.dtype
        )

        self.tracking.update_kf_reference(kf_ref_data)

    def iter_sensor_tracking_only(self, timestamp, rgb, depth):
        # 先照常更新当前图像显示
        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.update_curr_image_render(rgb)
        )

        # 第一帧：直接作为 reference keyframe
        if not self.tracking.mapping_init:
            pose_w0 = torch.eye(
                4, device=self.tracking.device, dtype=self.tracking.dtype
            ).unsqueeze(0)

            self.push_sensor_reference(timestamp, rgb.clone(), depth.clone(), pose_w0)
            return

        # 后续帧：只跑 tracking
        track_data_in = (timestamp, rgb.clone())
        track_data_in = transfer_data(
            track_data_in, self.tracking.device, self.tracking.dtype
        )

        track_data_viz, track_data_map = self.tracking.track(track_data_in)

        # 先保留 tracking device 上的 pose，后面如需 debug / refresh 可复用
        tracked_timestamp_tracking, tracked_pose_tracking = track_data_viz

        # 如果 tracking 已经数值发散，直接跳过当前帧
        if not torch.isfinite(tracked_pose_tracking).all():
            print("[sensor_tracking_only] invalid tracked pose detected; skip this frame")
            return

        # 再搬到 GUI/device 用于显示和记录
        track_data_viz = transfer_data(track_data_viz, self.device, self.dtype)
        tracked_timestamp, tracked_pose = track_data_viz

        self.timestamps.append(tracked_timestamp)
        self.est_poses = np.concatenate((self.est_poses, tracked_pose))

        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.update_tracking_only_traj_render(tracked_pose)
        )

        # tracking-only 模式下先不走 Phong 渲染
        # 因为没有 mapping 维护的 keyframe window
        # if self.render_val == "Phong":
        #     gui.Application.instance.post_to_main_thread(
        #         self.window, lambda: self.render_o3d_image()
        #     )

        if track_data_map is not None and track_data_map[0] == "keyframe":
            print(
                "[sensor_tracking_only] keyframe requested, "
                "but refresh disabled for fixed-reference baseline"
            )