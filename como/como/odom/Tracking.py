import torch
import torchvision.transforms.functional as TF

from como.odom.frontend.photo_tracking import photo_tracking_pyr, precalc_jacobians
from como.geometry.affine_brightness import get_aff_w_curr, get_rel_aff
from como.geometry.transforms import get_T_w_curr, get_rel_pose, transform_points
from como.geometry.camera import backprojection, projection
from como.geometry.lie_algebra import invertSE3
from como.utils.config import str_to_dtype
from como.utils.image_processing import (
    ImageGradientModule,
    ImagePyramidModule,
    IntrinsicsPyramidModule,
    DepthPyramidModule,
)
from como.utils.coords import swap_coords_xy, get_test_coords, fill_image

from como.utils.multiprocessing import init_gpu
from como.utils.image_processing import CNNFeatureExtractor, UNetFeatureExtractor
import os

class Tracking:
    def __init__(self, cfg, intrinsics, img_size):
        super().__init__()

        self.cfg = cfg
        self.device = cfg["device"]
        self.dtype = str_to_dtype(cfg["dtype"])

        self.intrinsics = intrinsics
        self.img_size = img_size

        self.mapping_init = False

    def track(self, data):
        raise NotImplementedError

    def setup(self):
        init_gpu(self.device)
        self.init_basic_vars()
        self.init_kf_vars()
        self.reset_one_way_vars()
        self.T_w_rec_last = None
        return

    def init_basic_vars(self):
        self.intrinsics = self.intrinsics.to(device=self.device, dtype=self.dtype)

        start_level = self.cfg["pyr"]["start_level"]
        end_level = self.cfg["pyr"]["end_level"]
        depth_interp_mode = self.cfg["pyr"]["depth_interp_mode"]

        intrinsics_pyr_module = IntrinsicsPyramidModule(
            start_level, end_level, self.device
        )
        self.intrinsics_pyr = intrinsics_pyr_module(self.intrinsics, [1.0, 1.0])

        if self.cfg["color"] == "gray":
            c = 1
        elif self.cfg["color"] == "rgb":
            c = 3
        
        # update!
        elif self.cfg["color"] == "cnn":
            cnn_mode = self.cfg.get("cnn_mode", "rgb_cnn")
            channel_select = self.cfg.get("cnn_channel_select", "all")

            # prefer new key `cnn_layer_name`, fall back to legacy `cnn_layer`
            cnn_layer = self.cfg.get("cnn_layer_name", self.cfg.get("cnn_layer", "conv1"))
            
            # 临时实例化以获取实际计算出的 CNN 通道数
            self.feature_extractor = CNNFeatureExtractor(
                target_channels=self.cfg.get("cnn_channels", 8), 
                device=self.device, 
                mode=cnn_mode, 
                channel_select=channel_select,
                cnn_layer=cnn_layer
            )
            actual_cnn_ch = self.feature_extractor.actual_cnn_channels
            
            # 计算总通道数 c
            c = actual_cnn_ch if cnn_mode == "cnn_only" else 3 + actual_cnn_ch

        elif self.cfg["color"] == "cnn_c2f":
            # Coarse 提取器（Layer2）
            self.feature_extractor_coarse = CNNFeatureExtractor(
                target_channels=self.cfg.get("cnn_channels_coarse", 3),
                device=self.device,
                mode="cnn_only",
                channel_select=self.cfg.get("cnn_channel_select_coarse", "d120,d66,d39"),
                cnn_layer=self.cfg.get("cnn_layer_coarse", "layer2"),
            )
            c_coarse = self.feature_extractor_coarse.actual_cnn_channels

            # Fine 提取器（Conv1）
            self.feature_extractor_fine = CNNFeatureExtractor(
                target_channels=self.cfg.get("cnn_channels_fine", 6),
                device=self.device,
                mode="cnn_only",
                channel_select=self.cfg.get("cnn_channel_select_fine", "d6,d28,d34,d50,d39,d16"),
                cnn_layer=self.cfg.get("cnn_layer_fine", "conv1"),
            )
            c_fine = self.feature_extractor_fine.actual_cnn_channels

            # 两个独立的金字塔模块（通道数不同，必须分开）
            self.img_pyr_module_coarse = ImagePyramidModule(
                c_coarse, start_level, end_level, self.device, dtype=self.dtype
            )
            self.img_pyr_module_fine = ImagePyramidModule(
                c_fine, start_level, end_level, self.device, dtype=self.dtype
            )

            # 两个独立的梯度模块
            self.gradient_module_coarse = ImageGradientModule(
                channels=c_coarse, device=self.device, dtype=self.dtype
            )
            self.gradient_module_fine = ImageGradientModule(
                channels=c_fine, device=self.device, dtype=self.dtype
            )

            # c 设为 fine 的通道数（最高分辨率层用 fine，决定最终精度）
            # 注意：这里 c 只用于后面 print，不再用于初始化单一 module
            c = c_fine


        # ===== P3新增：unet模式 =====
        elif self.cfg["color"] == "unet":
            # enc_level: 0=16ch全分辨率, 1=32ch H/2（推荐）
            enc_level = self.cfg.get("unet_enc_level", 1)
            channel_select = self.cfg.get("unet_channel_select", "all")
            # 通道数：enc_level=0->16, enc_level=1->32, enc_level=k->16*2^k
            total_ch = 16 * (2 ** enc_level)
            if str(channel_select).lower() == "all":
                c = total_ch
            else:
                raw = str(channel_select).strip()
                if raw.lower().startswith("d"):
                    c = len([x for x in raw.split(",") if x.strip()])
                else:
                    c = len([x for x in raw.split(",") if x.strip()])
            # UNetFeatureExtractor 在 set_unet() 中初始化（需要 unet 对象引用）
            # 这里仅记录通道数，extractor 在 set_unet() 中创建
            self._unet_enc_level = enc_level
            self._unet_channel_select = channel_select
            self._unet_feature_extractor = None  # 占位，set_unet()中赋值

        print(f"[Tracking] color={self.cfg['color']}, channels c={c}")

        self.gradient_module = ImageGradientModule(
            channels=c, device=self.device, dtype=self.dtype
        )
        self.img_pyr_module = ImagePyramidModule(
            c, start_level, end_level, self.device, dtype=self.dtype
        )
        self.depth_pyr_module = DepthPyramidModule(
            start_level, end_level, depth_interp_mode, self.device
        )


    # ===== P3新增：接收来自Mapping端的UNet引用 =====
    def set_unet(self, unet):
        if self.cfg["color"] != "unet":
            return
        
        # 不 deepcopy，直接持有 GPU 原始 U-Net 的引用
        self._unet_feature_extractor = UNetFeatureExtractor(
            unet=unet,   # ← 原始 GPU 模型
            enc_level=self._unet_enc_level,
            channel_select=self._unet_channel_select,
            device=self.device,  # tracking device = cpu
        )


    # ===== P3新增结束 =====

    def reset_one_way_vars(self):
        self.num_one_way_since_kf = 0

        self.last_one_way_empty_pixels = 0

        self.last_flow_rmse = 0.0
        self.last_flow_wo_rot_rmse = 0.0

    def get_curr_world_pose(self):
        T_w_curr = get_T_w_curr(self.T_w_kf, self.T_curr_kf)
        return T_w_curr

    def get_curr_world_aff(self):
        aff_curr = get_aff_w_curr(self.aff_w_kf, self.aff_curr_kf)
        return aff_curr

    def prep_tracking_img(self, rgb):
        if self.cfg["color"] == "gray":
            img_tracking = TF.rgb_to_grayscale(rgb)
        elif self.cfg["color"] == "rgb":
            img_tracking = rgb.clone()

        # update!
        elif self.cfg["color"] == "cnn":
            img_tracking = self.feature_extractor(rgb)

        elif self.cfg["color"] == "cnn_c2f":
            pyr_coarse = self.img_pyr_module_coarse(self.feature_extractor_coarse(rgb))
            pyr_fine   = self.img_pyr_module_fine(self.feature_extractor_fine(rgb))
            # pyr[0]=最低分辨率, pyr[-1]=最高分辨率
            # 前 (num_levels-1) 层用 coarse，最后一层用 fine
            num_levels = len(pyr_coarse)
            mixed_pyr = pyr_coarse[:num_levels - 1] + [pyr_fine[-1]]
            return mixed_pyr
        
                # ===== P3新增：unet模式 =====
        elif self.cfg["color"] == "unet":
            if self._unet_feature_extractor is None:
                raise RuntimeError(
                    "[Tracking] unet模式下 set_unet() 尚未被调用！\n"
                    "请在系统初始化后调用 tracking.set_unet(mapping.model.gaussian_cov_net)"
                )
            # 修复：直接对当前帧运行 encoder，而非读取存在时序bug的缓存
            img_tracking = self._unet_feature_extractor.extract(rgb)

        # ===== P3新增结束 =====

        img_pyr = self.img_pyr_module(img_tracking)
        return img_pyr

    def get_img_gradients(self, img_pyr):
        img_and_grads = []
        for l in range(len(img_pyr)):
            gx, gy = self.gradient_module(img_pyr[l])
            img_and_grads_level = torch.cat((img_pyr[l], gx, gy), dim=1)
            img_and_grads.append(img_and_grads_level)
        return img_and_grads

    # These are variables relative to a reference keyframe
    # Affine brightness parameters are not global for that frame!
    def init_kf_vars(self):
        self.T_curr_kf = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0)
        self.aff_curr_kf = torch.zeros((1, 2, 1), device=self.device, dtype=self.dtype)

        self.aff_w_kf = torch.zeros((1, 2, 1), device=self.device, dtype=self.dtype)
        self.last_one_way_num_pixels = self.img_size[-1] * self.img_size[-2]

        self.last_kf_sent_ts = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.kf_received_ts = torch.zeros(1, device=self.device, dtype=self.dtype)

    def check_keyframe(self, median_depth, num_reproj_depth, T_curr_kf):
        new_kf = False

        num_kf_pixels = self.vals_pyr[-1].shape[1]

        # Need to have received new kf from mapping to avoid immediately setting keyframe
        if self.last_kf_sent_ts <= self.kf_received_ts:
            kf_dist = torch.linalg.norm(T_curr_kf[:, :3, 3])
            if kf_dist > self.cfg["keyframing"]["kf_depth_motion_ratio"] * median_depth:
                new_kf = True
            elif (
                self.cfg["keyframing"]["kf_num_pixels_frac"]
                > num_reproj_depth / num_kf_pixels
            ):
                new_kf = True
        # else:
        #   print("Keyframe ", self.last_kf_sent_ts, " still not received, continue tracking against kf ", self.kf_received_ts)

        return new_kf

    def check_one_way_frame(self, median_depth, num_reproj_depth, T_curr_kf, T_w_curr):
        new_one_way_frame = False

        # Make threshold larger if waiting for keyframe to come soon
        extra_count = 0
        if self.last_kf_sent_ts > self.kf_received_ts:
            extra_count = 1

        # Modify threshold depending on how many num one way frames
        thresh_scale_kf = (1.0 + self.num_one_way_since_kf + extra_count) / (
            1.0 + self.cfg["keyframing"]["one_way_freq"]
        )

        dist_thresh = self.cfg["keyframing"]["kf_depth_motion_ratio"] * median_depth
        num_kf_pixels = self.vals_pyr[-1].shape[1]
        pixel_thresh = (
            1 - self.cfg["keyframing"]["kf_num_pixels_frac"]
        ) * num_kf_pixels

        # Number of empty pixels from KF reference
        num_empty_pixels = num_kf_pixels - num_reproj_depth

        # Thresholds wrt KF params
        kf_dist = torch.linalg.norm(T_curr_kf[:, :3, 3])
        if kf_dist > thresh_scale_kf * dist_thresh:
            new_one_way_frame = True
        elif num_empty_pixels > thresh_scale_kf * pixel_thresh:
            new_one_way_frame = True

        if new_one_way_frame:
            self.last_one_way_empty_pixels = num_empty_pixels
            self.T_w_rec_last = T_w_curr

        return new_one_way_frame

    def get_reproj_last_kf(self, T_curr_kf):
        P_last_kf = self.P_pyr[-1][None, -1, :, :]
        P_curr, _, _ = transform_points(T_curr_kf, P_last_kf)
        p_proj, _ = projection(self.intrinsics_pyr[-1], P_curr)
        coords_proj = swap_coords_xy(p_proj)
        depth_curr = P_curr[:, :, 2:3]

        # Mask out valid coords and depths
        def get_valid_reproj_mask(p, depth, img_size):
            valid_x = torch.logical_and(p[:, :, 0] > 0, p[:, :, 0] < img_size[-1] - 1)
            valid_y = torch.logical_and(p[:, :, 1] > 0, p[:, :, 1] < img_size[-2] - 1)
            valid_mask = torch.logical_and(valid_x, valid_y)
            valid_mask = torch.logical_and(valid_mask, depth[:, :, 0] > 0.0)
            return valid_mask

        mask = get_valid_reproj_mask(p_proj, depth_curr, self.img_size)
        coords_filt = coords_proj[mask, :]
        depth_filt = depth_curr[mask, :]
        reproj_depth = fill_image(coords_filt, depth_filt, self.img_size)
        return reproj_depth

    # Assumes KF data image sizes is same as what goes into tracking
    def update_kf_reference(self, kf_data):
        timestamps, kf_rgb, kf_pose, kf_aff, depth = kf_data

        # Update curr frame to kf variables
        if timestamps[-1] > self.kf_received_ts and self.mapping_init:
            num_kf = kf_pose.shape[0]
            #notes!!!
            self.T_w_f = get_T_w_curr(self.T_w_kf, self.T_curr_kf)
            self.T_curr_kf = get_rel_pose(self.T_w_f, kf_pose[num_kf - 1 : num_kf])

            self.aff_w_f = get_aff_w_curr(self.aff_w_kf, self.aff_curr_kf)

            #update!
            if self.cfg["color"] not in ("cnn", "cnn_c2f"):
                self.aff_curr_kf = get_rel_aff(self.aff_w_f, kf_aff[num_kf - 1 : num_kf])

            # Don't have this info but assume full image
            self.reset_one_way_vars()

        elif not self.mapping_init:
            self.mapping_init = True
            self.last_kf_sent_ts = timestamps[-1]

        # Completely new keyframe, update photometric vars
        if timestamps[-1] != self.kf_received_ts:

            # update: debug!
            print(f"[KF aff received] ts={float(timestamps[-1]):.4f} | "
                  f"kf_aff scale (a): {kf_aff[:, 0, :].flatten().tolist()} | "
                  f"kf_aff bias  (b): {kf_aff[:, 1, :].flatten().tolist()}")
            
            aff_a = kf_aff[:, 0, :].abs().max().item()
            aff_b = kf_aff[:, 1, :].abs().max().item()
            if aff_a > 0.5 or aff_b > 0.3:
                print(f"[WARNING] Crazy affine detected! max|a|={aff_a:.4f}, max|b|={aff_b:.4f}")

            # Photometric
            img_pyr = self.prep_tracking_img(kf_rgb)

            self.coords_pyr = []
            self.vals_pyr = []
            self.img_grads_pyr = []
            
            for i in range(len(img_pyr)):
                if self.cfg["color"] == "cnn_c2f":
                    # 最后一层用 fine 梯度模块，其余用 coarse
                    grad_mod = self.gradient_module_fine if i == len(img_pyr) - 1 \
                            else self.gradient_module_coarse
                    gx, gy = grad_mod(img_pyr[i])
                else:
                    gx, gy = self.gradient_module(img_pyr[i])   # 原有逻辑不动

                num_kf = img_pyr[i].shape[0]
                test_coords = get_test_coords(
                    img_pyr[i].shape[-2:], device=self.device, batch_size=num_kf
                )
                num_coords = test_coords.shape[1]

                batch_inds = (
                    torch.arange(num_kf, device=self.device)
                    .unsqueeze(1)
                    .repeat(1, num_coords)
                )
                vals = img_pyr[i][
                    batch_inds, :, test_coords[:, :, 0], test_coords[:, :, 1]
                ]
                self.vals_pyr.append(vals)  # (B,N,C)

                gx = gx[batch_inds, :, test_coords[:, :, 0], test_coords[:, :, 1]]
                gy = gy[batch_inds, :, test_coords[:, :, 0], test_coords[:, :, 1]]
                dI_dw = torch.stack((gx, gy), dim=-1)  # (B,N,C,2)
                self.img_grads_pyr.append(dI_dw)  # (B,N,2C)
                self.coords_pyr.append(test_coords)

        # Compute variables involving geometry regardless
        self.P_pyr = []
        self.dI_dT_pyr = []
        self.mask_pyr = []
        depth_pyr = self.depth_pyr_module(depth)
        for i in range(len(depth_pyr)):
            test_coords = self.coords_pyr[i]

            num_kf = test_coords.shape[0]
            num_coords = test_coords.shape[1]
            batch_inds = (
                torch.arange(num_kf, device=self.device)
                .unsqueeze(1)
                .repeat(1, num_coords)
            )

            depths = depth_pyr[i][
                batch_inds, 0, test_coords[:, :, 0], test_coords[:, :, 1]
            ]
            depths = depths.unsqueeze(-1)

            test_coords_xy = swap_coords_xy(test_coords)
            P, _ = backprojection(self.intrinsics_pyr[i], test_coords_xy, depths)

            rel_poses = (
                invertSE3(kf_pose[num_kf - 1 : num_kf]) @ kf_pose
            )  # Transform points from any kf to last kf
            P_all, _, _ = transform_points(rel_poses, P)  # (B,N,3)

            # Only use points that project close to camera boundaries and in front
            # NOTE: In theory this can be outside the FoV of the camera, but want to avoid very bad points
            p_all, _ = projection(self.intrinsics_pyr[i], P_all)

            def get_valid_reproj_mask(p, depth, img_size, img_border, depth_thresh):
                valid_x = torch.logical_and(
                    p[:, :, 0] >= -img_border,
                    p[:, :, 0] <= img_size[-1] - 1 + img_border,
                )
                valid_y = torch.logical_and(
                    p[:, :, 1] >= -img_border,
                    p[:, :, 1] <= img_size[-2] - 1 + img_border,
                )
                valid_mask = torch.logical_and(valid_x, valid_y)
                valid_mask = torch.logical_and(
                    valid_mask, depth[:, :, 0] > depth_thresh
                )
                return valid_mask

            mask = get_valid_reproj_mask(
                p_all,
                P_all[
                    :,
                    :,
                    2:3,
                ],
                depth_pyr[i].shape[-2:],
                img_border=50,
                depth_thresh=1e-4,
            )

            dI_dT = precalc_jacobians(
                self.img_grads_pyr[i], P_all, self.vals_pyr[i], self.intrinsics_pyr[i]
            )

            self.P_pyr.append(P_all)
            self.dI_dT_pyr.append(dI_dT)
            self.mask_pyr.append(mask)

        num_kf = kf_pose.shape[0]
        self.kf_received_ts = timestamps[-1]
        self.T_w_kf = kf_pose[num_kf - 1 : num_kf]

        #update!
        if self.cfg["color"] not in ("cnn", "cnn_c2f"):
            self.aff_w_kf = kf_aff[num_kf - 1 : num_kf]

    def handle_frame(self, data):
        timestamp, rgb = data
        timestamp_val = (
            float(timestamp.detach().cpu().item())
            if torch.is_tensor(timestamp)
            else float(timestamp)
        )

        # Track against reference
        img_pyr = self.prep_tracking_img(rgb)
        tracking_debug = None

        if self.cfg.get("debug_tracking_diagnostics", False):
            self.T_curr_kf, self.aff_curr_kf, tracking_debug = photo_tracking_pyr(
                self.T_curr_kf,
                self.aff_curr_kf,
                self.vals_pyr,
                self.P_pyr,
                self.dI_dT_pyr,
                self.mask_pyr,
                self.intrinsics_pyr,
                img_pyr,
                self.cfg["sigmas"]["photo"],
                self.cfg["term_criteria"],
                return_debug=True,
            )
        else:
            self.T_curr_kf, self.aff_curr_kf = photo_tracking_pyr(
                self.T_curr_kf,
                self.aff_curr_kf,
                self.vals_pyr,
                self.P_pyr,
                self.dI_dT_pyr,
                self.mask_pyr,
                self.intrinsics_pyr,
                img_pyr,
                self.cfg["sigmas"]["photo"],
                self.cfg["term_criteria"],
            )

        # === 你原来保存 residual 的逻辑，保留 ===
        if not hasattr(self, "_vis_frame_count"):
            self._vis_frame_count = 0
        self._vis_frame_count += 1

        save_interval = 100
        if self._vis_frame_count % save_interval == 0 or self._vis_frame_count == 1:
            save_dir = "vis_results/residuals"
            os.makedirs(save_dir, exist_ok=True)

            l = len(img_pyr) - 1
            mask_l = self.mask_pyr[l]
            vals_l = self.vals_pyr[l][None, mask_l, :]
            P_l = self.P_pyr[l][None, mask_l, :]

            from como.geometry.camera import transform_project
            from como.odom.frontend.photo_utils import img_interp

            pj, depth_j = transform_project(self.intrinsics_pyr[l], self.T_curr_kf, P_l)
            A_norm = 1.0 / torch.as_tensor(
                (img_pyr[l].shape[-1], img_pyr[l].shape[-2]),
                device=img_pyr[l].device,
                dtype=img_pyr[l].dtype,
            )
            vals_target, valid_mask = img_interp(img_pyr[l], pj, A_norm)
            valid_mask = torch.logical_and(valid_mask, depth_j[..., 0] > 0)

            vals_ref = torch.permute(vals_l, (0, 2, 1))
            tmp = torch.exp(-self.aff_curr_kf[:, None, 0]) * vals_target
            vals_target_adj = tmp + self.aff_curr_kf[:, None, 1]
            r = vals_target_adj - vals_ref
            r = torch.permute(r, (0, 2, 1))

            torch.save(
                {
                    "r": r.detach().cpu(),
                    "valid_mask": valid_mask.detach().cpu(),
                    "frame_idx": self._vis_frame_count,
                    "timestamp": timestamp_val,
                },
                os.path.join(
                    save_dir, f"residual_frame{self._vis_frame_count:04d}.pt"
                ),
            )
            print(f"  [VIS] Saved residual for frame {self._vis_frame_count}")

        # === 新增：tracking 数值诊断输出 ===
        if tracking_debug is not None:
            final_level = tracking_debug[-1] if len(tracking_debug) > 0 else None
            final_iter = (
                final_level["iters"][-1]
                if final_level is not None and len(final_level["iters"]) > 0
                else None
            )

            if final_iter is not None:
                suspicious = (
                    (not final_iter["cholesky_ok"])
                    or (not torch.isfinite(self.T_curr_kf).all().item())
                    or final_iter["valid_ratio"] < 0.20
                    or final_iter["h_cond"] > 1e8
                    or (not torch.isfinite(torch.tensor(final_iter["sigma_r"])).item())
                )

                if (
                    suspicious
                    or self._vis_frame_count % 50 == 0
                    or self._vis_frame_count == 1
                ):
                    print(
                        "[TRACK_DIAG] "
                        f"frame={self._vis_frame_count} "
                        f"ts={timestamp_val:.4f} "
                        f"level={final_level['level']} "
                        f"iters={final_level['num_iters']} "
                        f"stop={final_level['stop_reason']} "
                        f"valid_ratio={final_iter['valid_ratio']:.3f} "
                        f"sigma_r={final_iter['sigma_r']:.6f} "
                        f"res_med={final_iter['residual_abs_median']:.6f} "
                        f"grad_norm={final_iter['grad_norm']:.6f} "
                        f"delta_norm={final_iter['delta_norm']:.6f} "
                        f"h_cond={final_iter['h_cond']:.3e} "
                        f"jac_mean={final_iter['pose_jac_abs_mean']:.6f} "
                        f"jac_max={final_iter['pose_jac_abs_max']:.6f} "
                        f"chol_ok={final_iter['cholesky_ok']}"
                    )

                if suspicious:
                    diag_dir = "vis_results/tracking_diagnostics"
                    os.makedirs(diag_dir, exist_ok=True)
                    torch.save(
                        {
                            "frame_idx": self._vis_frame_count,
                            "timestamp": timestamp_val,
                            "tracking_debug": tracking_debug,
                            "T_curr_kf": self.T_curr_kf.detach().cpu(),
                            "aff_curr_kf": self.aff_curr_kf.detach().cpu(),
                        },
                        os.path.join(
                            diag_dir,
                            f"tracking_diag_frame{self._vis_frame_count:04d}.pt",
                        ),
                    )
                    print(
                        f"  [TRACK_DIAG] saved suspicious frame {self._vis_frame_count}"
                    )

        # Send tracked pose
        T_w_curr = self.get_curr_world_pose()

        track_data_viz = (timestamp, T_w_curr.clone())

        # Decide if keyframe or one-way frame for mapping
        track_data_map = None

        reproj_depth = self.get_reproj_last_kf(self.T_curr_kf)
        valid_depth_mask = ~torch.isnan(reproj_depth)
        num_valid_reproj_depth = torch.count_nonzero(valid_depth_mask)
        median_depth = torch.median(reproj_depth[valid_depth_mask])

        new_kf = self.check_keyframe(
            median_depth, num_valid_reproj_depth, self.T_curr_kf
        )
        if new_kf:
            track_data_map = (
                "keyframe",
                rgb.clone(),
                self.T_curr_kf,
                self.aff_curr_kf,
                self.kf_received_ts,
                timestamp,
            )
            # Need this to know whether tracking against older keyframe
            self.last_kf_sent_ts = timestamp
        else:
            # Try to see if add one way frame
            new_one_way_frame = self.check_one_way_frame(
                median_depth, num_valid_reproj_depth, self.T_curr_kf, T_w_curr
            )
            if new_one_way_frame:
                track_data_map = (
                    "one-way",
                    rgb.clone(),
                    self.T_curr_kf,
                    self.aff_curr_kf,
                    self.kf_received_ts,
                    timestamp,
                )

        return track_data_viz, track_data_map