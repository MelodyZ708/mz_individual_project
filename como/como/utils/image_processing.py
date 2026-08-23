import torch
import torch.nn as nn

from como.data.depth_resize import pyr_depth
from como.geometry.camera import resize_intrinsics
from torchvision.models import resnet18, ResNet18_Weights


class ImageGradientModule(nn.Module):
    def __init__(self, channels, device, dtype):
        super(ImageGradientModule, self).__init__()

        # Scharr kernel
        kernel_x = (1.0 / 32.0) * torch.tensor(
            [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
            requires_grad=False,
            device=device,
            dtype=dtype,
        )
        kernel_x = kernel_x.view((1, 1, 3, 3))
        self.kernel_x = kernel_x.repeat(channels, 1, 1, 1)

        kernel_y = (1.0 / 32.0) * torch.tensor(
            [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
            requires_grad=False,
            device=device,
            dtype=dtype,
        )
        kernel_y = kernel_y.view((1, 1, 3, 3))
        self.kernel_y = kernel_y.repeat(channels, 1, 1, 1)

    def forward(self, x):
        gx = nn.functional.conv2d(
            nn.functional.pad(x, (1, 1, 1, 1), mode="reflect"),
            self.kernel_x,
            groups=x.shape[1],
        )

        gy = nn.functional.conv2d(
            nn.functional.pad(x, (1, 1, 1, 1), mode="reflect"),
            self.kernel_y,
            groups=x.shape[1],
        )

        return gx, gy


class GaussianBlurModule(nn.Module):
    def __init__(self, channels, device, dtype):
        super(GaussianBlurModule, self).__init__()

        # Matches opencv documentation
        gaussian_kernel = (1.0 / 16.0) * torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            requires_grad=False,
            device=device,
            dtype=dtype,
        )
        self.gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    def forward(self, x):
        x_blur = nn.functional.conv2d(
            nn.functional.pad(x, (1, 1, 1, 1), mode="reflect"),
            self.gaussian_kernel,
            groups=x.shape[1],
        )
        return x_blur


class ImagePyramidModule(nn.Module):
    def __init__(self, channels, start_level, end_level, device, dtype):
        super(ImagePyramidModule, self).__init__()

        self.blur_module = GaussianBlurModule(
            channels=channels, device=device, dtype=dtype
        )
        self.start_level = start_level
        self.end_level = end_level

    def forward(self, x):
        pyr = []
        x_level = x
        for i in range(self.end_level - 1):
            if i >= self.start_level:
                pyr.insert(0, x_level)
            x_level = self.blur_module(x_level)[:, :, 0::2, 0::2]
        pyr.insert(0, x_level)
        return pyr


class DepthPyramidModule(nn.Module):
    def __init__(self, start_level, end_level, mode, device):
        super(DepthPyramidModule, self).__init__()

        self.start_level = start_level
        self.end_level = end_level
        self.mode = mode

    def forward(self, x):
        pyr = []
        x_level = x
        for i in range(self.end_level - 1):
            if i >= self.start_level:
                pyr.insert(0, x_level)
            x_level = pyr_depth(x_level, self.mode, kernel_size=2)
        pyr.insert(0, x_level)
        return pyr


class IntrinsicsPyramidModule(nn.Module):
    def __init__(self, start_level, end_level, device):
        super(IntrinsicsPyramidModule, self).__init__()

        self.start_level = start_level
        self.end_level = end_level

    def forward(self, K_orig, image_scale_start):
        pyr = []
        for i in range(self.start_level, self.end_level):
            y_scale = image_scale_start[0] * pow(2.0, -i)
            x_scale = image_scale_start[1] * pow(2.0, -i)
            K_level = resize_intrinsics(K_orig, [y_scale, x_scale])
            pyr.insert(0, K_level)
        return pyr


class CNNFeatureExtractor(nn.Module):
    def __init__(self, target_channels=8, device="cuda:0", mode="rgb_cnn", 
                 channel_select="all", cnn_layer="conv1"):
        super().__init__()
        self.device = device
        self.mode = mode  # "rgb_cnn" or "cnn_only"
        self.cnn_layer = cnn_layer  # "conv1" or "layer1"
        
        # 加载预训练的 ResNet18
        from torchvision.models import resnet18, ResNet18_Weights
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        resnet.eval()
        
        # 根据 cnn_layer 参数构建特征提取器
        # 支持: "conv1" (64ch), "layer1" (64ch), "layer2" (128ch)
        if cnn_layer == "layer2":
            # conv1 + bn1 + relu + maxpool + layer1 + layer2
            # Extraction path: conv1(stride=2) -> maxpool(stride=2) -> layer1(stride=1) -> layer2(stride=2)
            # Layer2 output is H/8 × W/8 and has 128 channels; need 8× upsample
            self.feature_extractor = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
            )
            self.upsample_factor = 8
            print(f"[CNNFeatureExtractor] Using layer2 features (128ch, resolution=H/8×W/8, upsample=8×)")

        elif cnn_layer == "layer3":
            self.feature_extractor = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
            )
            self.upsample_factor = 16
            print(f"[CNNFeatureExtractor] Using layer3 features (256ch, resolution=H/16×W/16, upsample=16×)")
        elif cnn_layer == "layer4":
            self.feature_extractor = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )
            self.upsample_factor = 32
            print(f"[CNNFeatureExtractor] Using layer4 features (512ch, resolution=H/32×W/32, upsample=32×)")

        elif cnn_layer == "layer1":
            # conv1 + bn1 + relu + maxpool + layer1
            # Actually: conv1(stride=2) → H/2, maxpool(stride=2) → H/4, layer1(stride=1) → H/4
            # So layer1 output is H/4 × W/4, need 4× upsample
            self.feature_extractor = nn.Sequential(
                resnet.conv1,    # 3 → 64, stride=2, H/2 × W/2
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,  # stride=2, H/4 × W/4
                resnet.layer1    # 64 → 64, stride=1, H/4 × W/4
            )
            self.upsample_factor = 4  # need 4× upsample to restore resolution
            print(f"[CNNFeatureExtractor] Using layer1 features (64ch, resolution=H/4×W/4, upsample=4×)")
        else:
            # Default: conv1 + bn1 + relu only
            self.feature_extractor = nn.Sequential(
                resnet.conv1,    # 3 → 64, stride=2, H/2 × W/2
                resnet.bn1,
                resnet.relu
            )
            self.upsample_factor = 2  # need 2× upsample to restore resolution
            print(f"[CNNFeatureExtractor] Using conv1 features (64ch, resolution=H/2×W/2, upsample=2×)")
        
        # 冻结参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # 冻结参数已完成

        # 基于所选层设定该层的总通道数（用于解析 'd' 模式和随机抽样）
        if cnn_layer == "layer4":
            total_layer_channels = 512
        elif cnn_layer == "layer3":
            total_layer_channels = 256
        elif cnn_layer == "layer2":
            total_layer_channels = 128
        else:  # conv1, layer1
            total_layer_channels = 64

        # 随机选择通道的索引（固定种子保证可复现），样本数量等于 target_channels
        torch.manual_seed(42)
        all_rand_indices = torch.randperm(total_layer_channels)[:target_channels].to(device)

        # 解析通道选择逻辑
        if str(channel_select).lower() == "all":
            # 使用全部 target_channels 个通道（从该层的通道空间中随机挑选）
            self.selected_indices = all_rand_indices
            self.actual_cnn_channels = target_channels
        elif str(channel_select).lower() == "active":
            # 仅在 conv1 下有预定义的 "active" 模式
            if total_layer_channels == 64:
                active_mask = torch.tensor([False, True, False, True, True, True, False, True], device=device)
                self.selected_indices = all_rand_indices[active_mask]
                self.actual_cnn_channels = self.selected_indices.numel()
            else:
                print(f"[CNNFeatureExtractor] 'active' channel_select is undefined for layer {cnn_layer}; defaulting to 'all'.")
                self.selected_indices = all_rand_indices
                self.actual_cnn_channels = target_channels
        else:
            # 解析具体指定的通道
            try:
                raw = str(channel_select).strip()
                
                # "d" 前缀表示直接使用原始64通道的绝对索引（0-based）
                # 例如: "d6" → 第6个通道, "d6,d23" → 第6和第23个通道
                if raw.lower().startswith("d"):
                    ch_nums = [int(ch.strip().lstrip("dD")) for ch in raw.split(",")]
                    # 验证索引范围
                    for ch in ch_nums:
                        if ch < 0 or ch >= total_layer_channels:
                            raise ValueError(f"Direct channel index {ch} out of range [0, {total_layer_channels-1}]")
                    self.selected_indices = torch.tensor(ch_nums, device=device, dtype=torch.long)
                    self.actual_cnn_channels = len(ch_nums)
                    print(f"[CNNFeatureExtractor] Direct channel mode: using absolute indices {ch_nums} from {total_layer_channels} channels")
                else:
                    # 原有逻辑：1-based Ch编号，索引到8个随机通道中
                    # 例如: "8" → 第8个随机通道, "2,4,5,6" → 第2,4,5,6个随机通道
                    ch_strs = raw.split(",")
                    indices_to_keep = [int(ch.strip()) - 1 for ch in ch_strs if ch.strip()]
                    mask = torch.zeros(target_channels, dtype=torch.bool, device=device)
                    for idx in indices_to_keep:
                        if 0 <= idx < target_channels:
                            mask[idx] = True
                    self.selected_indices = all_rand_indices[mask]
                    self.actual_cnn_channels = mask.sum().item()
            except Exception as e:
                print(f"[Error] Failed to parse channel_select '{channel_select}': {e}. Defaulting to 'all'.")
                self.selected_indices = all_8_indices
                self.actual_cnn_channels = 8
                
        if target_channels != self.actual_cnn_channels:
            print(f"[Warning] Requested {target_channels} CNN channels, but using {self.actual_cnn_channels} based on channel_select='{channel_select}'")
        
        print(f"[CNNFeatureExtractor] Config: layer={cnn_layer}, mode={mode}, channels={self.actual_cnn_channels}, "
              f"channel_select={channel_select}")

    def forward(self, rgb_img):
        orig_dtype = rgb_img.dtype

        # ImageNet 归一化
        x = rgb_img.float()
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # 提取特征
        with torch.no_grad():
            features = self.feature_extractor(x)

        # 选择特定通道
        selected_features = features[:, self.selected_indices, :, :]

        # 双线性插值回原始分辨率
        import torch.nn.functional as F
        upsampled_features = F.interpolate(
            selected_features,
            size=rgb_img.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # 根据 mode 决定输出拼接 RGB 还是纯 CNN
        if self.mode == "cnn_only":
            output = upsampled_features
        else:  # "rgb_cnn"
            rgb_normalized = rgb_img.to(dtype=torch.float32)
            output = torch.cat([rgb_normalized, upsampled_features], dim=1)

        return output.to(dtype=orig_dtype)
    

# ===== P3新增：UNet特征提取器（Zero-cost Feature Injection） =====
class UNetFeatureExtractor(nn.Module):
    """
    从 DepthCovModule 的 U-Net 中提取 Encoder 浅层特征，
    用于 Tracking 端的光度残差计算。
    
    不运行独立的神经网络，而是直接读取 U-Net forward() 时缓存的
    中间特征（_cached_enc0 / _cached_enc1），实现零额外推理开销。
    
    使用方式：
        extractor = UNetFeatureExtractor(
            unet=mapping.model.gaussian_cov_net,
            enc_level=1,           # 0=16ch全分辨率, 1=32ch H/2
            channel_select="all",  # "all" 或 "d0,d5,d12" 等直接索引
            device="cpu",
        )
        features = extractor.get_cached(rgb_shape)
    """

    def __init__(self, unet, enc_level=1, channel_select="all", device="cpu"):
        super().__init__()
        self.unet = unet          # 不加 .to(device)，传进来已经是 CPU 版了
        self.enc_level = enc_level
        self.device = device

        # 确定该层的通道数
        # enc_level=0 -> base输出 -> base_feature_channels=16
        # enc_level=1 -> down_convs[0]输出 -> 32ch
        # enc_level=k -> 16 * 2^k
        total_ch = 16 * (2 ** enc_level)

        # 解析通道选择
        if str(channel_select).lower() == "all":
            self.selected_indices = list(range(total_ch))
        else:
            raw = str(channel_select).strip()
            if raw.lower().startswith("d"):
                self.selected_indices = [int(c.strip().lstrip("dD")) for c in raw.split(",")]
            else:
                self.selected_indices = [int(c.strip()) for c in raw.split(",")]

        self.actual_channels = len(self.selected_indices)
        self._idx_tensor = torch.tensor(self.selected_indices, dtype=torch.long, device=device)

        print(f"[UNetFeatureExtractor] enc_level={enc_level}, "
              f"total_ch={total_ch}, selected={self.actual_channels}, "
              f"channel_select={channel_select}")

        
    def extract(self, rgb):
        import torch.nn.functional as F
        target_hw = tuple(rgb.shape[-2:])
        unet_device = next(self.unet.parameters()).device  # GPU

        with torch.no_grad():
            x_norm = self.unet.normalize(rgb.float().to(unet_device))  # 搬到 GPU
            enc0 = self.unet.base(x_norm)
            if self.enc_level == 0:
                feat = enc0
            else:
                feat = self.unet.down_convs[0](enc0)
                for i in range(1, self.enc_level):
                    feat = self.unet.down_convs[i](feat)

        idx = self._idx_tensor.to(feat.device)
        feat = feat[:, idx, :, :]
        if feat.shape[-2:] != target_hw:
            feat = F.interpolate(feat.float(), size=target_hw, mode="bilinear", align_corners=False)

        return feat.to(self.device)  # 搬回 CPU 给 Tracking 用



    def get_cached(self, target_hw):
        """
        从 U-Net 的缓存中读取特征，上采样到 target_hw 分辨率。
        
        Args:
            target_hw: (H, W) tuple，目标分辨率（即当前帧的图像尺寸）
        
        Returns:
            Tensor, shape (1, actual_channels, H, W)，dtype=float32
        """
        import torch.nn.functional as F

        if self.enc_level == 0:
            cached = getattr(self.unet, "_cached_enc0", None)
        else:
            cached = getattr(self.unet, "_cached_enc1", None)

        if cached is None:
            # 缓存为空时返回全零 tensor，避免 tracking 崩溃
            import torch
            H, W = target_hw
            return torch.zeros(1, self.actual_channels, H, W, device=self.device)

        # 移动到 Tracking 设备
        feat = cached.to(self.device)

        # 选择通道
        idx = self._idx_tensor.to(feat.device)
        feat = feat[:, idx, :, :]

        # 上采样到目标分辨率
        if feat.shape[-2:] != target_hw:
            feat = F.interpolate(
                feat.float(),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        return feat
# ===== P3新增结束 =====


class UNetC2FFeatureExtractor(nn.Module):
    """Extract two selected U-Net encoder representations in one forward pass.

    ``unet`` tracking already uses encoder activations from the mapping U-Net.
    C2F needs two such representations for the *same* RGB input, so invoking
    :class:`UNetFeatureExtractor` twice would unnecessarily execute the shared
    encoder stem twice.  This wrapper evaluates ``base`` once, derives the
    requested encoder levels from that activation, selects their channels, and
    returns full-image-resolution coarse and fine tensors for the tracking
    pyramid modules.

    The current experiment protocol fixes Enc1 (32 channels at H/2) as the
    coarse branch and Enc0 (16 channels at H) as the fine branch.  The class is
    kept general enough to validate arbitrary shallow encoder-level requests.
    """

    def __init__(
        self,
        unet,
        coarse_enc_level=1,
        coarse_channel_select="all",
        fine_enc_level=0,
        fine_channel_select="all",
        device="cpu",
    ):
        super().__init__()
        self.unet = unet
        self.coarse_enc_level = int(coarse_enc_level)
        self.fine_enc_level = int(fine_enc_level)
        self.device = device
        self.coarse_indices = self._parse_selection(
            coarse_channel_select, self.coarse_enc_level, "coarse"
        )
        self.fine_indices = self._parse_selection(
            fine_channel_select, self.fine_enc_level, "fine"
        )
        self.coarse_channels = len(self.coarse_indices)
        self.fine_channels = len(self.fine_indices)

        print(
            "[UNetC2FFeatureExtractor] "
            f"coarse=enc{self.coarse_enc_level} ({self.coarse_channels} channels, "
            f"select={coarse_channel_select}); fine=enc{self.fine_enc_level} "
            f"({self.fine_channels} channels, select={fine_channel_select})"
        )

    @staticmethod
    def _parse_selection(channel_select, enc_level, branch_name):
        if enc_level < 0:
            raise ValueError(f"U-Net {branch_name} encoder level must be non-negative")
        total_channels = 16 * (2**enc_level)
        if str(channel_select).strip().lower() == "all":
            indices = list(range(total_channels))
        else:
            raw_values = [item.strip() for item in str(channel_select).split(",")]
            if not raw_values or any(not item for item in raw_values):
                raise ValueError(
                    f"U-Net {branch_name} channel selection is empty or malformed: "
                    f"{channel_select!r}"
                )
            try:
                indices = [int(item.lstrip("dD")) for item in raw_values]
            except ValueError as exc:
                raise ValueError(
                    f"U-Net {branch_name} channel selection must contain integer/d-index values: "
                    f"{channel_select!r}"
                ) from exc
        if len(set(indices)) != len(indices) or not indices:
            raise ValueError(f"U-Net {branch_name} channels must be unique and non-empty")
        if min(indices) < 0 or max(indices) >= total_channels:
            raise ValueError(
                f"U-Net {branch_name} Enc{enc_level} channel indices must lie in "
                f"0--{total_channels - 1}: {indices}"
            )
        return indices

    @staticmethod
    def _encoder_level(unet, enc0, level):
        feature = enc0
        for down_index in range(level):
            feature = unet.down_convs[down_index](feature)
        return feature

    @staticmethod
    def _select_and_resize(feature, indices, target_hw):
        import torch.nn.functional as F

        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=feature.device)
        selected = feature[:, index_tensor, :, :]
        if selected.shape[-2:] != target_hw:
            selected = F.interpolate(
                selected.float(), size=target_hw, mode="bilinear", align_corners=False
            )
        return selected

    def extract(self, rgb):
        target_hw = tuple(rgb.shape[-2:])
        unet_device = next(self.unet.parameters()).device
        with torch.no_grad():
            x_norm = self.unet.normalize(rgb.float().to(unet_device))
            enc0 = self.unet.base(x_norm)
            coarse = self._encoder_level(self.unet, enc0, self.coarse_enc_level)
            fine = self._encoder_level(self.unet, enc0, self.fine_enc_level)
            coarse = self._select_and_resize(coarse, self.coarse_indices, target_hw)
            fine = self._select_and_resize(fine, self.fine_indices, target_hw)
        return coarse.to(self.device), fine.to(self.device)
