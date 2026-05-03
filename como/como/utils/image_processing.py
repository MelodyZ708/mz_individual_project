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
    def __init__(self, target_channels=8, device="cuda:0", mode="rgb_cnn", channel_select="all"):
        super().__init__()
        self.device = device
        self.mode = mode  # "rgb_cnn" or "cnn_only"
        
        # 加载预训练的 ResNet18
        from torchvision.models import resnet18, ResNet18_Weights
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        resnet.eval()
        
        # 提取浅层特征 (conv1 + bn1 + relu)
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        
        # 冻结参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # 随机选择通道的索引（固定种子保证与之前实验一致）
        torch.manual_seed(42)
        all_8_indices = torch.randperm(64)[:8].to(device)
        
        # 解析通道选择逻辑
        if str(channel_select).lower() == "all":
            # 使用全部 8 个通道
            self.selected_indices = all_8_indices
            self.actual_cnn_channels = 8
        elif str(channel_select).lower() == "active":
            # 仅使用活跃通道 (Ch2, 4, 5, 6, 8 -> 编程索引 1, 3, 4, 5, 7)
            active_mask = torch.tensor([False, True, False, True, True, True, False, True], device=device)
            self.selected_indices = all_8_indices[active_mask]
            self.actual_cnn_channels = 5
        else:
            # 解析具体指定的通道，如 "8" 或 "2,4,5,6" (注意用户输入的是 1-based Ch编号)
            try:
                ch_strs = str(channel_select).split(",")
                # 将 1-based 的 Ch 编号转换为 0-based 索引
                indices_to_keep = [int(ch.strip()) - 1 for ch in ch_strs if ch.strip()]
                mask = torch.zeros(8, dtype=torch.bool, device=device)
                for idx in indices_to_keep:
                    if 0 <= idx < 8:
                        mask[idx] = True
                self.selected_indices = all_8_indices[mask]
                self.actual_cnn_channels = mask.sum().item()
            except Exception as e:
                print(f"[Error] Failed to parse channel_select '{channel_select}'. Defaulting to 'all'.")
                self.selected_indices = all_8_indices
                self.actual_cnn_channels = 8
                
        if target_channels != self.actual_cnn_channels:
            print(f"[Warning] Requested {target_channels} CNN channels, but using {self.actual_cnn_channels} based on channel_select='{channel_select}'")

        print(f"[CNNFeatureExtractor] mode={self.mode}, channel_select='{channel_select}', actual_cnn_channels={self.actual_cnn_channels}, selected_indices={self.selected_indices.tolist()}, total_output_channels={self.actual_cnn_channels if self.mode == 'cnn_only' else 3 + self.actual_cnn_channels}")

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
        else: # "rgb_cnn"
            rgb_normalized = rgb_img.to(dtype=torch.float32)
            output = torch.cat([rgb_normalized, upsampled_features], dim=1)

        return output.to(dtype=orig_dtype)