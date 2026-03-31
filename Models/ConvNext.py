import torch
import torch.nn as nn
import torchvision.models as models

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = identity + x
        return x


class ConvNeXt(nn.Module):
    def __init__(self, num_classes=2, pretrained_path=None, freeze_backbone=False):
        super().__init__()
        self.stem = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        self.stem_norm = nn.LayerNorm(96, eps=1e-6)

        self.block_1 = ConvNeXtBlock(96)
        self.block_2 = ConvNeXtBlock(96)
        self.block_3 = ConvNeXtBlock(96)

        self.downsample_norm_1 = nn.LayerNorm(96, eps=1e-6)
        self.downsample_conv_1 = nn.Conv2d(96, 192, kernel_size=2, stride=2)

        self.block_4 = ConvNeXtBlock(192)
        self.block_5 = ConvNeXtBlock(192)
        self.block_6 = ConvNeXtBlock(192)

        self.downsample_norm_2 = nn.LayerNorm(192, eps=1e-6)
        self.downsample_conv_2 = nn.Conv2d(192, 384, kernel_size=2, stride=2)

        self.block_7 = ConvNeXtBlock(384)
        self.block_8 = ConvNeXtBlock(384)
        self.block_9 = ConvNeXtBlock(384)
        self.block_10 = ConvNeXtBlock(384)
        self.block_11 = ConvNeXtBlock(384)
        self.block_12 = ConvNeXtBlock(384)
        self.block_13 = ConvNeXtBlock(384)
        self.block_14 = ConvNeXtBlock(384)
        self.block_15 = ConvNeXtBlock(384)

        self.downsample_norm_3 = nn.LayerNorm(384, eps=1e-6)
        self.downsample_conv_3 = nn.Conv2d(384, 768, kernel_size=2, stride=2)

        self.block_16 = ConvNeXtBlock(768)
        self.block_17 = ConvNeXtBlock(768)
        self.block_18 = ConvNeXtBlock(768)

        self.downsample_norm_4 = nn.LayerNorm(768, eps=1e-6)
        self.downsample_conv_4 = nn.Conv2d(768, 1024, kernel_size=2, stride=2)

        self.block_19 = ConvNeXtBlock(1024)
        self.block_20 = ConvNeXtBlock(1024)
        self.block_21 = ConvNeXtBlock(1024)

        self.norm = nn.LayerNorm(1024, eps=1e-6)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(1024, num_classes)

        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)

        if freeze_backbone:
            self._freeze_backbone()

    def forward(self, x):
        x = self.stem(x)
        x = x.permute(0, 2, 3, 1)
        x = self.stem_norm(x)
        x = x.permute(0, 3, 1, 2)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        x = x.permute(0, 2, 3, 1)
        x = self.downsample_norm_1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample_conv_1(x)

        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)

        x = x.permute(0, 2, 3, 1)
        x = self.downsample_norm_2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample_conv_2(x)

        x = self.block_7(x)
        x = self.block_8(x)
        x = self.block_9(x)
        x = self.block_10(x)
        x = self.block_11(x)
        x = self.block_12(x)
        x = self.block_13(x)
        x = self.block_14(x)
        x = self.block_15(x)

        x = x.permute(0, 2, 3, 1)
        x = self.downsample_norm_3(x)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample_conv_3(x)

        x = self.block_16(x)
        x = self.block_17(x)
        x = self.block_18(x)

        x = x.permute(0, 2, 3, 1)
        x = self.downsample_norm_4(x)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample_conv_4(x)

        x = self.block_19(x)
        x = self.block_20(x)
        x = self.block_21(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x

    def _load_pretrained_weights(self, pretrained_path):
        state_dict = torch.load(pretrained_path, map_location='cpu')
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'backbone' in k:
                continue
            new_state_dict[k] = v
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    def _freeze_backbone(self):
        for name, param in self.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        print("Backbone frozen, only classifier trainable")


class ConvNeXtClassifierBackbone(nn.Module):
    def __init__(self, num_classes=2, pretrained_path=None, freeze_backbone=False, variant='base'):
        super().__init__()

        variant_map = {
            'tiny': (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
            'small': (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1),
            'base': (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
            'large': (models.convnext_large, models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
        }
        model_constructor, default_weights = variant_map.get(variant.lower(),
                                                             (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1))

        if pretrained_path:
            self.backbone = model_constructor(weights=None)
            self.backbone.load_state_dict(torch.load(pretrained_path), strict=False)
        else:
            self.backbone = model_constructor(weights=default_weights)
            print(f"Загружены веса ImageNet для ConvNeXt-{variant}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = None
        for module in reversed(self.backbone.classifier):
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                break

        if in_features is None:
            for module in self.backbone.classifier:
                if hasattr(module, 'normalized_shape') and len(module.normalized_shape) == 1:
                    in_features = module.normalized_shape[0]
                    break

        if in_features is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                x = self.backbone.features(dummy)
                x = self.backbone.avgpool(x)
                in_features = x.numel() // x.size(0)
            print("Определили in_features через прогон dummy-тензора")

        print(f"in_features = {in_features}")

        self.backbone.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)