import torch.nn as nn
import torchvision.models as models


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes:int=2, pretrained_path: str = None, freeze_backbone: bool = False):
        super().__init__()
        self.output_dim = num_classes


        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None),
            BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None)
        )

        downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=128)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=128, stride=2, downsample=downsample2),
            BasicBlock(in_channels=128, out_channels=128, stride=1, downsample=None)
        )

        downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=256)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(in_channels=128, out_channels=256, stride=2, downsample=downsample3),
            BasicBlock(in_channels=256, out_channels=256, stride=1, downsample=None)
        )

        downsample4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=512)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(in_channels=256, out_channels=512, stride=2, downsample=downsample4),
            BasicBlock(in_channels=512, out_channels=512, stride=1, downsample=None)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)

        x = self.fc(x)
        return x


class ResNet18ClassifierBackbone(nn.Module):
    def __init__(self, num_classes=2, pretrained_path=None, freeze_backbone=False):
        super().__init__()
        if pretrained_path:
            self.backbone = models.resnet18(weights=None)
            self.backbone.load_state_dict(torch.load(pretrained_path), strict=False)
        else:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)


class ResNet34Regressor(nn.Module):
    def __init__(self, num_points, img_height, img_width):
        super().__init__()
        self.num_points = num_points

        output_dim = num_points * 2

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None),
            BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None),
            BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None),
        )

        downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=128),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=128, stride=2, downsample=downsample2),
            BasicBlock(in_channels=128, out_channels=128, stride=1, downsample=None),
            BasicBlock(in_channels=128, out_channels=128, stride=1, downsample=None),
            BasicBlock(in_channels=128, out_channels=128, stride=1, downsample=None),
        )

        downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=256)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(in_channels=128, out_channels=256, stride=2, downsample=downsample3),
            BasicBlock(in_channels=256, out_channels=256, stride=1, downsample=None),
            BasicBlock(in_channels=256, out_channels=256, stride=1, downsample=None),
            BasicBlock(in_channels=256, out_channels=256, stride=1, downsample=None),
            BasicBlock(in_channels=256, out_channels=256, stride=1, downsample=None),
            BasicBlock(in_channels=256, out_channels=256, stride=1, downsample=None),
        )

        downsample4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=512)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(in_channels=256, out_channels=512, stride=2, downsample=downsample4),
            BasicBlock(in_channels=512, out_channels=512, stride=1, downsample=None),
            BasicBlock(in_channels=512, out_channels=512, stride=1, downsample=None),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.flatten = nn.Flatten()

        self.new_img_h = img_height // 128
        self.new_img_w = img_width // 128
        self.fc = nn.Sequential(
            nn.LayerNorm([512 * self.new_img_w * self.new_img_h]),
            nn.Linear(512 * self.new_img_w * self.new_img_h, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = self.ad_avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x.view(-1, self.num_points, 2)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50Regressor(nn.Module):
    def __init__(self, num_points, img_height, img_width):
        super().__init__()
        self.num_points = num_points

        output_dim = num_points * 2

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        downsample1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=256)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(in_channels=64, out_channels=64, stride=1, downsample=downsample1),
            Bottleneck(in_channels=256, out_channels=64, stride=1, downsample=None),
            Bottleneck(in_channels=256, out_channels=64, stride=1, downsample=None),
        )

        downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=512),
        )
        self.layer2 = nn.Sequential(
            Bottleneck(in_channels=256, out_channels=128, stride=2, downsample=downsample2),
            Bottleneck(in_channels=512, out_channels=128, stride=1, downsample=None),
            Bottleneck(in_channels=512, out_channels=128, stride=1, downsample=None),
            Bottleneck(in_channels=512, out_channels=128, stride=1, downsample=None),
        )

        downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=1024)
        )
        self.layer3 = nn.Sequential(
            Bottleneck(in_channels=512, out_channels=256, stride=2, downsample=downsample3),
            Bottleneck(in_channels=1024, out_channels=256, stride=1, downsample=None),
            Bottleneck(in_channels=1024, out_channels=256, stride=1, downsample=None),
            Bottleneck(in_channels=1024, out_channels=256, stride=1, downsample=None),
            Bottleneck(in_channels=1024, out_channels=256, stride=1, downsample=None),
            Bottleneck(in_channels=1024, out_channels=256, stride=1, downsample=None),
        )

        downsample4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=2048)
        )
        self.layer4 = nn.Sequential(
            Bottleneck(in_channels=1024, out_channels=512, stride=2, downsample=downsample4),
            Bottleneck(in_channels=2048, out_channels=512, stride=1, downsample=None),
            Bottleneck(in_channels=2048, out_channels=512, stride=1, downsample=None),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.flatten = nn.Flatten()

        self.new_img_h = img_height // 256
        self.new_img_w = img_width // 256
        self.fc = nn.Sequential(
            nn.LayerNorm([2048 * self.new_img_w * self.new_img_h]),
            nn.Linear(2048 * self.new_img_w * self.new_img_h, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x.view(-1, self.num_points, 2)


if __name__ == "__main__":
    import torch

    model = ResNet18ClassifierBackbone(num_classes=2)
    print(model)

    dummy = torch.randn(2, 3, 768, 1024)

    out = model(dummy)
    print(out.shape)