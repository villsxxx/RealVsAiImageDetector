import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm_1 = nn.BatchNorm2d(out_channels)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm_2 = nn.BatchNorm2d(out_channels)
        self.relu_2 = nn.ReLU(inplace=True)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        self.skip_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv_1(x)
        out = self.norm_1(out)
        out = self.relu_1(out)

        out = self.conv_2(out)
        out = self.norm_2(out)

        identity = self.skip_conv(identity)
        identity = self.skip_norm(identity)

        out = out + identity
        out = self.relu_2(out)

        return out


class CustomResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.conv_1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.norm_1 = nn.BatchNorm2d(64)
        self.relu_1 = nn.ReLU(inplace=True)
        self.maxp_1 = nn.MaxPool2d(3, 2, 1)

        self.layer_1 = ResBlock(64, 64, stride=1)
        self.layer_2 = ResBlock(64, 128, stride=2)
        self.layer_3 = ResBlock(128, 256, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat_1 = nn.Flatten()
        self.line_1 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.relu_1(x)
        x = self.maxp_1(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        x = self.avgpool(x)
        x = self.flat_1(x)
        x = self.line_1(x)

        return x


if __name__ == '__main__':
    model = CustomResNet(num_classes=2)
