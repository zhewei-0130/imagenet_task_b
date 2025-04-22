import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class SimpleNetV2_LiteX(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 有效層 1

        # DWConv + PWConv block（有效層 2）
        self.dwconv = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.pwconv = nn.Conv2d(64, 64, kernel_size=1)

        # Dilated conv（有效層 3）
        self.dilated = nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2)

        # SE block（有效層 4）
        self.se = SEBlock(128)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))               # conv1
        x = self.pool(F.relu(self.bn2(self.pwconv(self.dwconv(x))))) # DWConv + PWConv
        x = F.relu(self.bn3(self.dilated(x)))                        # Dilated conv
        x = self.se(x)                                               # SE
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
