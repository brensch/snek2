import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class EgoSnakeNet(nn.Module):
    def __init__(self, width=11, height=11):
        super(EgoSnakeNet, self).__init__()
        self.width = width
        self.height = height

        # 0. Food
        # 1. Hazards
        # 2. My Head
        # 3. My Body (stacked counts)
        # 4. My Health (global plane)
        # 5-7. Enemy 1 (head, body, health)
        # 8-10. Enemy 2 (head, body, health)
        # 11-13. Enemy 3 (head, body, health)
        in_channels = 14

        # Backbone
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.res_blocks = nn.Sequential(*[ResBlock(256) for _ in range(5)])

        # Policy head: 4 move logits
        self.policy_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * width * height, 4)

        # Value head: scalar (-1..1)
        self.value_conv = nn.Conv2d(256, 16, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * width * height, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        p = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
