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
    def __init__(
        self,
        width: int = 11,
        height: int = 11,
        in_channels: int = 10,
        base_channels: int = 256,
        num_blocks: int = 5,
        policy_channels: int = 32,
        value_channels: int = 16,
        value_hidden: int = 256,
    ):
        super().__init__()
        self.width = width
        self.height = height

        # 0. Food
        # 1. Hazards
        # 2. Ego snake (single plane)
        #    - health is written as a background value across the full plane
        #    - occupied cells add: +segment_count (stacks supported), +headness, +0.25*tail_flag
        # 3. Enemy 1
        # 4. Enemy 2
        # 5. Enemy 3
        in_channels = int(in_channels)
        base_channels = int(base_channels)
        num_blocks = int(num_blocks)
        policy_channels = int(policy_channels)
        value_channels = int(value_channels)
        value_hidden = int(value_hidden)

        if base_channels <= 0 or num_blocks < 0:
            raise ValueError("base_channels must be >0 and num_blocks must be >=0")

        # Backbone
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.res_blocks = nn.Sequential(*[ResBlock(base_channels) for _ in range(num_blocks)])

        # Policy head: 4 move logits
        self.policy_conv = nn.Conv2d(base_channels, policy_channels, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(policy_channels * width * height, 4)

        # Value head: scalar (-1..1)
        self.value_conv = nn.Conv2d(base_channels, value_channels, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * width * height, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        p = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.leaky_relu(self.value_fc1(v), 0.1)
        v = torch.tanh(self.value_fc2(v))

        return p, v
