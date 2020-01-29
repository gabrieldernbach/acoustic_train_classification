"""
Custom Pre Activation Residual Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def __call__(self, sample):
        return sample.view(sample.size(0), -1)


class PreActBlock(nn.Module):
    def __init__(self, ins, outs, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(ins)
        self.conv1 = nn.Conv2d(ins, outs, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outs)
        self.conv2 = nn.Conv2d(outs, outs, stride=1, kernel_size=3, padding=1, bias=False)

        self.shortcut = nn.Identity()
        if stride != 1:
            self.shortcut = nn.Conv2d(ins, outs, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class DoubleBlock(nn.Module):
    def __init__(self, ins, outs):
        super(DoubleBlock, self).__init__()
        self.features = nn.Sequential(
            PreActBlock(ins, outs, stride=2),
            PreActBlock(outs, outs, stride=1))

    def forward(self, x):
        return self.features(x)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.features = nn.Sequential(
            DoubleBlock(1, 64),  # out shape 20, 63
            DoubleBlock(64, 128),  # 10, 31
            DoubleBlock(128, 256),  # 5, 15
            DoubleBlock(256, 512),  # 3, 7
            nn.AdaptiveAvgPool2d(1),  # 1, 1
            Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    n_samples = 50
    samples = torch.randn(n_samples, 1, 40, 126)
    targets = torch.randint(0, 2, (n_samples, 1))
    net = ResNet()
    outs = net(samples)
    print(outs.mean(dim=0), outs.var(dim=0))
