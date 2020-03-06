"""
Custom Pre Activation Residual Network
use either ELU or ReLu activations
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
        shortcut = self.shortcut(x)
        out = self.conv1(F.elu(self.bn1(x)))
        out = self.conv2(F.elu(self.bn2(out)))
        return out + shortcut


class DoubleBlock(nn.Module):
    def __init__(self, ins, outs, stride=2):
        super(DoubleBlock, self).__init__()
        self.features = nn.Sequential(
            PreActBlock(ins, outs, stride=stride),
            PreActBlock(outs, outs, stride=1))

    def forward(self, x):
        return self.features(x)


class AverageMaxPool(nn.Module):
    def __init__(self):
        super(AverageMaxPool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        a = self.avg_pool(x)
        m = self.max_pool(x)
        out = torch.cat((a, m), dim=1)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.features = nn.Sequential(
            DoubleBlock(1, 64),  # out shape 20, 63
            DoubleBlock(64, 128),  # 10, 31
            DoubleBlock(128, 256),  # 5, 15
            DoubleBlock(256, 512),  # 3, 7
            AverageMaxPool(),  # 1, 1
            Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.ELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNetLong(nn.Module):
    def __init__(self):
        super(ResNetLong, self).__init__()
        self.features = nn.Sequential(
            DoubleBlock(1, 64, stride=(2, 1)),  # out shape 20, 63
            DoubleBlock(64, 128, stride=(2, 1)),  # 10, 31
            DoubleBlock(128, 256, stride=(2, 1)),  # 5, 15
            DoubleBlock(256, 512, stride=(2, 1)),  # 3, 7
            AverageMaxPool(),  # 1, 1
            Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.ELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    from torchsummary import summary

    n_samples = 50
    samples = torch.randn(n_samples, 1, 40, 129)
    targets = torch.randint(0, 2, (n_samples, 1))
    model = ResNetLong()
    outs = model(samples)
    summary(model, input_size=(1, 40, 129))
    print(outs.mean(dim=0), outs.var(dim=0))
