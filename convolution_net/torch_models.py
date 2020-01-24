import torch.nn as nn


def make_2conv_block(ins, outs, drop_out_rate=0.25):
    conv_conv_mp = nn.Sequential(
        nn.Conv2d(ins, outs, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(outs),
        nn.Dropout2d(p=drop_out_rate),
        nn.ReLU(),
        nn.Conv2d(outs, outs, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(outs),
        nn.Dropout2d(p=drop_out_rate),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    return conv_conv_mp


def make_linear_block(ins, outs, drop_out_rate=0.5):
    linear_drop_relu = nn.Sequential(
        nn.Linear(ins, outs),
        nn.Dropout(p=drop_out_rate),
        nn.ReLU()
    )
    return linear_drop_relu


class PrintLayer(nn.Module):
    def __call__(self, sample):
        print(sample.shape)
        return sample


class Flatten(nn.Module):
    def __call__(self, sample):
        return sample.view(sample.size(0), -1)


CustomVGG = nn.Sequential(
    make_2conv_block(1, 32),  # 128 x 63
    make_2conv_block(32, 64),  # 64 x 31
    make_2conv_block(64, 128),  # 32 x 15
    make_2conv_block(128, 128),  # 16 x 7
    nn.AdaptiveMaxPool2d(1),  # 8 x 3 -> 1 x 1
    Flatten(),

    make_linear_block(128, 512),
    make_linear_block(512, 128),
    nn.Linear(128, 1),
    nn.Sigmoid()
)


class ConvBnReluMaxPool(nn.Module):
    def __init__(self, ins, outs):
        super(ConvBnReluMaxPool, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, padding=1),
            nn.BatchNorm2d(outs),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def __call__(self, sample):
        return self.layers(sample)


CustomVGG_tiny = nn.Sequential(
    ConvBnReluMaxPool(1, 8),  # 40 x 126
    ConvBnReluMaxPool(8, 16),  # 20 x 63
    ConvBnReluMaxPool(16, 32),  # 10 x 31
    ConvBnReluMaxPool(32, 64),  # 5 x 15
    ConvBnReluMaxPool(64, 128),  # 2 x 7
    nn.AdaptiveMaxPool2d(1),  # 1 x 3 -> 1 x 1
    Flatten(),

    nn.Linear(128, 512),
    nn.ReLU(),
    nn.Linear(512, 1),
    nn.Sigmoid(),
)


class ConvBlock(nn.Module):
    """
    Convolution - Batchnorm - ReLU Layer
    that halves the input shape by default e.g 32x32 to 16x16.
    to remain at same size, use stride=1
    """

    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(nf)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """
    classical residual block
    """

    def __init__(self, ni, nf, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ni, nf, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nf)

        if stride != 1 or ni != nf:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ni, nf, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(nf)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = F.relu(out)
        return out


ResNet128 = nn.Sequential(  # input shape 40, 126
    ConvBlock(1, 32, stride=2),  # remaining shape 20, 63
    ResBlock(32, 64, stride=2),  # 10, 31
    ResBlock(64, 128, stride=2),  # 5, 15
    ResBlock(128, 256, stride=2),  # 2, 7
    ResBlock(256, 512, stride=2),  # 1, 3
    nn.AdaptiveMaxPool2d(1),
    nn.Linear(512, 1),
    nn.Sigmoid()
)

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 40 x 126
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # 20 x 63
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 10 x 31
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 5 x 16
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 2 x 7
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return torch.sigmoid(out)


def PreActResNet18():
    return PreActResNet(PreActBlock, [2, 2, 2, 2])


def PreActResNet34():
    return PreActResNet(PreActBlock, [3, 4, 6, 3])


def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3])


def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3])


def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3])
