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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
