import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

"""
Modified Efficient Net (Key Feature : Scalability)
The network was originally specified for imagenet which
takes 3 x 224 x 224 images as inputs and predicts 1000 classes.

The input resolution of the spectrograms is adapted by resizing.
Input channels get expanded by prepending an additional convolutional layer that project to 3 channels
For the outputs we prepend one fully connected layer mapping to 1000 nodes to 1.
"""
eff_net = EfficientNet.from_name('efficientnet-b0')
first_conv_layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
eff_net = nn.Sequential(first_conv_layer, eff_net, nn.Linear(1000, 2), nn.Softmax(dim=-1))

"""
classical resnet18 and 50
"""
resnet18 = models.resnet18()
resnet50 = models.resnet50()


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


class Flatten(nn.Module):
    """
    Flatten the 1 x 1 x C Tensor of the fully convolutional network
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


ResNet = nn.Sequential(  # beginning shape 224
    ConvBlock(1, 64, stride=2),  # remaining shape 122
    ResBlock(64, 64),
    ResBlock(64, 64),
    ResBlock(64, 128, stride=2),  # 61
    ResBlock(128, 128),
    ResBlock(128, 128),
    ResBlock(128, 256, stride=2),  # 31
    ResBlock(256, 256),
    ResBlock(256, 256),
    ResBlock(256, 512, stride=2),  # 16
    ResBlock(512, 512),
    ResBlock(512, 512),
    ResBlock(512, 512, stride=2),  # 8
    ResBlock(512, 512),
    ResBlock(512, 512),
    ResBlock(512, 512, stride=2),  # 4
    ResBlock(512, 512),
    ResBlock(512, 512),
    ResBlock(512, 512, stride=2),  # 2
    ResBlock(512, 512),
    ResBlock(512, 512),
    ConvBlock(512, 256, stride=2),  # 1
    ConvBlock(256, 128),
    ConvBlock(128, 64),
    ConvBlock(64, 32),
    ConvBlock(32, 2),
    Flatten()
)

# todo: build with bilinear layers torch.nn.Bilinear
