import torch.hub
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

EfficientNetMod = nn.Sequential(
    nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
    EfficientNet.from_name('efficientnet-b0'),
    nn.Linear(1000, 1),
    nn.Sigmoid()
)

SqueezeNetMod = nn.Sequential(
    nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
    torch.hub.load('pytorch/vision:v0.2.4', 'squeezenet1_0', pretrained=False),
    nn.Linear(1000, 1),
    nn.Sigmoid()
)
