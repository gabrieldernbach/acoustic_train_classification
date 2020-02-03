import torch.nn as nn
from efficientnet_pytorch import EfficientNet

EfficientNetMod = nn.Sequential(
    nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
    EfficientNet.from_name('efficientnet-b0'),
    nn.Linear(1000, 1),
    nn.Sigmoid()
)

if __name__ == '__main__':
    import torch
    from torchsummary import summary

    ins = torch.randn(50, 1, 40, 126)
    model = EfficientNetMod
    model(ins)
    print(model)
    summary(model, input_size=(1, 40, 126))
