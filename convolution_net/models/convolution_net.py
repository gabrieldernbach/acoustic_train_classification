import torch.nn as nn
import torch.nn.functional as F


class LinearExtended(nn.Module):
    """Linear - Batchnorm - Dropout - Relu"""

    def __init__(self, ins, outs, drop_out_rate=0.5):
        super(LinearExtended, self).__init__()
        self.fc = nn.Linear(ins, outs)
        self.bn = nn.BatchNorm1d(outs)
        self.do = nn.Dropout(p=drop_out_rate)

    def forward(self, x):
        return F.relu(self.do(self.bn(self.fc(x))))


class ConvMaxpool(nn.Module):
    def __init__(self, ins, outs):
        super(ConvMaxpool, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, padding=1),
            nn.BatchNorm2d(outs),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def __call__(self, sample):
        return self.block(sample)


class ConvConvMaxpool(nn.Module):
    """2x (Conv - Batchnorm - Dropout - Relu)  - Maxpool"""

    def __init__(self, ins, outs, drop_out_rate=0.25):
        super(ConvConvMaxpool, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outs),
            nn.Dropout2d(p=drop_out_rate),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(outs, outs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outs),
            nn.Dropout2d(p=drop_out_rate),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.maxpool(x)


class PrintLayer(nn.Module):
    """debugging representation size in sequential layers"""

    def __call__(self, sample):
        print(sample.shape)
        return sample


class Flatten(nn.Module):
    """unsqueeze empty dimension after final pooling operation"""

    def __call__(self, sample):
        return sample.view(sample.size(0), -1)


class TinyCNN(nn.Module):
    """Convolutional Network of 3x3 filters"""

    def __init__(self):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            ConvMaxpool(1, 8),  # 40 x 126
            ConvMaxpool(8, 16),  # 20 x 63
            ConvMaxpool(16, 32),  # 10 x 31
            ConvMaxpool(32, 64),  # 5 x 15
            ConvMaxpool(64, 128),  # 2 x 7
            nn.AdaptiveMaxPool2d(1),  # 1 x 3 -> 1 x 1
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNDouble(nn.Module):
    """Convolutional Network of 3x3 filters with double convs (VGG style)"""

    def __init__(self):
        super(CNNDouble, self).__init__()
        self.features = nn.Sequential(
            ConvConvMaxpool(1, 32),  # 128 x 63
            ConvConvMaxpool(32, 64),  # 64 x 31
            ConvConvMaxpool(64, 128),  # 32 x 15
            ConvConvMaxpool(128, 128),  # 16 x 7
            nn.AdaptiveMaxPool2d(1),  # 8 x 3 -> 1 x 1
            Flatten(),
        )

        self.classifier = nn.Sequential(
            LinearExtended(128, 512),
            LinearExtended(512, 128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    ins = torch.randn(500, 1, 40, 126)
    model = TinyCNN()
    print(model)
    summary(model, input_size=(1, 40, 126))
    #
    # outs = model(ins)
    # print(outs.mean())
    # print(outs.var())
