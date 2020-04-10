import torch.nn as nn
import torch.nn.functional as F

from convolution_net.callback import BinaryClassificationMetrics
from convolution_net.loss import BCELoss


class LinearBlock(nn.Module):
    def __init__(self, ins, outs, p=0.5):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(ins, outs)
        self.bn = nn.BatchNorm1d(outs)
        self.do = nn.Dropout(p=p)

    def forward(self, x):
        return F.relu(self.do(self.bn(self.fc(x))))


class ConvolutionBlock(nn.Module):
    def __init__(self, ins, outs, p):
        super(ConvolutionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, padding=1),
            nn.BatchNorm2d(outs),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=p / 2),
        )

    def __call__(self, sample):
        return self.block(sample)


class DoubleConv(nn.Module):
    def __init__(self, ins, outs, p):
        super(DoubleConv, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outs),
            nn.Dropout2d(p=p),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(outs, outs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outs),
            nn.Dropout2d(p=p),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.maxpool(x)


class Flatten(nn.Module):
    def __call__(self, sample):
        return sample.view(sample.size(0), -1)


class MelCNN(nn.Module):
    def __init__(self, dropout_ratio=0.5):
        super(MelCNN, self).__init__()
        self.features = nn.Sequential(
            ConvolutionBlock(1, 8, p=dropout_ratio),  # 40 x 126
            ConvolutionBlock(8, 16, p=dropout_ratio),  # 20 x 63
            ConvolutionBlock(16, 32, p=dropout_ratio),  # 10 x 31
            ConvolutionBlock(32, 64, p=dropout_ratio),  # 5 x 15
            ConvolutionBlock(64, 128, p=dropout_ratio),  # 2 x 7
            nn.AdaptiveMaxPool2d(1),  # 1 x 3 -> 1 x 1
            Flatten(),
        )
        self.classifier = nn.Sequential(
            LinearBlock(128, 64, p=dropout_ratio),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.criterion = BCELoss()
        self.metric = BinaryClassificationMetrics

    def forward(self, batch):
        x = batch['audio']
        x = self.features(x)
        x = self.classifier(x)
        return {'target': x}


if __name__ == '__main__':
    import torch

    batch = {'audio': torch.randn(2, 1, 40, 126)}
    model = MelCNN()
    # print(model)
    print(model(batch).shape)
    # summary(model, input_size=(1, 40, 126))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
