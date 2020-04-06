import torch
import torch.nn as nn

from convolution_net.callback import BinaryClassificationMetrics
from convolution_net.loss import BCELoss


class Flatten(nn.Module):
    def __call__(self, sample):
        return sample.view(sample.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(self, ins, outs):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(ins, outs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(outs),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

    def forward(self, sample):
        return self.block(sample)


# class SeConvBlock(nn.Module):
#     def __init__(self, ins, outs, squeeze_ratio=0.5):
#         super(SeConvBlock, self).__init__()
#         n_squeeze = int(outs * squeeze_ratio)
#
#         self.layers = nn.Sequential(
#             nn.Conv1d(ins, outs, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(outs),
#             nn.ReLU(),
#             nn.MaxPool1d(3, stride=3)
#         )
#
#         self.excitation = nn.Sequential(
#             nn.AdaptiveMaxPool1d(1),
#             nn.Conv1d(outs, n_squeeze, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv1d(n_squeeze, outs, kernel_size=1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, sample):
#         out = self.layers(sample)
#         excitation = self.excitation(out)
#         return out * excitation


class LargeSampleCNN(nn.Module):
    def __init__(self):
        super(LargeSampleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=1)),
            nn.ReLU(),

            ConvBlock(128, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 512),

            nn.AdaptiveAvgPool1d(1),
            Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.Dropout(p=0.5), nn.ReLU(),
            nn.Linear(512, 10), nn.BatchNorm1d(10), nn.Dropout(p=0.5), nn.ReLU(),
            nn.Linear(10, 1), nn.Sigmoid()
        )

        self.criterion = BCELoss
        self.metric = BinaryClassificationMetrics

    def forward(self, batch):
        out = self.features(batch['audio'])
        out = self.classifier(out)
        return {'target': out}


class SampleCNN(nn.Module):
    def __init__(self):
        super(SampleCNN, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),  # 16384 - 5461
            ConvBlock(32, 32),  # - 1820
            ConvBlock(32, 32),  # - 606
            ConvBlock(32, 64),  # - 202
            ConvBlock(64, 64),  # - 67
            ConvBlock(64, 64),  # - 22
            ConvBlock(64, 128),  # - 7
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.Dropout(p=0.5), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )

        self.criterion = BCELoss()
        self.metric = BinaryClassificationMetrics

    def forward(self, batch):
        out = self.features(batch['audio'])
        out = self.classifier(out)
        return {'target': out}


if __name__ == "__main__":
    n_samples = 50
    audio = torch.randn(n_samples, 1, 16384)
    targets = torch.randint(0, 2, (n_samples, 1))
    batch = {'audio': audio}

    model = SampleCNN()
    model(batch)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # summary(model, (1, 16384))
    # outs = net(samples)
    # print(outs.mean(dim=0), outs.var(dim=0))
