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
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

    def forward(self, sample):
        return self.block(sample)


class DwsBlock(nn.Module):
    def __init__(self, ins, outs, p):
        super(DwsBlock, self).__init__()

        depth = outs * 6
        self.block = nn.Sequential(
            nn.Conv1d(ins, depth, kernel_size=1),
            nn.BatchNorm1d(depth),
            nn.ReLU(),

            nn.Conv1d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(depth),
            nn.ReLU(),

            nn.Conv1d(depth, outs, kernel_size=1),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p=p)
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


class SampCNN(nn.Module):
    def __init__(self):
        super(SampCNN, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),  # 16384 - 5461
            ConvBlock(32, 32),  # - 1820
            ConvBlock(32, 32),  # - 606
            ConvBlock(32, 64),  # - 202
            ConvBlock(64, 64),  # - 67
            ConvBlock(64, 64),  # - 22
            ConvBlock(64, 128),  # - 7
            nn.AdaptiveMaxPool1d(1),
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


class SampleCNN(nn.Module):
    def __init__(self, n_filter, p):
        super(SampleCNN, self).__init__()

        fn = list(zip(n_filter, n_filter[1:]))
        self.features = nn.Sequential(*[DwsBlock(i, o, p=p) for i, o in fn])

        self.clf = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            Flatten(),
            nn.Linear(n_filter[-1], 1),
            nn.Sigmoid()
        )

        self.criterion = BCELoss()
        self.metric = BinaryClassificationMetrics

    def forward(self, batch):
        out = self.features(batch['audio'])
        out = self.clf(out)
        return {'target': out}


if __name__ == "__main__":
    n_samples = 50
    # audio = torch.randn(n_samples, 1, 16384)
    audio = torch.randn(n_samples, 1, 5 * 8192)
    targets = torch.randint(0, 2, (n_samples, 1))
    batch = {'audio': audio}

    model = SampleCNN([1, 2, 4, 8, 16, 32, 64, 128, 256], 0.4)
    model(batch)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # summary(model, (1, 16384))
    # outs = net(samples)
    # print(outs.mean(dim=0), outs.var(dim=0))
