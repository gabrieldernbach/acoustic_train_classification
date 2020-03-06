import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __call__(self, sample):
        return sample.view(sample.size(0), -1)


class SeConvBlock(nn.Module):
    def __init__(self, ins, outs, squeeze_ratio=0.5):
        super(SeConvBlock, self).__init__()
        n_squeeze = int(outs * squeeze_ratio)

        self.layers = nn.Sequential(
            nn.Conv1d(ins, outs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(outs),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3)
        )

        self.excitation = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Conv1d(outs, n_squeeze, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(n_squeeze, outs, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, sample):
        out = self.layers(sample)
        excitation = self.excitation(out)
        return out * excitation


class SampleCNN(nn.Module):
    def __init__(self):
        super(SampleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=1)),
            nn.ReLU(),

            SeConvBlock(128, 128),
            SeConvBlock(128, 128),
            SeConvBlock(128, 256),
            SeConvBlock(256, 256),
            SeConvBlock(256, 512),
            SeConvBlock(512, 512),

            nn.AdaptiveAvgPool1d(1),
            Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.Dropout(p=0.5), nn.ReLU(),
            nn.Linear(512, 10), nn.BatchNorm1d(10), nn.Dropout(p=0.5), nn.ReLU(),
            nn.Linear(10, 1), nn.Sigmoid()
        )

    def forward(self, sample):
        out = self.features(sample)
        out = self.classifier(out)
        return out


class TinySampleCNN(nn.Module):
    def __init__(self):
        super(TinySampleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            SeConvBlock(32, 64),
            SeConvBlock(64, 64),
            SeConvBlock(64, 128),
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.Dropout(p=0.5), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, sample):
        out = self.features(sample)
        out = self.classifier(out)
        return out

if __name__ == "__main__":
    from torchsummary import summary

    n_samples = 50
    samples = torch.randn(n_samples, 1, 16384)
    targets = torch.randint(0, 2, (n_samples, 1))
    net = TinySampleCNN()
    summary(net, (1, 16384))
    outs = net(samples)
    print(outs.mean(dim=0), outs.var(dim=0))
