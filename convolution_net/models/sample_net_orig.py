import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __call__(self, sample):
        return sample.view(sample.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(self, ins, outs):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(ins, outs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(outs),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3)
        )

    def forward(self, sample):
        return self.layers(sample)


class SampleCNN(nn.Module):
    def __init__(self):
        super(SampleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(128),
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

    def forward(self, sample):
        out = self.features(sample)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    from torchsummary import summary

    n_samples = 50
    samples = torch.randn(n_samples, 1, 16000)
    targets = torch.randint(0, 2, (n_samples, 1))
    net = SampleCNN()
    summary(net, (1, 16000))
    outs = net(samples)
    print(outs.mean(dim=0), outs.var(dim=0))
