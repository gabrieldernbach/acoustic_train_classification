import torch
import torch.nn as nn

from convolution_net.callback import BinaryClassificationMetrics
from convolution_net.loss import BCELoss


class Flatten(nn.Module):
    def __call__(self, sample):
        return sample.view(sample.size(0), -1)


class SqueezeExcitation(nn.Module):
    def __init__(self, ins):
        super(SqueezeExcitation, self).__init__()
        self.ap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ins, ins // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ins // 8, ins, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.ap(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ConvBlock(nn.Module):
    def __init__(self, ins, outs, p):
        super(ConvBlock, self).__init__()

        self.mp = nn.MaxPool1d(kernel_size=3, stride=3)
        self.skip = nn.Conv1d(ins, outs, kernel_size=1, bias=False)

        depth = outs * 8
        self.conv = nn.Sequential(
            nn.Conv1d(ins, depth, kernel_size=1, bias=False),
            nn.BatchNorm1d(depth),
            nn.ReLU(),

            nn.Conv1d(depth, depth, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(depth),
            SqueezeExcitation(depth),
            nn.ReLU(),

            nn.Conv1d(depth, outs, kernel_size=1, bias=False),
            nn.BatchNorm1d(outs),
            nn.Dropout(p=p)
        )

    def forward(self, x):
        return self.mp(self.conv(x) + self.skip(x))


class SampleCNN(nn.Module):
    def __init__(self, n_filter, p):
        super(SampleCNN, self).__init__()

        fn = list(zip(n_filter, n_filter[1:]))
        self.features = nn.Sequential(*[ConvBlock(i, o, p=p) for i, o in fn])

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
