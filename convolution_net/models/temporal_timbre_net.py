import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from callback import BinaryClassificationMetrics
from loss import BCELoss


class ConvBnMp(nn.Module):
    def __init__(self, ins, outs, p):
        super(ConvBnMp, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(ins, outs, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(outs),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(p),
        )

    def forward(self, x):
        return self.layers(x)


class AverageMaxpoolCat(nn.Module):
    def __init__(self):
        super(AverageMaxpoolCat, self).__init__()
        self.ap = nn.AdaptiveAvgPool1d(1)
        self.mp = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], dim=1).squeeze(dim=2)


class AvgPoolFrequency(nn.Module):
    def forward(self, x):
        return x.mean(dim=2).squeeze(dim=2)


class TemporalConv(nn.Module):
    def __init__(self, ins, outs):
        super(TemporalConv, self).__init__()

        self.apf = AvgPoolFrequency()
        self.layers = nn.ModuleList()
        for i, k in enumerate([127, 63, 31], start=1):
            self.layers.append(nn.Conv1d(ins, outs * i ** 2, kernel_size=k, stride=1, padding=int(np.floor(k / 2))))

    def forward(self, x):
        x = self.apf(x)
        x = torch.cat([la(x) for la in self.layers], dim=1)
        return x


class TimbreConv(nn.Module):
    def __init__(self, ins, outs):
        super(TimbreConv, self).__init__()

        self.layers = nn.ModuleList()
        for timb in [39, 19]:
            for i, temp in enumerate([7, 3, 1], start=1):
                la = nn.Conv2d(ins, outs * i ** 2,
                               kernel_size=(timb, temp),
                               stride=1,
                               padding=(int(np.floor(timb / 2)), int(np.floor(temp / 2))))
                self.layers.append(la)

        self.apf = AvgPoolFrequency()

    def forward(self, x):
        x = torch.cat([la(x) for la in self.layers], dim=1)
        x = self.apf(x)
        return x


class TemporalTimbreHead(nn.Module):
    def __init__(self, n_filter, p):
        super(TemporalTimbreHead, self).__init__()

        self.temporal = TemporalConv(1, n_filter)
        self.timbre = TimbreConv(1, n_filter)

        self.bn = nn.BatchNorm1d(168)
        self.do = nn.Dropout(p=p)

    def forward(self, x):
        x = torch.cat([self.temporal(x), self.timbre(x)], dim=1)
        x = self.do(F.elu(self.bn(x)))
        return x


class TinyTemporalTimbreCNN(nn.Module):
    def __init__(self):
        super(TinyTemporalTimbreCNN, self).__init__()
        self.conv = nn.Sequential(
            TemporalTimbreHead(n_filter=4, p=0.25),
            ConvBnMp(168, 64, p=0.25),  # 126 -> 42
            ConvBnMp(64, 64, p=0.25),  # 42 -> 14
            ConvBnMp(64, 128, p=0.25),  # 14 -> 4
            AverageMaxpoolCat(),
        )

        self.clf = nn.Sequential(
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.Dropout(p=0.5), nn.ELU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

        self.criterion = BCELoss()
        self.metric = BinaryClassificationMetrics

    def forward(self, batch):
        x = batch['audio']
        x = self.conv(x)
        x = self.clf(x)
        return {'target': x}


if __name__ == "__main__":
    model = TinyTemporalTimbreCNN()

    inputs = {'audio': torch.randn(2, 1, 40, 126)}
    print(model(inputs)['target'].shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
