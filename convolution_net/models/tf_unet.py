from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from convolution_net.callback import SegmentationMetrics
from convolution_net.loss import PooledSegmentationLoss


class DoubleConv(nn.Module):
    def __init__(self, ins, outs, sample_mode, p=0.1):
        super(DoubleConv, self).__init__()

        resample = nn.ModuleDict({
            'up': nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            'down': nn.MaxPool2d(2, 2),
        })

        self.double_conv = nn.Sequential(

            nn.Conv2d(ins, outs, 3, padding=1),
            nn.BatchNorm2d(outs),
            nn.ELU(inplace=True),
            nn.Dropout(p=p),

            nn.Conv2d(outs, outs, 3, padding=1),
            nn.BatchNorm2d(outs),
            nn.ELU(inplace=True),
            nn.Dropout(p=p),

            resample[sample_mode],
        )

    def forward(self, x):
        return self.double_conv(x)


class TemporalLayer(nn.Module):
    def __init__(self, ins):
        super(TemporalLayer, self).__init__()
        n_layer, fn = 5, 8  # resuls in 40 output channels

        self.reduce_channel = nn.Conv1d(ins, fn, kernel_size=1, stride=1)
        self.temporal = nn.ModuleList([nn.Conv1d(fn, fn, kernel_size=3, stride=1, padding=1) for _ in range(n_layer)])

    def forward(self, x):
        x = x.mean(dim=2).squeeze(dim=2)
        x = self.reduce_channel(x)

        residual = []
        for layer in self.temporal:
            x = layer(x)
            residual.append(x)

        x = torch.cat(residual, dim=1)
        return x


class TimbreLayer(nn.Module):
    def __init__(self, ins):
        super(TimbreLayer, self).__init__()

        self.layers = nn.ModuleList()  # will have 28 output channels
        for timb in [31, 19]:
            for i, temp in enumerate([7, 3, 1], start=1):
                la = nn.Conv2d(ins, 2 * 2 ** i,
                               kernel_size=(timb, temp),
                               stride=1,
                               padding=(int(np.floor(timb / 2)), int(np.floor(temp / 2))))
                self.layers.append(la)

    def forward(self, x):
        x = torch.cat([la(x) for la in self.layers], dim=1)
        x = x.mean(dim=2).squeeze(dim=2)
        return x


class TemporalTimbreBlock(nn.Module):
    def __init__(self, ins):
        super(TemporalTimbreBlock, self).__init__()
        self.temporal = TemporalLayer(ins=ins)
        self.timbre = TimbreLayer(ins=ins)

        cfg = [96, 32, 32, 32]
        cfg = list(zip(cfg, cfg[1:]))
        self.dense = nn.ModuleList([nn.Conv1d(i, o, kernel_size=3, padding=1) for i, o in cfg])

        self.clf = nn.Conv1d(96, 1, kernel_size=1)

    def forward(self, x):
        x = torch.cat([self.temporal(x), self.timbre(x)], dim=1)
        residual = []
        for layer in self.dense:
            x = layer(x)
            residual.append(x)
        x = self.clf(torch.cat(residual, dim=1))
        return x


class TFUNet(nn.Module):
    def __init__(self, num_filters, ins=1, outs=8, loss_ratio=0.5, **kwargs):
        super(TFUNet, self).__init__()

        # fn = [8, 16, 32]
        fn = num_filters
        fdown = list(zip(fn, fn[1:]))
        fnr = list(reversed(fn))
        fup = list(zip(fnr, fnr[1:]))

        self.encoder = nn.ModuleList(
            [nn.Conv2d(ins, fn[0], kernel_size=3, stride=1, padding=1)] +
            [DoubleConv(i, o, 'down') for i, o in fdown]
        )
        self.decoder = nn.ModuleList(
            [DoubleConv(i * 2, o, 'up') for i, o in fup] +
            [nn.Conv2d(fn[0] * 2, outs, kernel_size=3, padding=1)]
        )

        self.outs = TemporalTimbreBlock(fn[0])

        self.criterion = PooledSegmentationLoss(llambda=loss_ratio)
        self.metric = SegmentationMetrics

    def forward(self, batch):
        x = batch['audio']
        residual = []
        for layer in self.encoder:
            x = layer(x)
            residual.append(x)

        for residual, layer in zip(reversed(residual), self.decoder):
            x = self.pad2match(x, residual)
            x = torch.cat((residual, x), 1)
            x = layer(x)

        x = torch.sigmoid(self.outs(x)).squeeze()
        return {'target': x}

    def pad2match(self, x, residual):
        if x.shape != residual.shape:
            h = residual.shape[2] - x.shape[2]
            w = residual.shape[3] - x.shape[3]
            x = F.pad(x, [w // 2, w - w // 2, h // 2, h - h // 2])
        return x


if __name__ == "__main__":
    model = TFUNet([8, 16, 32])
    print(model)
    batch = {'audio': torch.randn(50, 1, 40, 321)}
    print('starting')
    start = time()
    model(batch)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
