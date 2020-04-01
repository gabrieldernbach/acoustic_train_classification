from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from main.callback import SegmentationMetrics
from main.loss import PooledSegmentationLoss


class ConvBlock(nn.Module):
    def __init__(self, ins, outs):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, padding=1),
            nn.BatchNorm2d(outs),
            nn.ELU(),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, ins, outs):
        super(Down, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = ConvBlock(ins, outs)

    def forward(self, x):
        return self.conv(self.down(x))


class Up(nn.Module):
    def __init__(self, ins, outs):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(ins, outs)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.match_padding(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

    def match_padding(self, x, skip):
        dx = skip.size()[2] - x.size()[2]
        dy = skip.size()[3] - x.size()[3]
        x = F.pad(x, [dy // 2, dy - dy // 2,
                      dx // 2, dx - dx // 2])
        return x


class TemporalFilter(nn.Module):
    def __init__(self, ins, outs):
        super(TemporalFilter, self).__init__()
        depth = 9
        self.ap = nn.AdaptiveAvgPool2d((1, None))
        self.convs = nn.ModuleList(
            [nn.Conv1d(ins, ins, kernel_size=7, padding=7 // 2) for _ in range(depth)]
        )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(ins * depth),
            nn.ELU(),
            nn.Conv1d(ins * depth, outs, kernel_size=1),
            nn.BatchNorm1d(outs),
            nn.ELU(),
        )

    def forward(self, x):
        start = time()
        x = self.ap(x).squeeze(dim=2)
        x = torch.cat([layer(x) for layer in self.convs], dim=1)
        x = self.fc(x)
        # print(f'finished time filter after {time() - start}')
        return x


class TimbreFilter(nn.Module):
    def __init__(self, ins, outs):
        super(TimbreFilter, self).__init__()

        self.compress = nn.Conv2d(ins, ins // 4, kernel_size=1)
        self.convs = nn.ModuleList()
        for temp in [1, 3, 5]:
            for timb in [7, 13, 25]:
                self.convs.append(
                    nn.Conv2d(ins // 4, ins // 4, kernel_size=(timb, temp), padding=(timb // 2, temp // 2)))

        self.fc = nn.Sequential(
            nn.BatchNorm2d(ins // 4 * 9),
            nn.ELU(),
            nn.Conv2d(ins // 4 * 9, outs, kernel_size=1),
            nn.BatchNorm2d(outs),
            nn.ELU(),
            nn.AdaptiveMaxPool2d((1, None)),
        )

    def forward(self, x):
        start = time()
        x = self.compress(x)
        x = torch.cat([layer(x) for layer in self.convs], dim=1)
        x = self.fc(x).squeeze(dim=2)
        # print(f'finished timbre filter after {time() - start}')
        return x


class TemporalTimbreHead(nn.Module):
    def __init__(self, ins, outs):
        super(TemporalTimbreHead, self).__init__()

        self.temporal = TemporalFilter(ins, ins)
        self.timbre = TimbreFilter(ins, ins)
        self.classifier = nn.Sequential(
            nn.Conv1d(ins * 2, ins * 2, kernel_size=1),
            nn.BatchNorm1d(ins * 2),
            nn.ELU(),
            nn.Conv1d(ins * 2, outs, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = torch.cat([self.temporal(x), self.timbre(x)], dim=1)
        return self.classifier(x).squeeze(dim=1)


class TTUNet(nn.Module):
    def __init__(self, num_filters, n_channel=1, n_classes=1, loss_ratio=0.5):
        super(TTUNet, self).__init__()

        filter_down, filter_up = self.filter_tuples(num_filters)

        self.ins = ConvBlock(n_channel, num_filters[0])
        self.encoder = nn.ModuleList([Down(i, o) for i, o in filter_down])
        self.decoder = nn.ModuleList([Up(i, o) for i, o in filter_up])
        self.outs = ConvBlock(num_filters[0], num_filters[0])

        self.tt_head = TemporalTimbreHead(num_filters[0], n_classes)

        self.criterion = PooledSegmentationLoss(llambda=loss_ratio)
        self.metric = SegmentationMetrics

    def forward(self, batch):
        x = batch['audio']
        start = time()
        x = self.ins(x)

        skips = []
        for layer in self.encoder:
            skips.append(x)
            x = layer(x)

        for skip, layer in zip(reversed(skips), self.decoder):
            x = layer(x, skip)
        # print(f'finished unet part after {time() - start}')

        x = self.tt_head(x)
        return {'target': x}

    def filter_tuples(self, num_filters):
        fn = np.array(num_filters)
        filter_down = list(zip(fn, fn[1:]))
        filter_down.append((fn[-1], fn[-1]))

        fnr = fn[::-1]
        filter_up = list(zip(fnr * 2, fnr[1:]))
        filter_up.append((fnr[-1] * 2, fnr[-1]))
        return filter_down, filter_up


if __name__ == "__main__":
    batch = {
        'audio': torch.randn(20, 1, 40, 321),
        'target': torch.rand(20, 321).float()
    }
    input = torch.randn(20, 1, 40, 321)
    # model = TTUNet(num_filters=[16, 32, 64])
    model = TTUNet(num_filters=[8, 16, 32])
    print(model)
    outs = model(batch)
    print(outs['target'].shape)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
