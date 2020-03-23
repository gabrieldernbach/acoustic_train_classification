import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from main.callback import SegmentationMetrics
from main.loss import PooledSegmentationLoss


class Down(nn.Module):
    def __init__(self, ins, outs):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv1d(ins, outs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(outs),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, ins, outs):
        super(Up, self).__init__()

        self.upsample = nn.Upsample(scale_factor=3)
        self.conv = nn.Sequential(
            nn.Conv1d(ins, outs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(outs),
            nn.ELU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.pad2match(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

    def pad2match(self, x, skip):
        if x.shape != skip.shape:
            d = skip.shape[2] - x.shape[2]
            x = F.pad(x, [d // 2, d - d // 2])
        return x


class FixedWaveUnet(nn.Module):
    def __init__(self):
        super(FixedWaveUnet, self).__init__()

        self.inc = nn.Conv1d(1, 64, kernel_size=3, padding=1)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        self.outc = nn.Conv1d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = torch.sigmoid(self.outc(x))
        return x


class WaveUnet(nn.Module):
    def __init__(self, num_filters,
                 n_channels=1,
                 n_classes=1,
                 loss_ratio=0.5,
                 dropout_ratio=0.01,
                 **kwargs):
        super(WaveUnet, self).__init__()

        fn = np.array(num_filters)
        filter_down = list(zip(fn, fn[1:]))
        filter_down.append((fn[-1], fn[-1]))

        fnr = fn[::-1]
        filter_up = list(zip(fnr * 2, fnr[1:]))
        filter_up.append((fnr[-1] * 2, fnr[-1]))

        self.ins = nn.Conv1d(n_channels, fn[0], kernel_size=3, padding=1)
        self.encoder = nn.ModuleList([Down(i, o) for i, o in filter_down])
        self.decoder = nn.ModuleList([Up(i, o) for i, o in filter_up])
        self.outs = nn.Conv1d(fn[0], n_classes, kernel_size=3, padding=1)

        self.criterion = PooledSegmentationLoss(llambda=loss_ratio)
        self.metric = SegmentationMetrics
        self.do = nn.Dropout(p=dropout_ratio)

    def forward(self, batch):
        x = batch['audio']
        x = self.ins(x)

        skips = []
        for layer in self.encoder:
            skips.append(x)
            x = self.do(layer(x))

        for skip, layer in zip(reversed(skips), self.decoder):
            x = self.do(layer(x, skip))

        x = torch.sigmoid(self.outs(x)).squeeze()

        return {'target': x}


if __name__ == "__main__":
    model = WaveUnet([2, 4, 8, 16, 32, 64])
    batch = {'audio': torch.randn(50, 1, 40_960)}
    model(batch)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
