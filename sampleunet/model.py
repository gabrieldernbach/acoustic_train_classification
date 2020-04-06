import torch
import torch.nn as nn
import torch.nn.functional as F

from convolution_net.callback import SegmentationMetrics
from convolution_net.loss import PooledSegmentationLoss


class ConvBlock(nn.Module):
    def __init__(self, ins, outs, sample_mode, p=0.1):
        super(ConvBlock, self).__init__()

        resample = nn.ModuleDict({
            'none': nn.Identity(),
            'up': nn.Upsample(scale_factor=3),
            'down': nn.MaxPool1d(kernel_size=3, stride=3),
        })

        self.conv = nn.Sequential(
            nn.Conv1d(ins, outs, kernel_size=3, padding=1),
            nn.BatchNorm1d(outs),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            resample[sample_mode],
        )

    def forward(self, x):
        return self.conv(x)


class WaveUnet(nn.Module):
    def __init__(self, num_filters, ins=1, outs=1, loss_ratio=0.1, p=0.1, **kwargs):
        super(WaveUnet, self).__init__()

        fn = num_filters
        fdown = list(zip(fn, fn[1:]))
        fnr = list(reversed(fn))
        fup = list(zip(fnr, fnr[1:]))

        self.encoder = nn.ModuleList(
            [ConvBlock(ins, fn[0], 'none', p=p)] +
            [ConvBlock(i, o, 'down', p=p) for i, o in fdown]
        )
        self.decoder = nn.ModuleList(
            [ConvBlock(i * 2, o, 'up', p=p) for i, o in fup] +
            [ConvBlock(fn[0] * 2, outs, 'none', p=p)]
        )

        self.criterion = PooledSegmentationLoss(llambda=loss_ratio)
        self.metric = SegmentationMetrics

    def forward(self, batch):
        x = batch['audio']
        residual = []
        for layer in self.encoder:
            x = layer(x)
            # print(x.shape)
            residual.append(x)

        for residual, layer in zip(reversed(residual), self.decoder):
            x = self.pad2match(x, residual)
            x = torch.cat((residual, x), 1)
            # print(x.shape)
            x = layer(x)

        x = torch.sigmoid(x)
        return {'target': x}

    def pad2match(self, x, skip):
        if x.shape != skip.shape:
            d = skip.shape[2] - x.shape[2]
            x = F.pad(x, [d // 2, d - d // 2])
        return x


if __name__ == "__main__":
    model = WaveUnet([2, 4, 8, 16, 32, 64, 128], p=0.1)

    batch = {'audio': torch.randn(50, 1, 40_960)}
    # batch = {'audio': torch.randn(50, 1, 16_384)}
    model(batch)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
