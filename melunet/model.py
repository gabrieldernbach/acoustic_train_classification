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
            'up': nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            'down': nn.MaxPool2d(2, 2),
        })

        self.conv = nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, padding=1),
            nn.BatchNorm2d(outs),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            resample[sample_mode],
        )

    def forward(self, x):
        return self.conv(x)


class PoolFrequency(nn.Module):
    def __init__(self, ins, outs):
        super(PoolFrequency, self).__init__()
        self.conv = nn.Conv2d(ins, outs, kernel_size=(10, 1), stride=(5, 1))
        self.apool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        return self.apool(self.conv(x)).squeeze()


class Unet(nn.Module):
    def __init__(self, num_filters, ins=1, outs=1, loss_ratio=0.1, p=0.1, **kwargs):
        super(Unet, self).__init__()

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
            [ConvBlock(fn[0] * 2, 8, 'none', p=p)]
        )

        self.poolfreq = PoolFrequency(8, outs)

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

        x = torch.sigmoid(self.poolfreq(x)).squeeze()
        return {'target': x}

    def pad2match(self, x, skip):
        h = skip.shape[2] - x.shape[2]
        w = skip.shape[3] - x.shape[3]
        x = F.pad(x, [w // 2, w - w // 2, h // 2, h - h // 2])
        return x


if __name__ == "__main__":
    batch = {
        'audio': torch.randn(20, 1, 40, 321),
        'target': torch.rand(20, 321).float()
    }

    # model = Unet(num_filters=[2, 4, 8, 16, 32, 64])
    model = Unet(num_filters=[16, 32, 64], p=0.4)

    model(batch)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
