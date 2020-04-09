import torch
import torch.nn as nn
import torch.nn.functional as F

from convolution_net.callback import SegmentationMetrics
from convolution_net.loss import PooledSegmentationLoss


class ConvBlock(nn.Module):

    def __init__(self, ins, outs, sample_mode, dropout):
        super(ConvBlock, self).__init__()

        resample = nn.ModuleDict({
            'none': nn.Identity(),
            'up': nn.Upsample(scale_factor=3),
            'down': nn.MaxPool1d(kernel_size=3, stride=3),
        })
        self.resample = resample[sample_mode]
        self.skip = nn.Conv1d(ins, outs, kernel_size=1)

        depth = outs * 8
        self.conv = nn.Sequential(
            nn.Conv1d(ins, depth, kernel_size=1, bias=False),
            nn.BatchNorm1d(depth),
            nn.ReLU(),

            nn.Conv1d(depth, depth, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(depth),
            nn.ReLU(),

            nn.Conv1d(depth, outs, kernel_size=1, bias=False),
            nn.BatchNorm1d(outs),
            nn.Dropout(p=dropout),
        )

    def forward(self, sample):
        return self.resample(self.conv(sample) + self.skip(sample))


class SampleUnet(nn.Module):
    def __init__(self, num_filters,
                 in_channel=1, out_classes=1,
                 dropout=0.1, loss_ratio=0.1):
        super(SampleUnet, self).__init__()

        fn = num_filters
        fdown = list(zip(fn, fn[1:]))
        fnr = list(reversed(fn))
        fup = list(zip(fnr, fnr[1:]))

        self.encoder = nn.ModuleList(
            [ConvBlock(in_channel, fn[0], 'none', dropout=dropout)] +
            [ConvBlock(i, o, 'down', dropout=dropout) for i, o in fdown]
        )
        self.decoder = nn.ModuleList(
            [ConvBlock(i * 2, o, 'up', dropout=dropout) for i, o in fup] +
            [ConvBlock(fn[0] * 2, out_classes, 'none', dropout=dropout)]
        )

        self.criterion = PooledSegmentationLoss(llambda=loss_ratio)
        self.metric = SegmentationMetrics

    def forward(self, batch):
        x = batch['audio']
        # x = batch
        residual = []
        for layer in self.encoder:
            x = layer(x)
            residual.append(x)

        for residual, layer in zip(reversed(residual), self.decoder):
            x = self.pad2match(x, residual)
            x = torch.cat((residual, x), 1)
            x = layer(x)

        x = torch.sigmoid(x).squeeze(dim=1)
        return {'target': x}
        # return x

    def pad2match(self, x, skip):
        if x.shape != skip.shape:
            d = skip.shape[2] - x.shape[2]
            x = F.pad(x, [d // 2, d - d // 2])
        return x


if __name__ == "__main__":
    model = SampleUnet([2, 4, 8, 16, 32, 64, 128, 256], dropout=0.1)
    # model = SampleUnet([2, 4, 8], dropout=0.1)
    # summary(model, input_size=(1, 40_960))
    # exit()

    # batch = {'audio': torch.randn(50, 1, 40_960)}
    batch = {'audio': torch.randn(50, 1, 16_384)}
    print(model(batch)['target'].shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
