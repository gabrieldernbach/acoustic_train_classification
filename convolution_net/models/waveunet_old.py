import torch
import torch.nn.functional as F
from torch import nn

from convolution_net.callback import SegmentationMetrics
from convolution_net.loss import PooledSegmentationLoss


class DwsConv(nn.Module):
    def __init__(self, ins, outs):
        super(DwsConv, self).__init__()
        hidden = 8 * outs
        self.conv = nn.Sequential(
            nn.Conv1d(ins, hidden, kernel_size=1),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.Conv1d(hidden, outs, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, ins, outs, sample_mode, conv_mode):
        super(ConvBlock, self).__init__()

        resample = nn.ModuleDict({
            'down': nn.MaxPool1d(kernel_size=3, stride=3),
            'up': nn.Upsample(scale_factor=3),
        })

        conv = nn.ModuleDict({
            'depth_wise': nn.Conv1d(ins, outs, kernel_size=3, padding=1),
            'depth_wise_separable': DwsConv(ins, outs),
        })

        self.block = nn.Sequential(
            conv[conv_mode],
            nn.BatchNorm1d(outs),
            nn.ELU(),
            resample[sample_mode],
        )

    def forward(self, sample):
        return self.block(sample)


class WaveUnet(nn.Module):
    def __init__(self, num_filters, ins=1, outs=1, loss_ratio=0.5, dw_mode='depth_wise', **kwargs):
        super(WaveUnet, self).__init__()

        fn = num_filters
        fdown = list(zip(fn, fn[1:]))
        fnr = list(reversed(fn))
        fup = list(zip(fnr, fnr[1:]))

        self.encoder = nn.ModuleList(
            [nn.Conv1d(ins, fn[0], kernel_size=3, padding=1)] +
            [ConvBlock(i, o, 'down', dw_mode) for i, o in fdown]
        )
        self.decoder = nn.ModuleList(
            [ConvBlock(i * 2, o, 'up', dw_mode) for i, o in fup] +
            [nn.Conv1d(fn[0] * 2, outs, kernel_size=3, padding=1)]
        )

        self.criterion = PooledSegmentationLoss(llambda=loss_ratio)
        self.metric = SegmentationMetrics

    def forward(self, batch):
        x = batch['audio']
        residual = []
        for layer in self.encoder:
            x = layer(x)
            residual.append(x)

        for residual, layer in zip(reversed(residual), self.decoder):
            x = self.pad_match(x, residual)
            x = torch.cat((residual, x), 1)
            x = layer(x)

        x = torch.sigmoid(x).squeeze()
        return {'target': x}

    def pad_match(self, x, residual):
        if x.shape != residual.shape:
            l = residual.shape[2] - x.shape[2]
            x = F.pad(x, [l // 2, l - l // 2])
        return x


if __name__ == "__main__":
    # model = WaveUnet([2, 4, 8, 16, 32, 64, 128])
    model = WaveUnet([2, 4, 8, 16, 32, 64])
    # from torchsummary import summary
    # summary(model=model, input_size=(1, 40960))

    print(model)
    batch = torch.randn(50, 1, 40_960)
    model(batch)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
