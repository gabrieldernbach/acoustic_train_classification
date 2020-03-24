import torch
import torch.nn as nn
import torch.nn.functional as F

from main.callback import SegmentationMetrics
from main.loss import PooledSegmentationLoss


class DoubleConv(nn.Module):
    def __init__(self, ins, outs):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, padding=1),
            nn.BatchNorm2d(outs),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, ins, outs):
        super(Down, self).__init__()
        self.mp_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(ins, outs)
        )

    def forward(self, x):
        return self.mp_conv(x)


class Up(nn.Module):
    def __init__(self, ins, outs, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(ins // 2, ins // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(ins, outs)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.match_padding(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

    def match_padding(self, x, skip):
        dx = skip.size()[2] - x.size()[2]
        dy = skip.size()[3] - x.size()[3]
        x = F.pad(x, [dy // 2, dy - dy // 2,
                      dx // 2, dx - dx // 2])
        return x


class OutConv(nn.Module):
    def __init__(self, ins, outs):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(ins, outs, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PoolFrequency(nn.Module):
    def __init__(self, ins, outs):
        super(PoolFrequency, self).__init__()
        self.conv = nn.Conv2d(ins, outs, kernel_size=(10, 1), stride=(5, 1))
        self.apool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        return self.apool(self.conv(x)).squeeze()

class TinyUnet(nn.Module):
    def __init__(self, num_filters, channels=1, classes=1, dropout_ratio=0.001, bilinear=True, loss_ratio=0.5,
                 **kwargs):
        super(TinyUnet, self).__init__()
        self.channels = channels
        self.classes = classes
        self.bilinear = bilinear

        self.do = nn.Dropout2d(p=dropout_ratio)
        d = num_filters
        self.inc = DoubleConv(channels, d[0])
        self.down1 = Down(d[0], d[1])
        self.down2 = Down(d[1], d[2])
        self.down3 = Down(d[2], d[2])
        self.up1 = Up(d[2] + d[2], d[1], bilinear)
        self.up2 = Up(d[1] + d[1], d[0], bilinear)
        self.up3 = Up(d[0] + d[0], d[0], bilinear)
        self.outc = OutConv(d[0], classes)
        self.apool = PoolFrequency(classes, classes)

        self.criterion = PooledSegmentationLoss(llambda=loss_ratio)
        self.metric = SegmentationMetrics

    def forward(self, batch):
        enc1 = self.do(self.inc(batch['audio']))
        enc2 = self.do(self.down1(enc1))
        enc3 = self.do(self.down2(enc2))
        x = self.do(self.down3(enc3))
        x = self.do(self.up1(x, enc3))
        x = self.do(self.up2(x, enc2))
        x = self.do(self.up3(x, enc1))
        x = self.apool(self.outc(x))
        x = torch.sigmoid(x)
        return {'target': x}


if __name__ == "__main__":
    batch = {
        'audio': torch.randn(20, 1, 40, 321),
        'target': torch.rand(20, 321).float()
    }
    model = TinyUnet(num_filters=[32, 64, 128, 256])
    outs = model(batch)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
