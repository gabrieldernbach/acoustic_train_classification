import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, ins, outs):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, padding=1),
            nn.BatchNorm2d(outs),
            nn.ELU(inplace=True),
            nn.Conv2d(outs, outs, kernel_size=3, padding=1),
            nn.ELU(inplace=True)
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
        diffY = torch.tensor([skip.size()[2] - x.size()[2]])
        diffX = torch.tensor([skip.size()[3] - x.size()[3]])
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        return x


class OutConv(nn.Module):
    def __init__(self, ins, outs):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(ins, outs, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PoolFrequency(nn.Module):
    def __init__(self):
        super(PoolFrequency, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=(10, 1), stride=(5, 1))
        self.apool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        return self.apool(self.conv(x)).squeeze()


class Unet(nn.Module):
    def __init__(self, channels, classes, bilinear=True):
        super(Unet, self).__init__()
        self.channels = channels
        self.classes = classes
        self.bilinear = bilinear

        self.inc = DoubleConv(channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, classes)
        self.apool = PoolFrequency()

    def forward(self, x):
        enc1 = self.inc(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)
        x = self.down4(enc4)
        x = self.up1(x, enc4)
        x = self.up2(x, enc3)
        x = self.up3(x, enc2)
        x = self.up4(x, enc1)
        logits = self.apool(self.outc(x))
        return logits


import torch

if __name__ == "__main__":
    ins = torch.randn(20, 1, 40, 129)
    targets = torch.rand(20, 129).long()
    model = Unet(1, 2)
    outs = model(ins)

    # from torchsummary import summary
    # summary(model, input_size=(1, 40, 387))
    criterion = nn.CrossEntropyLoss()
    criterion(outs, targets)
    # criterion(outs, targets)
