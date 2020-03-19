import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearExtended(nn.Module):
    """Linear - Batchnorm - Dropout - Relu"""

    def __init__(self, ins, outs, drop_out_rate=0.5):
        super(LinearExtended, self).__init__()
        self.fc = nn.Linear(ins, outs)
        self.bn = nn.BatchNorm1d(outs)
        self.do = nn.Dropout(p=drop_out_rate)

    def forward(self, x):
        return F.relu(self.do(self.bn(self.fc(x))))


class Flatten(nn.Module):
    """unsqueeze empty dimension after final pooling operation"""

    def __call__(self, sample):
        return sample.view(sample.size(0), -1)


class TimeBlock1D(nn.Module):
    def __init__(self, ins, outs):
        super(TimeBlock1D, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.Conv2d(ins, outs, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.ReLU()
        )

        self.upsample = None
        if ins != outs:
            self.upsample = nn.Conv2d(ins, outs, kernel_size=1)

    def forward(self, x):
        identity = x
        out = self.layers(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        return out + identity


class MultiscaleBlock(nn.Module):
    def __init__(self, ins, cardinality, outs, pool=2):
        super(MultiscaleBlock, self).__init__()
        self.bn = nn.BatchNorm2d(ins)  # todo: max pool and average pool
        self.pool = nn.AvgPool2d(kernel_size=(pool, pool))  # todo: 3 6 9 12, weight norm,
        self.conv35 = nn.Conv2d(ins, cardinality, kernel_size=(1, 35), stride=1, padding=(0, 17))
        self.conv65 = nn.Conv2d(ins, cardinality, kernel_size=(1, 65), stride=1, padding=(0, 32))
        self.conv95 = nn.Conv2d(ins, cardinality, kernel_size=(1, 95), stride=1, padding=(0, 47))
        self.conv125 = nn.Conv2d(ins, cardinality, kernel_size=(1, 125), stride=1, padding=(0, 62))
        self.pointwise = nn.Conv2d(4 * cardinality, outs, kernel_size=1)

    def forward(self, x):
        x = self.pool(self.bn(x))
        skip = x
        x35 = F.relu(self.conv35(x))
        x65 = F.relu(self.conv65(x))
        x95 = F.relu(self.conv95(x))
        x125 = F.relu(self.conv125(x))
        x = torch.cat([x35, x65, x95, x125], dim=1)
        x = self.pointwise(x)
        return x + skip


class MeanStd(nn.Module):
    def __init__(self):
        super(MeanStd, self).__init__()

    def forward(self, x):
        print('\nmean\n', x.mean(dim=0))
        print('\nstd\n', x.std(dim=0))
        return x


class TimeFilterNet(nn.Module):

    def __init__(self):
        super(TimeFilterNet, self).__init__()
        self.shortfilers = nn.Sequential(
            TimeBlock1D(1, 16),
            TimeBlock1D(16, 16),
            TimeBlock1D(16, 16),
            TimeBlock1D(16, 16),
            TimeBlock1D(16, 32),
            TimeBlock1D(32, 64)
        )
        self.longfilters = nn.Sequential(
            MultiscaleBlock(ins=64, cardinality=24, outs=64, pool=2),  # 20 - 10
            MultiscaleBlock(ins=64, cardinality=24, outs=64, pool=2),  # 10 - 5
            MultiscaleBlock(ins=64, cardinality=24, outs=64, pool=2),  # 5 2
            MultiscaleBlock(ins=64, cardinality=24, outs=64, pool=2),
            nn.AdaptiveMaxPool2d(1),
            Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            LinearExtended(64, 512),
            nn.Dropout(p=0.5),
            LinearExtended(512, 64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.shortfilers(x)
        x = self.longfilters(x)
        x = self.classifier(x)
        return x


class MultiresBlockDilated(nn.Module):
    """
    Multiresolution Block with dilated convolutions

    Notes on dilated convolution arithmetic:
    pytorch only allows for symmetric padding, therefore we can only provide
    convolution 'same' when kernel size k % 2 != 0.
    The necessary padding without dilation is floor(k/2).
    When adding dilation we need to multiply that value times the dilation factor
    e.g. dilation=3 and kernel=5 results in [3 * floor(5/2)] = 6
    """

    def __init__(self, ins, cardinality, outs, pool=2):
        super(MultiresBlockDilated, self).__init__()

        self.avp = nn.AvgPool2d(kernel_size=(pool, 1))  # todo : max pool freq, avg pool time
        self.bn = nn.BatchNorm2d(ins)
        self.f1 = nn.Conv2d(ins, cardinality, kernel_size=(1, 5), stride=1, dilation=1, padding=(0, 2))
        self.f2 = nn.Conv2d(ins, cardinality, kernel_size=(1, 5), stride=1, dilation=2, padding=(0, 4))
        self.f3 = nn.Conv2d(ins, cardinality, kernel_size=(1, 5), stride=1, dilation=3, padding=(0, 6))
        self.f4 = nn.Conv2d(ins, cardinality, kernel_size=(1, 5), stride=1, dilation=4, padding=(0, 8))
        self.pointwise = nn.Conv2d(cardinality * 4, outs, kernel_size=1)

    def forward(self, x):
        x = self.bn(self.avp(x))
        skip = x
        x1 = self.f1(x)
        x2 = self.f2(x)
        x3 = self.f3(x)
        x4 = self.f4(x)
        x = self.pointwise(torch.cat([x1, x2, x3, x4], dim=1))
        return x + skip


class TimeFilterNetDilated(nn.Module):
    def __init__(self):
        super(TimeFilterNetDilated, self).__init__()
        self.shortfilers = nn.Sequential(
            TimeBlock1D(1, 16),
            TimeBlock1D(16, 16),
            TimeBlock1D(16, 32)
        )

        self.longfilters = nn.Sequential(
            MultiresBlockDilated(32, 24, 32),
            MultiresBlockDilated(32, 24, 32),
            MultiresBlockDilated(32, 24, 32),
            MultiresBlockDilated(32, 24, 32),
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.Dropout(p=0.5),
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.Dropout(p=0.25),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.Dropout(p=0.25),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.shortfilers(x)
        x = self.longfilters(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    ins = torch.randn(500, 1, 40, 126)
    model = TimeFilterNet()
    # print(model)
    summary(model, input_size=(1, 40, 126))

    outs = model(ins)
    print(outs.mean())
    print(outs.var())
