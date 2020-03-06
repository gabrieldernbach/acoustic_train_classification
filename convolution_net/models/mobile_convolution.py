import torch.nn as nn


class PrintLayer(nn.Module):
    """debugging representation size in sequential layers"""

    def __call__(self, sample):
        print(sample.shape)
        return sample


class Flatten(nn.Module):
    """unsqueeze empty dimension after final pooling operation"""

    def __call__(self, sample):
        return sample.view(sample.size(0), -1)


class MBConvBlock(nn.Module):
    def __init__(self, ins, outs, dw_ratio, se_ratio, kernel_size=3, stride=1):
        super(MBConvBlock, self).__init__()
        padding = kernel_size // 2
        dw_depth = int(ins * dw_ratio)
        se_depth = max(1, int(dw_depth * se_ratio))

        self.use_res_connect = stride == 1 and ins == outs

        self.expansion = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(),
            nn.Conv2d(ins, dw_depth, kernel_size=1, bias=False),
        )

        self.depthwise = nn.Sequential(
            nn.BatchNorm2d(dw_depth),
            nn.ReLU(),
            nn.Conv2d(dw_depth, dw_depth, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=dw_depth, bias=False),
        )

        self.excitation = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(dw_depth, se_depth, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(se_depth, dw_depth, kernel_size=1),
        )

        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(dw_depth),
            nn.Conv2d(dw_depth, outs, kernel_size=1, bias=False),
        )

    def forward(self, inputs):
        out = self.expansion(inputs)
        out = self.depthwise(out)
        out = self.excitation(out) * out
        out = self.projection(out)
        if self.use_res_connect:
            out = inputs + out
        return out


class SeMobileNet(nn.Module):
    def __init__(self):
        super(SeMobileNet, self).__init__()

        self.features = nn.Sequential(
            MBConvBlock(1, 16, dw_ratio=2, se_ratio=.5, kernel_size=3, stride=1),
            MBConvBlock(16, 32, dw_ratio=2, se_ratio=.5, kernel_size=3, stride=(2, 1)),
            MBConvBlock(32, 64, dw_ratio=2, se_ratio=.5, kernel_size=5, stride=(2, 1)),
            MBConvBlock(64, 64, dw_ratio=2, se_ratio=.5, kernel_size=5, stride=1),
            MBConvBlock(64, 64, dw_ratio=2, se_ratio=.5, kernel_size=5, stride=1),
            MBConvBlock(64, 64, dw_ratio=2, se_ratio=.5, kernel_size=5, stride=1),
            MBConvBlock(64, 128, dw_ratio=2, se_ratio=.5, kernel_size=5, stride=(2, 1)),
            MBConvBlock(128, 128, dw_ratio=4, se_ratio=.5, kernel_size=5, stride=1),
            MBConvBlock(128, 128, dw_ratio=4, se_ratio=.5, kernel_size=5, stride=1),
            MBConvBlock(128, 256, dw_ratio=4, se_ratio=.5, kernel_size=5, stride=(2, 1)),
            MBConvBlock(256, 256, dw_ratio=4, se_ratio=.5, kernel_size=5, stride=1),
            MBConvBlock(256, 256, dw_ratio=4, se_ratio=.5, kernel_size=5, stride=1),
            MBConvBlock(256, 256, dw_ratio=4, se_ratio=.5, kernel_size=5, stride=1),
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.Dropout(p=0.5), nn.ReLU(),
            nn.Linear(512, 10), nn.BatchNorm1d(10), nn.Dropout(p=0.5), nn.ReLU(),
            nn.Linear(10, 1), nn.Sigmoid()
        )

    def forward(self, sample):
        out = self.features(sample)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    ins = torch.randn(50, 1, 40, 126)
    model = SeMobileNet()
    model(ins)
    print(model)
    summary(model, input_size=(1, 40, 126))
