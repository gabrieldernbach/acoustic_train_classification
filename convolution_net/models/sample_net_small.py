import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __call__(self, sample):
        return sample.view(sample.size(0), -1)


class PrintShape(nn.Module):
    def forward(self, sample):
        print(sample.shape)
        return sample


class Conv1x(nn.Module):
    def __init__(self, ins, outs, kernel_size, stride, pool):
        super(Conv1x, self).__init__()
        layers = [nn.Conv1d(ins, outs, kernel_size, stride),
                  nn.BatchNorm1d(outs),
                  nn.ReLU()]

        if pool is not None:
            layers.append(nn.MaxPool1d(pool))

        self.layers = nn.Sequential(*layers)

    def forward(self, sample):
        return self.layers(sample)


class SampleNet(nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()

        self.features = nn.Sequential(
            Conv1x(1, 16, kernel_size=64, stride=2, pool=8),
            Conv1x(16, 32, kernel_size=32, stride=2, pool=8),
            Conv1x(32, 64, kernel_size=16, stride=2, pool=None),
            Conv1x(64, 128, kernel_size=8, stride=2, pool=None),

            nn.AdaptiveAvgPool1d(1),
            Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.Dropout(p=0.25), nn.ReLU(),
            nn.Linear(64, 10), nn.BatchNorm1d(10), nn.Dropout(p=0.25), nn.ReLU(),
            nn.Linear(10, 1), nn.Sigmoid()
        )

    def forward(self, sample):
        out = self.features(sample)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    from torchsummary import summary

    model = SampleNet()
    summary(model, input_size=(1, 16000))
    print(model)
    samples = torch.randn(50, 1, 16000)
    targets = torch.randint(0, 2, (50, 1))
    outs = model(samples)
    print(outs.mean(dim=0).item(), outs.std(dim=0).item())
