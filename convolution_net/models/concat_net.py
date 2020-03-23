import torch.nn as nn

from convolution_net.callback import BinaryClassificationMetrics
from convolution_net.loss import BCELoss


class ConvMp(nn.Module):
    def __init__(self, ins, outs):
        super(ConvMp, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ins, outs, kernel_size=3, padding=1),
            nn.BatchNorm2d(outs),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def __call__(self, x):
        return self.block(x)


class AverageMaxPool(nn.Module):
    def __init__(self):
        super(AverageMaxPool, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], dim=1).squeeze()


class ConcatLinearLayer(nn.Module):
    def __init__(self, ins, outs, n_context, rank=32):
        super(ConcatLinearLayer, self).__init__()
        self.linear = nn.Linear(ins + rank, outs)
        self.embed = nn.Sequential(
            nn.Embedding(n_context, rank),
            nn.ReLU(),
            nn.Linear(rank, rank // 2),
            nn.ReLU(),
            nn.Linear(rank // 2, rank)
        )

    def forward(self, batch):
        context_embedding = self.embed(batch['speed_id']).squeeze()
        sample_w_context = torch.cat((batch['audio'], context_embedding), dim=1)
        return self.linear(sample_w_context)


class ConcatLinearBlock(nn.Module):
    def __init__(self, ins, outs, n_context, p=0.5):
        super(ConcatLinearBlock, self).__init__()
        self.block = nn.Sequential(
            ConcatLinearLayer(ins, outs, n_context),
            nn.BatchNorm1d(outs),
            nn.Dropout(p=p),
            nn.ELU(),
        )

    def forward(self, batch):
        batch['audio'] = self.block(batch)
        return batch


class TinyConcatCNN(nn.Module):
    """Convolutional Network of 3x3 filters"""

    def __init__(self):
        super(TinyConcatCNN, self).__init__()
        self.features = nn.Sequential(
            ConvMp(1, 8),  # 40 x 126
            ConvMp(8, 16),  # 20 x 63
            ConvMp(16, 32),  # 10 x 31
            ConvMp(32, 64),  # 5 x 15
            ConvMp(64, 128),  # 2 x 7
            AverageMaxPool(),  # 1 x 3 -> 1 x 1
        )
        self.classifier = nn.Sequential(
            ConcatLinearBlock(256, 128, n_context=10),
            ConcatLinearBlock(128, 128, n_context=10),
            ConcatLinearLayer(128, 1, n_context=10),
            nn.Sigmoid()
        )

        self.criterion = BCELoss()
        self.metric = BinaryClassificationMetrics

    def forward(self, batch):
        batch['audio'] = self.features(batch['audio'])
        x = self.classifier(batch)
        return {'target': x}


if __name__ == '__main__':
    import torch

    audio = torch.randn(500, 1, 40, 126)
    speed_id = torch.randint(0, 10, (500, 1))
    batch = {'audio': audio, 'speed_id': speed_id}

    model = TinyConcatCNN()
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
