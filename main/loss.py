import torch

criterion = torch.nn.BCELoss()


class BCELoss:
    def __init__(self):
        self.criterion = torch.nn.BCELoss()

    def __call__(self, out, batch):
        return self.criterion(out['target'], batch['target'])


class PooledSegmentationLoss:
    def __init__(self, llambda=0.5):
        self.criterion = torch.nn.BCELoss()
        self.llambda = llambda

    def __call__(self, out, batch):
        out = out['target']
        batch = batch['target'].float()

        segmentation_loss = self.criterion(out, batch)

        classification_loss = self.criterion(
            (out.mean(dim=-1) > 0.05).float(),
            (batch.mean(dim=-1) > 0.05).float()
        )

        loss = self.llambda * segmentation_loss + (1 - self.llambda) * classification_loss
        return loss
