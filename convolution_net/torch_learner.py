import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, average_precision_score, accuracy_score
from sklearn.metrics import precision_score, recall_score


class Learner:

    def __init__(self, model, criterion, optimizer, scheduler, db_observer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.db_observer = db_observer
        self.epoch = 0

    def fit(self, train_loader, validation_loader, epochs):
        for self.epoch in range(epochs):
            self.train(train_loader)
            aps = self.validate(validation_loader)

            self.scheduler.step(aps)

    def train(self, data_loader):
        outs_collect = []
        targets_collect = []
        for i, (samples, targets) in enumerate(data_loader):
            samples, targets = samples.to(self.device), targets.to(self.device)
            samples, targets = self.mixup(samples, targets, alpha=0.8)
            self.model.train()
            self.optimizer.zero_grad()
            outs = self.model(samples)
            loss = self.criterion(outs, targets)
            loss.backward()
            self.optimizer.step()

            per = (i + 1) / len(data_loader)
            print(f'\repoch: {self.epoch}, iter {i + 1:3} of {len(data_loader):3}'
                  f' - {(per * 100):5.1f}%, loss of {loss:.9}', end='')
            self.db_observer.log_scalar('train_loss', loss.item(), step=self.epoch + per)
            outs_collect.append(outs.detach().cpu().numpy())
            targets_collect.append(targets.detach().cpu().numpy())
        self.metrics('\ntrain', outs_collect, targets_collect)

    def validate(self, data_loader):
        with torch.no_grad():
            loss_collected = []
            outs_collect = []
            targets_collect = []
            for i, (samples, targets) in enumerate(data_loader):
                samples, targets = samples.to(self.device), targets.to(self.device)
                self.model.eval()
                outs = self.model(samples)
                loss = self.criterion(outs, targets)
                per = (i + 1) / len(data_loader)

                loss_collected.append(loss.item())
                outs_collect.append(outs.detach().cpu().numpy())
                targets_collect.append(targets.detach().cpu().numpy())

            mean_loss = np.array(loss_collected).mean()
            self.db_observer.log_scalar('validation_loss', mean_loss, step=self.epoch)
            aps = self.metrics('val', outs_collect, targets_collect)
            return aps

    def mixup(self, samples, targets, alpha=0.8):
        idx = torch.randperm(samples.size(0))
        if self.device == 'cuda':
            idx.to(self.device)

        samples_perm, targets_perm = samples[idx], targets[idx]
        llambda = np.random.beta(alpha, alpha)

        mixed_samples = llambda * samples + (1 - llambda) * samples_perm
        mixed_targets = llambda * targets + (1 - llambda) * targets_perm
        return mixed_samples, mixed_targets

    def metrics(self, mode, outs_collected, targets_collected):
        outs_collected = np.concatenate(outs_collected)
        targets_collected = np.concatenate(targets_collected)

        # soft predictions
        targets_collected = targets_collected > .125
        auc = roc_auc_score(targets_collected, outs_collected)
        aps = average_precision_score(targets_collected, outs_collected)

        # hard predictions
        outs_collected = outs_collected > .5
        precision = precision_score(targets_collected, outs_collected, zero_division=False)
        recall = recall_score(targets_collected, outs_collected, zero_division=False)
        f1 = f1_score(targets_collected, outs_collected, zero_division=False)
        acc = accuracy_score(targets_collected, outs_collected)
        confmat = confusion_matrix(targets_collected, outs_collected)
        tn, fp, fn, tp = confmat.ravel()
        print(f'{mode:5} - tp:{tp:5}, fp:{fp:5}, tn:{tn:5}, fn:{fn:5}, '
              f'precision:{precision:5.2}, recall:{recall:5.2}, f1:{f1:5.2}, '
              f'acc:{acc:5.2}, auc:{auc:5.2}, aps:{aps:5.2}')
        self.db_observer.log_scalar(f'{mode}_f1', f1, step=self.epoch)
        self.db_observer.log_scalar(f'{mode}_auc', auc, step=self.epoch)
        self.db_observer.log_scalar(f'{mode}_aps', aps, step=self.epoch)
        return aps
