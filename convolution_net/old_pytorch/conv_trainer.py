import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


class Learner:

    def __init__(self, model, criterion, optimizer, epochs, early_stop_patience, _run):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

        self.early_stopping = EarlyStopping(early_stop_patience, verbose=True)
        self.print_batch_interval = 20
        self._run = _run
        self.epoch = 0

    def fit(self, train_loader, val_loader):
        for self.epoch in range(1, self.epochs - 1):

            self._train_step(train_loader)
            validation_loss = self._validation_step(val_loader)

            if self.epoch % 5 == 0:
                self.evaluate(val_loader)

            self.early_stopping(validation_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    def _train_step(self, train_loader):
        self.model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.print_batch_interval == 0:
                print(f'Train Epoch {self.epoch} '
                      f'[{batch_idx * len(inputs)}/{len(train_loader.dataset)}] '
                      f'({100. * batch_idx / len(train_loader):.0f}%)\t'
                      f'Loss: {loss.item():.6f}')

            step = self.epoch + batch_idx / len(train_loader)
            self._run.log_scalar('train_loss', loss.item(), step=step)
            self._run.log_scalar('train_acc', self._accuracy(outputs, targets), step=step)

    def _validation_step(self, val_loader):
        running_loss, running_accuracy = [], []
        self.model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss.append(loss.item())
                running_accuracy.append(self._accuracy(outputs, targets))
            validation_loss = np.mean(running_loss)
            validation_accuracy = np.mean(running_accuracy)
            self._run.log_scalar('validation_loss', validation_loss, step=self.epoch)
            self._run.log_scalar('validation_accuracy', validation_accuracy, step=self.epoch)
        return validation_loss

    def evaluate(self, data_loader, label_threshold=0.35, prediction_threshold=0.5):
        label_list = []
        predictions_list = []

        self.model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loader):
                outputs = self.model(inputs.to(self.device))
                label_list.extend(labels.detach().numpy())
                predictions_list.extend(outputs.detach().cpu().numpy())

        labels = np.stack(label_list)
        predictions = np.stack(predictions_list)

        labels = labels > label_threshold
        auc = roc_auc_score(labels, predictions)
        f1 = f1_score(labels, predictions > prediction_threshold)
        confmat = confusion_matrix(labels, predictions > prediction_threshold)
        print(f'auc {auc:.3}, f1 {f1:.3}\n {confmat}')
        return auc, f1, confmat

    @staticmethod
    def _accuracy(outputs, labels):
        predictions = outputs.data > 0.5
        correct = (predictions == labels).sum().item()
        return correct / len(labels)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """save model in case of decrease of validation loss"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
