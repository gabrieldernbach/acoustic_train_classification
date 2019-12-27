import numpy as np
import torch


class Trainer:

    def __init__(self, model, criterion, optimizer, epochs, callback, early_stop_patience, early_stop_verbose):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.callback = callback
        self.early_stopping = EarlyStopping(early_stop_patience, early_stop_verbose)

        # report state for logging callback
        self.current_epoch = 0
        self.training_loss = 0.
        self.training_accuracy = 0.
        self.validation_loss = 0.
        self.validation_accuracy = 0.
        self.recent_test_acc = 0.

    def fit(self, train_loader, val_loader):
        for self.current_epoch in range(self.epochs):

            self._train_step(train_loader)
            self._validation_step(val_loader)

            self.early_stopping(self.validation_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            if self.callback is not None:
                self.callback(self)

    def _train_step(self, train_loader):
        self.training_loss, self.training_accuracy = 0.0, 0.0
        self.model.train()
        # for i, (inputs, labels, context) in enumerate(train_loader):
        for i, (inputs, labels) in enumerate(train_loader):
            self.optimizer.zero_grad()
            # outputs = self.model(inputs, context)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            accuracy = self._accuracy(outputs, labels)
            self.training_loss += loss.item()
            self.training_accuracy += 1 / (i + 1) * (accuracy - self.training_accuracy)
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 10 == 0:
                print('[epoch %d, batch %5d] loss: %.9f' %
                      (self.current_epoch + 1, i + 1, self.training_loss / 2000))

    def _validation_step(self, val_loader):
        self.validation_loss, self.validation_accuracy = 0.0, 0.0
        self.model.eval()
        with torch.no_grad():
            # for i, (inputs, labels, context) in enumerate(val_loader):
            for i, (inputs, labels) in enumerate(val_loader):
                # outputs = self.model(inputs, context)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                accuracy = self._accuracy(outputs, labels)
                self.validation_loss += loss.item()
                self.validation_accuracy += 1 / (i + 1) * (accuracy - self.validation_accuracy)

    def _accuracy(self, outputs, labels):
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        return correct / len(labels)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

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
