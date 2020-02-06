import re

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, average_precision_score, accuracy_score
from sklearn.metrics import precision_score, recall_score


class Callback:
    """
    The base class inherited by callbacks.
    """

    def on_fit_begin(self, **kwargs): pass

    def on_epoch_begin(self, **kwargs): pass

    def on_phase_begin(self, **kwargs): pass

    def on_train_load(self, data, **kwargs): return data

    def on_batch_end(self, **kwargs): pass

    def on_phase_end(self, **kwargs): pass

    def on_epoch_end(self, **kwargs): pass


class CallbackHandler:
    """
    Manages a list of callbacks
    """

    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.named_callbacks = {self.snake_case(self.classname(cb)): cb for cb in self.callbacks}

    def __getitem__(self, item):
        item = self.snake_case(item)
        if item in self.named_callbacks:
            return self.named_callbacks[item]
        raise KeyError(f'callback name is not found: {item}')

    def __setitem__(self, key, value):
        key = self.snake_case(key)
        self.named_callbacks[key] = value
        self.__init__(self.named_callbacks.values())

    def on_fit_begin(self, **kwargs):
        for callback in self.callbacks:
            callback.on_fit_begin(**kwargs)

    def on_epoch_begin(self, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(**kwargs)

    def on_phase_begin(self, **kwargs):
        for callback in self.callbacks:
            callback.on_phase_begin(**kwargs)

    def on_train_load(self, data, **kwargs):
        for callback in self.callbacks:
            data = callback.on_train_load(data, **kwargs)
        return data

    def on_batch_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(**kwargs)

    def on_phase_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_phase_end(**kwargs)

    def on_epoch_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(**kwargs)

    @staticmethod
    def classname(obj):
        return obj.__class__.__name__

    @staticmethod
    def snake_case(string):
        s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


class Mixup(Callback):
    """
    Mixup Data Augmentation
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.device = None

    def on_fit_begin(self, learner, **kwargs):
        self.device = learner.device

    def on_train_load(self, data, **kwargs):
        samples, targets = data
        idx = torch.randperm(samples.size(0))
        if self.device == 'cuda':
            idx.to(self.device)

        samples_perm, targets_perm = samples[idx], targets[idx]

        n_samples = samples.shape[0]
        llambda = np.random.beta(self.alpha, self.alpha, n_samples)
        llambda = torch.from_numpy(np.maximum(llambda, 1 - llambda)).float().to(self.device)

        # transpose batch dimension to become trailing dimension for broadcasting
        samples = samples.transpose(-1, 0)
        targets = targets.transpose(-1, 0)
        samples_perm = samples_perm.transpose(-1, 0)
        targets_perm = targets_perm.transpose(-1, 0)

        # apply mixing
        mixed_samples = llambda * samples + (1 - llambda) * samples_perm
        mixed_targets = llambda * targets + (1 - llambda) * targets_perm

        mixed_samples = mixed_samples.transpose(-1, 0)
        mixed_targets = mixed_targets.transpose(-1, 0)
        return mixed_samples, mixed_targets


class BinaryClassificationMetrics(Callback):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

        self.epoch = 0
        self.batch_idx = 0
        self.phase = None
        self.phase_len = None

        self.losses_collected = []
        self.outs_collected = []
        self.targets_collected = []
        self.val_score = 0

    def on_epoch_begin(self, learner, **kwargs):
        self.epoch = learner.epoch

    def on_phase_begin(self, phase, phase_len, **kwargs):
        self.phase = phase
        self.phase_len = phase_len
        self.batch_idx = 0
        self.losses_collected = []
        self.outs_collected = []
        self.targets_collected = []

    def on_batch_end(self, loss, outs, targets, **kwargs):
        self.outs_collected.append(outs.detach().cpu().numpy())
        self.targets_collected.append(targets.detach().cpu().numpy())

        if self.phase is 'train':
            per = (self.batch_idx + 1) / self.phase_len
            end = '' if (self.batch_idx + 1 is not self.phase_len) else '\n'
            print(f'\repoch: {self.epoch}, iter {self.batch_idx + 1:3} of {self.phase_len:3}'
                  f' - {(per * 100):5.1f}%, loss of {loss:.9}', end=end)
        self.batch_idx += 1

    def on_phase_end(self, learner, **kwargs):
        outs_collected = np.concatenate(self.outs_collected)
        targets_collected = np.concatenate(self.targets_collected)

        # soft target metrics
        targets_collected = targets_collected > .125
        auc = roc_auc_score(targets_collected, outs_collected)
        aps = average_precision_score(targets_collected, outs_collected)

        # hard target metrics
        outs_collected = outs_collected > .5
        precision = precision_score(targets_collected, outs_collected, zero_division=False)
        recall = recall_score(targets_collected, outs_collected, zero_division=False)
        f1 = f1_score(targets_collected, outs_collected, zero_division=False)
        acc = accuracy_score(targets_collected, outs_collected)
        confmat = confusion_matrix(targets_collected, outs_collected)
        tn, fp, fn, tp = confmat.ravel()
        print(f'{self.phase:5} - tp:{tp:5}, fp:{fp:5}, tn:{tn:5}, fn:{fn:5}, '
              f'precision:{precision:5.2}, recall:{recall:5.2}, f1:{f1:5.2}, '
              f'acc:{acc:5.2}, auc:{auc:5.2}, aps:{aps:5.2}')

        if self.phase is 'val':
            learner.val_score = f1


class SchedulerWrap(Callback):
    """
    Wraps a scheduler for use in the callback manager
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, learner, **kwargs):
        self.scheduler.step(learner.val_score)


class SaveCheckpoint(Callback):
    def __init__(self, path):
        self.best_score = 0
        self.path = path
        self.model = None
        self.optimizer = None

    def on_fit_begin(self, learner, **kwargs):
        self.model = learner.model
        self.optimizer = learner.optimizer

    def on_epoch_end(self, learner, **kwargs):
        if learner.val_score >= self.best_score:
            self.best_score = learner.val_score
            print(f'saving checkpoint to {self.path}')

            torch.save({
                'epoch': learner.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler': learner.cb['SchedulerWrap'],
                'val_score': learner.val_score},
                self.path)
