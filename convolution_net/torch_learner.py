import torch

from torch_callbacks import CallbackHandler


class Learner:

    def __init__(self, model, criterion, optimizer, callbacks=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.cb = CallbackHandler(callbacks)

        self.start_epoch = 0
        self.val_score = float()
        self.epoch = int()

    def fit(self, train_loader, validation_loader, max_epoch):
        self.cb.on_fit_begin(learner=self)
        for self.epoch in range(self.start_epoch, max_epoch):
            self.cb.on_epoch_begin(learner=self)

            self.train(train_loader)
            self.validate(validation_loader)

            self.cb.on_epoch_end(learner=self)

    def train(self, data_loader):
        self.model.train()
        self.cb.on_phase_begin(phase='train', phase_len=len(data_loader))

        for i, (samples, targets) in enumerate(data_loader):
            samples, targets = samples.to(self.device), targets.to(self.device)
            samples, targets = self.cb.on_train_load(data=(samples, targets))

            outs = self.model(samples)
            loss = self.criterion(outs, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.cb.on_batch_end(loss=loss, outs=outs, targets=targets)
        self.cb.on_phase_end(learner=self)

    def validate(self, data_loader):
        self.model.eval()
        self.cb.on_phase_begin(phase='val', phase_len=len(data_loader))

        with torch.no_grad():
            for i, (samples, targets) in enumerate(data_loader):
                samples, targets = samples.to(self.device), targets.to(self.device)
                outs = self.model(samples)
                loss = self.criterion(outs, targets)

                self.cb.on_batch_end(loss=loss, outs=outs, targets=targets)
        self.cb.on_phase_end(learner=self)

    def resume(self, path='ckpt.pt'):
        print(f'loading checkpoint from {path}')
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.cb['SchedulerWrap'] = checkpoint['lr_scheduler']
        self.start_epoch = checkpoint['epoch']
        print(f'resuming from epoch {self.start_epoch}')
