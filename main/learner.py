import torch

from main.callback import CallbackHandler


class Learner:

    def __init__(self, model, optimizer, callbacks=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = self.model.criterion
        self.cb = CallbackHandler(callbacks)

        self.start_epoch = 0
        self.val_score = float()
        self.epoch = int()
        self.stop = False

    def fit(self, train_loader, validation_loader, max_epoch):
        self.cb.on_fit_begin(learner=self)
        for self.epoch in range(self.start_epoch, max_epoch):
            self.cb.on_epoch_begin(learner=self)

            self.train(train_loader)
            self.validate(validation_loader)

            self.cb.on_epoch_end(learner=self)
            if self.stop:
                break

    def train(self, data_loader):
        self.model.train()
        self.cb.on_phase_begin(phase='train', phase_len=len(data_loader))

        for i, batch in enumerate(data_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(self.device)
            batch['audio'], batch['target'] = self.cb.on_train_load(data=(batch['audio'], batch['target']))

            out = self.model(batch)
            loss = self.criterion(out, batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.cb.on_batch_end(loss=loss, out=out, batch=batch)
        self.cb.on_phase_end(learner=self)

    def validate(self, data_loader):
        self.model.eval()
        self.cb.on_phase_begin(phase='val', phase_len=len(data_loader))

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                for key in batch.keys():
                    batch[key] = batch[key].to(self.device)
                out = self.model(batch)
                loss = self.criterion(out, batch)

                self.cb.on_batch_end(loss=loss, out=out, batch=batch)
        res = self.cb.on_phase_end(learner=self)
        return res

    def resume(self, path='ckpt.pt'):
        print(f'loading checkpoint from {path}')
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.cb['SchedulerWrap'] = checkpoint['lr_scheduler']
        self.start_epoch = checkpoint['epoch']
        print(f'resuming from epoch {self.start_epoch}')
