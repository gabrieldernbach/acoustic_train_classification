import os

import torch.nn as nn
import torch.optim as optim
from sacred import Experiment
from torch.utils.data import DataLoader

from old_pytorch.conv_models import VggNet
from old_pytorch.conv_trainer import Learner
from old_pytorch.data_set_custom import MelDataset, class_imbalance_sampler

ex = Experiment("atc: VGG pytorch")


# path = "mongodb+srv://gabrieldernbach:MUW9TFbgJO7Gm38W@cluster0-g69z0.gcp.mongodb.net"
# ex.observers.append(MongoObserver(url=path))


@ex.config
def cfg():
    learning_rate = 0.2
    epochs = 800
    batch_size = 512
    early_stop_patience = 40


@ex.automain
def main(learning_rate, batch_size, epochs, early_stop_patience, _run):
    files = '/mel_train.npz', '/mel_validation.npz', '/mel_test.npz'
    path = os.path.dirname(os.path.realpath(__file__))
    train_path, validation_path, test_path = [path + s for s in files]

    train_set = MelDataset(train_path)

    sampler = class_imbalance_sampler(train_set.labels)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    validation_loader = DataLoader(MelDataset(validation_path), batch_size=batch_size * 2,
                                   shuffle=True, pin_memory=True)
    test_loader = DataLoader(MelDataset(test_path), batch_size=batch_size * 2,
                             num_workers=4, pin_memory=True)

    model = VggNet

    trainer = Learner(model=model,
                      criterion=nn.BCEWithLogitsLoss(),
                      optimizer=optim.SGD(model.parameters(), lr=learning_rate),
                      epochs=epochs,
                      early_stop_patience=early_stop_patience,
                      _run=_run, )

    trainer.fit(train_loader, validation_loader)
    trainer.evaluate(validation_loader)
    auc, f1, _ = trainer.evaluate(test_loader)

    return f'auc {auc:.3} - f1 {f1:.3}'

# https://pytorch.org/docs/1.1.0/_modules/torch/nn/parallel/distributed_cpu.html
