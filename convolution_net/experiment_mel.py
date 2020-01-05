import os

import torch.nn as nn
import torch.optim as optim
from sacred import Experiment
from torch.utils.data import DataLoader

from conv_models import ResNet128
from conv_trainer import Trainer
from data_set_custom import MelDataset, class_imbalance_sampler, evaluate_model

ex = Experiment("SmallMel")


# path = "mongodb+srv://gabrieldernbach:MUW9TFbgJO7Gm38W@cluster0-g69z0.gcp.mongodb.net"
# ex.observers.append(MongoObserver(url=path))


@ex.config
def cfg():
    learning_rate = 0.1
    epochs = 20


@ex.capture
def logger(trainer):
    ex.log_scalar('training_loss', trainer.training_loss, step=trainer.current_epoch)
    ex.log_scalar('training_accuracy', trainer.training_accuracy, step=trainer.current_epoch)
    ex.log_scalar('validation_loss', trainer.validation_loss, step=trainer.current_epoch)
    ex.log_scalar('validation_accuracy', trainer.validation_accuracy, step=trainer.current_epoch)


@ex.automain
def main(learning_rate, epochs, _run):
    files = '/mel_train.npz', '/mel_validation.npz', '/mel_test.npz'
    path = os.path.dirname(os.path.realpath(__file__))
    train_path, validation_path, test_path = [path + s for s in files]

    train_set = MelDataset(train_path)

    sampler = class_imbalance_sampler(train_set.labels)
    train_loader = DataLoader(train_set,
                              sampler=sampler,
                              batch_size=200,
                              num_workers=4,
                              pin_memory=True)
    validation_loader = DataLoader(MelDataset(validation_path),
                                   batch_size=20,
                                   num_workers=4,
                                   pin_memory=True)
    test_loader = DataLoader(MelDataset(test_path),
                             batch_size=20,
                             num_workers=4,
                             pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet128.to(device)

    trainer = Trainer(model=model,
                      device=device,
                      criterion=nn.BCELoss(),
                      optimizer=optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999)),
                      epochs=epochs,
                      callback=logger,
                      early_stop_patience=20,
                      early_stop_verbose=True,
                      )

    trainer.fit(train_loader, validation_loader)

    roc, f1, confmat = evaluate_model(trainer, test_loader)

    return f'roc {roc:.3} - f1 {f1:.3}'
