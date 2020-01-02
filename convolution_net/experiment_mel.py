import torch
import torch.nn as nn
import torch.optim as optim
from sacred import Experiment
from sacred.observers import MongoObserver
from torch.utils.data import DataLoader

from convolution_net.data_set_custom import MelDataset
from convolution_net.model import ResNet128
from convolution_net.trainer import Trainer

ex = Experiment("OnMel")
path = "mongodb+srv://gabrieldernbach:MUW9TFbgJO7Gm38W@cluster0-g69z0.gcp.mongodb.net"
ex.observers.append(MongoObserver(url=path))


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
    train_path, validation_path, test_path = 'mel_train.npz', 'mel_dev.npz', 'mel_test.npz'
    train_loader = DataLoader(MelDataset(train_path), batch_size=20)
    validation_loader = DataLoader(MelDataset(validation_path), batch_size=20)
    test_loader = DataLoader(MelDataset(test_path), batch_size=20)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet128.to(device)

    trainer = Trainer(model=model,
                      device=device,
                      criterion=nn.CrossEntropyLoss(),
                      optimizer=optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999)),
                      epochs=epochs,
                      callback=logger,
                      early_stop_patience=20,
                      early_stop_verbose=True,
                      )

    trainer.fit(test_loader, validation_loader)

    _run.result = trainer.validation_accuracy
