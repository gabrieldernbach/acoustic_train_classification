import torch
import torch.nn as nn
from sacred import Experiment
from sacred.observers import MongoObserver
from torch.utils.data import TensorDataset, DataLoader

from models import ElementConditionNet
from trainer import Trainer
from utils import load_monolithic, subset_to_tensor, evaluate_model, class_imbalance_sampler

ex = Experiment("MFCC condition net with Class Imbalance")
path = "mongodb+srv://gabrieldernbach:MUW9TFbgJO7Gm38W@cluster0-g69z0.gcp.mongodb.net"
ex.observers.append(MongoObserver(url=path))


@ex.config
def cfg():
    learning_rate = 0.001
    epochs = 200
    early_stop_patience = 35
    batch_size = 5000


def logger(trainer):
    ex.log_scalar('training_loss', trainer.training_loss, step=trainer.current_epoch)
    ex.log_scalar('training_accuracy', trainer.training_accuracy, step=trainer.current_epoch)
    ex.log_scalar('validation_loss', trainer.validation_loss, step=trainer.current_epoch)
    ex.log_scalar('validation_accuracy', trainer.validation_accuracy, step=trainer.current_epoch)


@ex.automain
def main(learning_rate, epochs, early_stop_patience, batch_size, _run):
    path = 'data_monolithic_mfcc.pkl'
    train, validation, test = [subset_to_tensor(d) for d in load_monolithic(path)]

    sampler = class_imbalance_sampler(train[-1], threshold=0.1)

    train_loader = DataLoader(TensorDataset(*train), batch_size=batch_size, sampler=sampler, num_workers=4)
    validation_loader = DataLoader(TensorDataset(*validation), batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(TensorDataset(*test), batch_size=batch_size)

    model = ElementConditionNet()

    trainer = Trainer(model=model,
                      criterion=nn.BCELoss(),
                      optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999)),
                      epochs=epochs,
                      early_stop_patience=early_stop_patience,
                      early_stop_verbose=True,
                      callback=logger
                      )

    trainer.fit(train_loader, validation_loader)
    roc, f1, confmat = evaluate_model(model, validation)
    print(f'confusion matrix of validation set {confmat}')
    roc, f1, confmat = evaluate_model(model, test)
    print(f'confusion matrix of test set {confmat}')
    return f'roc {roc:.3}, f1 {f1:.3}'
