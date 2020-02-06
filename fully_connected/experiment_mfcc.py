import numpy as np
import torch
import torch.nn as nn
from sacred import Experiment
from torch.utils.data import TensorDataset, DataLoader

from fc_learner import Learner
from fc_models import DropNNBig
from utils import class_imbalance_sampler

ex = Experiment("MFCC condition net with Class Imbalance")


# path = "mongodb+srv://gabrieldernbach:MUW9TFbgJO7Gm38W@cluster0-g69z0.gcp.mongodb.net"
# ex.observers.append(MongoObserver(url=path))


@ex.config
def cfg():
    learning_rate = 0.1
    epochs = 200
    early_stop_patience = 35
    batch_size = 1000


@ex.automain
def main(learning_rate, epochs, early_stop_patience, batch_size, _run):
    # path = 'data_monolithic_mfcc.pkl'
    # train, validation, test = [subset_to_tensor(d) for d in load_monolithic(path)]
    #
    # sampler = class_imbalance_sampler(train[-1], threshold=0.1)
    #
    # train_loader = DataLoader(TensorDataset(*train), batch_size=batch_size, sampler=sampler)
    # validation_loader = DataLoader(TensorDataset(*validation), batch_size=batch_size)
    # test_loader = DataLoader(TensorDataset(*test), batch_size=batch_size)

    train_samples = torch.tensor(np.load('data_train_samples.npy'))
    train_targets = torch.tensor(np.load('data_train_targets.npy')).float()
    validation_samples = torch.tensor(np.load('data_validation_samples.npy'))
    validation_targets = torch.tensor(np.load('data_validation_targets.npy')).float()
    test_samples = torch.tensor(np.load('data_test_samples.npy'))
    test_targets = torch.tensor(np.load('data_test_targets.npy')).float()

    sampler = class_imbalance_sampler(train_targets.flatten(), threshold=0.125)
    train_loader = DataLoader(TensorDataset(train_samples, train_targets), batch_size=batch_size, sampler=sampler)
    validation_loader = DataLoader(TensorDataset(validation_samples, validation_targets), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_samples, test_targets), batch_size=batch_size)

    model = DropNNBig()

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    learner = Learner(model, criterion, optimizer, scheduler, db_observer=None)

    learner.fit(train_loader, validation_loader, 200)
    learner.validate(test_loader)
