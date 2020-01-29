import torch
import torch.nn as nn
from sacred import Experiment
from torch.utils.data import TensorDataset, DataLoader

from fc_learner import Learner
from fc_models import ConditionNet
from utils import load_monolithic, subset_to_tensor, class_imbalance_sampler

ex = Experiment("MFCC condition net with Class Imbalance")


# path = "mongodb+srv://gabrieldernbach:MUW9TFbgJO7Gm38W@cluster0-g69z0.gcp.mongodb.net"
# ex.observers.append(MongoObserver(url=path))


@ex.config
def cfg():
    learning_rate = 0.1
    epochs = 200
    early_stop_patience = 35
    batch_size = 1000


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

    train_loader = DataLoader(TensorDataset(*train), batch_size=batch_size, sampler=sampler)
    validation_loader = DataLoader(TensorDataset(*validation), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(*test), batch_size=batch_size)

    model = ConditionNet()
    # model = nn.DataParallel(model)

    # trainer = Trainer(model=model,
    #                   criterion=nn.BCELoss(),
    #                   optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999)),
    #                   epochs=epochs,
    #                   early_stop_patience=early_stop_patience,
    #                   early_stop_verbose=True,
    #                   callback=logger
    #                   )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70, 100], gamma=0.1)
    #
    # lrfinder = LRFinder(model, optimizer, criterion)
    # lrfinder.range_test(train_loader, validation_loader, start_lr=10e-9, end_lr=1, num_iter=100)
    # lrfinder.plot()
    # lrfinder.reset()

    learner = Learner(model, criterion, optimizer, scheduler, db_observer=None)

    # trainer.fit(train_loader, validation_loader)
    learner.fit(train_loader, validation_loader, 200)
    learner.validate(test_loader)
    # roc, f1, confmat = evaluate_model(model, validation)
    # print(f'confusion matrix of validation set {confmat}')
    # roc, f1, confmat = evaluate_model(model, test)
    # print(f'confusion matrix of test set {confmat}')
    # return f'roc {roc:.3}, f1 {f1:.3}'
