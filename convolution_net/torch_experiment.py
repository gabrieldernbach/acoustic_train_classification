import numpy as np
import torch
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from torch.optim.lr_scheduler import CyclicLR
from torch_lr_finder import LRFinder
from torchsummary import summary

from data_preprocessing import fetch_balanced_dataloaders
from torch_learner import Learner
from torch_models import CustomVGG

path = "mongodb+srv://gabrieldernbach:MUW9TFbgJO7Gm38W@cluster0-g69z0.gcp.mongodb.net"

ex = Experiment("atc: pytorch")
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(MongoObserver(url=path))


def find_min_max_lr(losses, lr):
    seq = np.insert(np.diff(losses), -1, 0)  # get slope and make up -1
    seq = seq < 0  # get all with negative slope
    n = len(seq)
    max_idx = 0
    max_len = 0
    curr_len = 0
    curr_idx = 0
    for k in range(n):
        if seq[k] > 0:
            curr_len += 1
            if curr_len == 1:
                curr_idx = k
        else:
            if curr_len > max_len:
                max_len = curr_len
                max_idx = curr_idx
            curr_len = 0

    if max_len > 0:
        print('Index : ', max_idx, ',Length : ', max_len, )
    else:
        print("No positive sequence detected.")

    min_lr = lr[max_idx]
    max_lr = lr[max_idx + max_len]

    return min_lr, max_lr


@ex.config
def cfg():
    batch_size = 256
    learning_rate = 0.2
    epochs = 800
    early_stop_patience = 200
    model_name = 'CustomVGG'


@ex.automain
def main(model_name, batch_size, learning_rate,
         epochs, early_stop_patience, _run):
    train_dl, validation_dl, test_dl = fetch_balanced_dataloaders()
    # train_dl = DataLoader(TensorDataset(torch.randn(200, 1, 40, 126), torch.rand(200, 1)), batch_size=5)
    # validation_dl = DataLoader(TensorDataset(torch.randn(200, 1, 40, 126), torch.rand(200, 1)), batch_size=5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomVGG.to(device)
    summary(model, input_size=(1, 40, 126))
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.2)

    lr_finder = LRFinder(model, optimizer, criterion)
    lr_finder.range_test(train_dl, start_lr=1e-7, end_lr=10, num_iter=200)
    min_lr, max_lr = find_min_max_lr(lr_finder.history['loss'], lr_finder.history['lr'])
    print(min_lr, max_lr)
    # lr_finder.plot()

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=10, verbose=True)
    scheduler = CyclicLR(optimizer, base_lr=min_lr * 10, max_lr=max_lr,
                         step_size_up=len(train_dl) // 2, mode='triangular2')

    learner = Learner(model, criterion, optimizer, scheduler)
    learner.fit(train_dl, validation_dl, epochs=200, db_observer=_run)
