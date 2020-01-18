import torch
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from torchsummary import summary

from data_preprocessing import fetch_balanced_dataloaders
from torch_learner import Learner
from torch_models import CustomVGG

path = "mongodb+srv://gabrieldernbach:MUW9TFbgJO7Gm38W@cluster0-g69z0.gcp.mongodb.net"

ex = Experiment("atc: pytorch custom vgg mixup")
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(MongoObserver(url=path))

@ex.config
def cfg():
    model_name = 'CustomVGG'
    batch_size = 256
    learning_rate = 0.1
    epochs = 200


@ex.automain
def main(batch_size, learning_rate, epochs, _run):
    train_dl, validation_dl, test_dl = fetch_balanced_dataloaders(batch_size)
    # train_dl, validation_dl, test_dl = fetch_dummy_dataloader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomVGG.to(device)
    summary(model, input_size=(1, 40, 126))

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True)

    learner = Learner(model, criterion, optimizer, scheduler, db_observer=_run)
    learner.fit(train_dl, validation_dl, epochs=epochs)
