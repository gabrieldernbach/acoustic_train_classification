import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from sacred import Experiment
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from convolution_net.conv_models import ResNet224
from convolution_net.conv_trainer import Learner
from convolution_net.data_augmentations import Resize, MelSpectrogram
from convolution_net.data_set_custom import RawDataset, split, balancing_sample_weights

ex = Experiment("OnMel")


# path = "mongodb+srv://gabrieldernbach:MUW9TFbgJO7Gm38W@cluster0-g69z0.gcp.mongodb.net"
# ex.observers.append(MongoObserver(url=path))


@ex.config
def cfg():
    epochs = 400
    batch_size = 16
    learning_rate = 0.1


@ex.capture
def logger(trainer, _run):
    ex.log_scalar('training_loss', trainer.training_loss, step=trainer.current_epoch)
    ex.log_scalar('training_accuracy', trainer.training_accuracy, step=trainer.current_epoch)
    ex.log_scalar('validation_loss', trainer.validation_loss, step=trainer.current_epoch)
    ex.log_scalar('validation_accuracy', trainer.validation_accuracy, step=trainer.current_epoch)


@ex.main
def main(batch_size, epochs, learning_rate):
    print('read data set')
    data_register = pickle.load(open('../data/data_register.pkl', 'rb'))
    # data_register = data_register.sample(n=150).reset_index(drop=True)
    train, validation, test = split(data_register)

    transform = transforms.Compose([
        MelSpectrogram(),  # todo needs normalization
        Resize(224, 224),
        # ExpandDim(),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(RawDataset(train), batch_size=batch_size)
    weights, n_samples = balancing_sample_weights(train_loader)
    train_sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(RawDataset(train, transform=transform),
                              sampler=train_sampler,
                              batch_size=batch_size,
                              num_workers=8,
                              pin_memory=True)

    validation_loader = DataLoader(RawDataset(validation, transform=transform),
                                   batch_size=batch_size,
                                   num_workers=8,
                                   pin_memory=True)
    test_loader = DataLoader(RawDataset(test, transform=transform),
                             batch_size=batch_size,
                             num_workers=8,
                             pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ResNet224.to(device)

    print('start training')
    trainer = Trainer(model=net,
                      device=device,
                      criterion=nn.CrossEntropyLoss(),
                      optimizer=optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999)),
                      epochs=epochs,
                      callback=logger,
                      early_stop_patience=20,
                      early_stop_verbose=True,
                      )

    trainer.fit(train_loader, validation_loader)

    # evaluator(model=net,
    #           train=train_loader,
    #           validation=validation_loader,
    #           test=test_loader)


if __name__ == "__main__":
    ex.run(config_updates={'batch_size': 32})
    ex.run(config_updates={'batch_size': 64})
    ex.run(config_updates={'batch_size': 128})
    ex.run(config_updates={'batch_size': 256})

# todo labwatch hyper param search
