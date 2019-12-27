"""
Convolutional Neural Net
"""

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from torchvision import transforms

from baseline_fully_connected.utils import split
from convolution_net_classification.augmentations import Spectrogram, Resize, ExpandDim
from convolution_net_classification.data_loader import AcousticSceneDataset


def train_model(model, criterion, optimizer, num_epochs, early_stopping):
    training_loss = []
    validation_loss = []
    early_stop_criterion = 1e-13
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f'processing training on epoch {epoch} batch {batch_idx}')
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'total training loss {running_loss}')
        training_loss.append(running_loss)

        model.eval()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(dev_loader):
            print(f'processing validation on {batch_idx}')
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
        print(f'total validation loss {running_loss}')
        validation_loss.append(running_loss)

        if (early_stopping is True) and (epoch > 10):
            eps = validation_loss[epoch - 10] - validation_loss[epoch]
            if eps < early_stop_criterion:
                print('early stopping criterion met')
                break

    return training_loss, validation_loss


if __name__ == '__main__':
    print('read data set')
    data_register = pickle.load(open('../data/data_register.pkl', 'rb'))
    train, dev, test = split(data_register)

    transform = transforms.Compose([
        Spectrogram(),  # todo needs normalization
        Resize(224, 224),
        ExpandDim()
    ])

    train_loader = DataLoader(AcousticSceneDataset(train, transform=transform), batch_size=50)
    dev_loader = DataLoader(AcousticSceneDataset(dev, transform=transform), batch_size=50)
    test_loader = DataLoader(AcousticSceneDataset(test, transform=transform), batch_size=50)

    net = EfficientNet.from_name('efficientnet-b0')

    first_conv_layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
    net = nn.Sequential(first_conv_layer, net, nn.Linear(1000, 1), nn.Softmax(dim=-1))

    train_model(model=net,
                criterion=nn.BCELoss(),
                optimizer=optim.Adam(net.parameters(),
                                     lr=0.01,
                                     betas=(0.9, 0.999)),
                num_epochs=400,
                early_stopping=True)

    torch.save(net.state_dict(), 'efficient_net1')

    # print(f'\n\n\n Results for {station}')
    # print('=== Training Set Performance ===')
    # Y_train_binary = Y_train.numpy() > .25
    # print(confusion_matrix(Y_train_binary, train_predict > .25))
    # print(roc_auc_score(Y_train_binary, train_predict))
    # print('=== Dev Set Performance ===')
    # Y_dev_binary = Y_dev.numpy() > .25
    # dev_predict = net(X_dev).detach().numpy().flatten()
    # print(confusion_matrix(Y_dev_binary, dev_predict > .25))
    # print(roc_auc_score(Y_dev_binary, dev_predict))
    # print('=== Test Set Performance ===')
    # Y_test_binary = Y_test.numpy() > .25
    # test_predict = net(X_test).detach().numpy().flatten()
    # print(confusion_matrix(Y_test_binary, test_predict > .25))
    # print(roc_auc_score(Y_test_binary, test_predict))
