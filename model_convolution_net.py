"""
Basic Convolutional Neural Net
"""

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from torchvision import transforms

from augmentations import Spectrogram, Resize, ExpandDim
from data_loader import AcousticSceneDataset
from utils import split


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(1, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.do1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.do2 = nn.Dropout(0.3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 2 * 3, 120)
        self.do3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(120, 84)
        self.do4 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(self.do1(F.relu(self.conv1(x))), (2, 2))
        x = F.max_pool2d(self.do2(F.relu(self.conv2(x))), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.do3(F.relu(self.fc1(x)))
        x = self.do4(F.relu(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x).squeeze())
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(args, model, device, train_loader, dev_loader, optimizer, epoch):
    early_stop_criterion = 1e-13
    running_loss = 0.0

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def train_model(model, criterion, optimizer, num_epochs, early_stopping):
    training_loss = []
    validation_loss = []
    early_stop_criterion = 1e-13
    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_size = 5000
        model.train()

        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f'processing training on {batch_idx}')
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        training_loss.append(running_loss)

        model.eval()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(dev_loader):
            print(f'processing validation on {batch_idx}')
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()

        validation_loss.append(running_loss)
        print(f'validation loss is {validation_loss[-1]}')
        if (early_stopping is True) and (epoch > 30):
            eps = validation_loss[epoch - 30] - validation_loss[epoch]
            if eps < early_stop_criterion:
                print('early stopping criterion met')
                break
        print(f'epoche {epoch}, loss = {loss.item()}')

    return training_loss, validation_loss


if __name__ == '__main__':
    print('read data set')
    data_register = pickle.load(open('data_register.pkl', 'rb'))
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
                                     lr=0.001,
                                     betas=(0.9, 0.999)),
                num_epochs=400,
                early_stopping=True)

    torch.save(net.state_dict(), 'efficient_net1')

    # # print(f'\n\n\n Results for {station}')
    # print('=== Training Set Performance ===')
    # Y_train_binary = Y_train.numpy() > .25
    # train_predict = net(X_train).detach().numpy().flatten()
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
    #
