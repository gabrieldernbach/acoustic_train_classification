"""
Basic Neural Net with
"""
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score


class ONN(nn.Module):
    """
    Ordinary Neural Network
    """

    def __init__(self):
        super(ONN, self).__init__()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class dONN(nn.Module):
    """
    Neural Network with drop out regularization
    """

    def __init__(self):
        super(sONN, self).__init__()
        self.fc1 = nn.Linear(320, 100)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 40)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(40, 20)
        self.drop3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.drop3(F.relu(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))
        return x


def train_model(model, criterion, optimizer, num_epochs, early_stopping):
    training_loss = []
    validation_loss = []
    early_stop_criterion = 1e-13
    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_size = 5000
        model.train()

        for beg_i in range(0, X_train.size(0), batch_size):
            x_batch = X_train[beg_i:beg_i + batch_size, :]
            y_batch = Y_train[beg_i:beg_i + batch_size]

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        training_loss.append(running_loss)
        validation_loss.append(criterion(model(X_dev), Y_dev))
        print(f'validation loss is {validation_loss[-1]}')
        if (early_stopping is True) and (epoch > 30):
            eps = validation_loss[epoch - 30] - validation_loss[epoch]
            if eps < early_stop_criterion:
                print('early stopping criterion met')
                break
        print(f'epoch {epoch}, loss = {loss.item()}')

    return training_loss, validation_loss


if __name__ == '__main__':
    print('read data set')
    datapath = 'data_monolithic_mfcc.pkl'
    if os.path.exists(datapath):
        train, dev, test = pickle.load(open('data_monolithic_mfcc.pkl', 'rb'))
    else:
        os.system('data_monolithic_mfcc_py')
        train, dev, test = pickle.load(open('data_monolithic_mfcc.pkl', 'rb'))

    X_train, S_train, Y_train = train
    X_dev, S_dev, Y_dev = dev
    X_test, S_test, Y_test = test

    Xm = X_train.mean(axis=0)
    X_train -= Xm
    X_dev -= Xm
    X_test -= Xm
    Xs = X_train.std(axis=0).max()
    X_train /= Xs
    X_dev /= Xs
    X_test /= Xs
    X_train = torch.from_numpy(X_train).unsqueeze(1)
    X_dev = torch.from_numpy(X_dev).unsqueeze(1)
    X_test = torch.from_numpy(X_test).unsqueeze(1)
    Y_train = torch.from_numpy(Y_train).float()
    Y_dev = torch.from_numpy(Y_dev).float()
    Y_test = torch.from_numpy(Y_test).float()
    X_train = X_train.reshape(len(X_train), 1, -1)
    X_dev = X_dev.reshape(len(X_dev), 1, -1)
    X_test = X_test.reshape(len(X_test), 1, -1)

    net = sONN()
    net(X_train)
    train_model(model=net,
                criterion=nn.BCELoss(),
                optimizer=optim.Adam(net.parameters(),
                                     lr=0.001,
                                     betas=(0.9, 0.999)),
                num_epochs=400,
                early_stopping=True)
    # print(f'\n\n\n Results for {station}')
    print('=== Training Set Performance ===')
    Y_train_binary = Y_train.numpy() > .25
    train_predict = net(X_train).detach().numpy().flatten()
    print(confusion_matrix(Y_train_binary, train_predict > .25))
    print(roc_auc_score(Y_train_binary, train_predict))
    print('=== Dev Set Performance ===')
    Y_dev_binary = Y_dev.numpy() > .25
    dev_predict = net(X_dev).detach().numpy().flatten()
    print(confusion_matrix(Y_dev_binary, dev_predict > .25))
    print(roc_auc_score(Y_dev_binary, dev_predict))
    print('=== Test Set Performance ===')
    Y_test_binary = Y_test.numpy() > .25
    test_predict = net(X_test).detach().numpy().flatten()
    print(confusion_matrix(Y_test_binary, test_predict > .25))
    print(roc_auc_score(Y_test_binary, test_predict))

