"""
Basic Neural Net with 320 inputs,
hidden layers of 50 and 20 and scalar logistic output.

Inputs are assumed to be flattened mfccs (20 n_mfcc x 16 steps = 320 features)
Targets are either the binary labels of detection {0, 1},
or the amount of samples in the section labeled as detected [0, 1].
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score

from fully_connected.utils import load_monolithic


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


class DropNN(nn.Module):
    """
    Neural Network with drop out regularization
    """

    def __init__(self):
        super(DropNN, self).__init__()
        self.fc1 = nn.Linear(320, 100)
        self.bn1 = nn.BatchNorm1d(1)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 40)
        self.bn2 = nn.BatchNorm1d(1)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(40, 20)
        self.bn3 = nn.BatchNorm1d(1)
        self.drop3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = torch.sigmoid(self.fc4(x))
        return x


def train_model(model, criterion, optimizer, num_epochs, early_stopping, verbose=True):
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
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        training_loss.append(running_loss)
        validation_loss.append(criterion(model(X_dev).squeeze(), Y_dev))
        verbose and print(f'validation loss is {validation_loss[-1]}')
        if (early_stopping is True) and (epoch > 30):
            eps = validation_loss[epoch - 30] - validation_loss[epoch]
            if eps < early_stop_criterion:
                print('early stopping criterion met')
                break
        verbose and print(f'epoch {epoch}, loss = {loss.item()}')

    return training_loss, validation_loss


if __name__ == '__main__':
    # subsets = ['data_monolithic_mfcc_BHV.pkl',
    #            'data_monolithic_mfcc_BRL.pkl',
    #            'data_monolithic_mfcc_VLD.pkl',
    #            'data_monolithic_mfcc.pkl', ]
    subsets = ['data_monolithic_mfcc.pkl']

    cross_scores = np.zeros([len(subsets), len(subsets)])

    for idx, i in enumerate(subsets):
        for jdx, j in enumerate(subsets):
            print(f'\nlearn on {i} \npredict on {j}')
            train, dev, _ = load_monolithic(i)
            _, _, test = load_monolithic(j)

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

            net = DropNN()
            # net(X_train)

            train_model(model=net,
                        criterion=nn.BCELoss(),
                        optimizer=optim.Adam(net.parameters(),
                                             lr=0.001,
                                             betas=(0.9, 0.999)),
                        num_epochs=400,
                        early_stopping=True,
                        verbose=True)
            # print(f'\n\n\n Results for {station}')
            threshold = .25
            print('=== Training Set Performance ===')
            Y_train_binary = Y_train.numpy() > threshold
            train_predict = net(X_train).detach().numpy().flatten()
            print(confusion_matrix(Y_train_binary, train_predict > threshold, normalize='pred'))
            print(roc_auc_score(Y_train_binary, train_predict))
            print('=== Dev Set Performance ===')
            Y_dev_binary = Y_dev.numpy() > threshold
            dev_predict = net(X_dev).detach().numpy().flatten()
            print(confusion_matrix(Y_dev_binary, dev_predict > threshold, normalize='pred'))
            print(roc_auc_score(Y_dev_binary, dev_predict))
            print('=== Test Set Performance ===')
            Y_test_binary = Y_test.numpy() > threshold
            test_predict = net(X_test).detach().numpy().flatten()
            print(confusion_matrix(Y_test_binary, test_predict > .35, normalize='pred'))
            print(confusion_matrix(Y_test_binary, test_predict > .35, normalize='true'))
            print(confusion_matrix(Y_test_binary, test_predict > .35))
            print(roc_auc_score(Y_test_binary, test_predict))

            cross_scores[idx, jdx] = roc_auc_score(Y_test_binary, test_predict)

    cross_scores = np.round(cross_scores, decimals=2)
    print(cross_scores)

    # labels = ['BHV', 'BRL', 'VLD', 'ALL']
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(cross_scores)
    # plt.title('AUC Across Stations')
    # fig.colorbar(cax)
    # ax.set_xticklabels([''] + labels)
    # ax.set_yticklabels([''] + labels)
    # plt.xlabel('Tested on')
    # plt.ylabel('Trained on')
    # for i in range(len(labels)):
    #     for j in range(len(labels)):
    #         text = ax.text(j, i, cross_scores[i, j],
    #                        ha="center", va="center", color="w")
    # plt.show()
