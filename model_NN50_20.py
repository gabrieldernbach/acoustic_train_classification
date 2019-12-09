"""
Basic Neural Net with
"""

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.do1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.do2 = nn.Dropout(0.3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 2 * 3, 120)  # is there an easy way to know the dim?
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


class ONN(nn.Module):

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


class sONN(nn.Module):

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
            x_batch = X_train[beg_i:beg_i + batch_size, :]  # optionally .to(device)
            y_batch = Y_train[beg_i:beg_i + batch_size]  # optionally .to(device)

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
        print(f'epoche {epoch}, loss = {loss.item()}')

    return training_loss, validation_loss


if __name__ == '__main__':
    print('read data set')
    #    for station in ['VLD', 'BHV', 'BRL']:
    #         data = pd.read_pickle('data.pkl')
    #         data = data[data.station == station] # BHV, VLD, BRL
    #         print('split data')
    #         train, dev, test = mfcc_constructor.split(data)
    #
    #         print('start feature extraction')
    #         train = mfcc_constructor.transform(train)
    #         dev = mfcc_constructor.transform(dev)
    #         test = mfcc_constructor.transform(test)
    data = pickle.load(open('data_split_mfcc_station.pkl', 'rb'))
    train, dev, test = data

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

    # # todo create classes for transforms
    #
    # pitchshift
    # amplitude
    # shift
    # mixup
    # to
    # tensor
