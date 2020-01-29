import os
import pickle

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import WeightedRandomSampler


def split(data, a=0.6, b=0.8):
    """
    Create a random train, dev, test split of a pandas data frame
    """
    a, b = int(a * len(data)), int(b * len(data))
    data_shuffled = data.sample(frac=1, random_state=1).reset_index(drop=True)
    return np.split(data_shuffled, [a, b])


def normalize_data(train, dev, test):
    train_mean = train.mean(axis=0, keepdims=1)
    train_variance = (train - train_mean).var(axis=0, keepdims=1)

    train = (train - train_mean) / train_variance
    dev = (dev - train_mean) / train_variance
    test = (test - train_mean) / train_variance
    return train, dev, test


def treshhold_labels(Y_train, Y_dev, Y_test, threshold=.25):
    Y_train = Y_train > threshold
    Y_dev = Y_dev > threshold
    Y_test = Y_test > threshold
    return Y_train, Y_dev, Y_test


def load_monolithic(datapath):
    if os.path.exists(datapath):
        train, dev, test, normalization = pickle.load(open(datapath, 'rb'))
        # train, dev, test = pickle.load(open(datapath, 'rb'))
    else:
        cwd = os.getcwd()
        os.system(cwd + '/data_monolithic_mfcc.py')
        train, dev, test, normalization = pickle.load(open(datapath, 'rb'))
    return train, dev, test


def subset_to_tensor(subset):
    inputs = torch.from_numpy(subset[0]).float()
    # context = torch.from_numpy(x[1].values).long()
    context = torch.from_numpy(np.zeros(len(inputs))).long()
    labels = torch.from_numpy(subset[2]).float()
    return inputs, context, labels


def evaluate_model(model, subset):
    inputs, contexts, labels = subset
    # predictions = F.softmax(model(inputs, contexts), dim=1)[:, 1].detach().numpy()  # multiclass
    predictions = model(inputs, contexts).detach().numpy()
    labels = labels.detach().numpy() > 0.35
    roc = roc_auc_score(labels, predictions)
    f1 = f1_score(labels, predictions > 0.5)
    confmat = confusion_matrix(labels, predictions > 0.5)
    return roc, f1, confmat


def class_imbalance_sampler(labels, threshold=0.35):
    """
    Takes integer class labels and returns the torch sampler
    for balancing the class prior distribution
    """
    labels = (labels > 0.35).long()
    class_count = torch.bincount(labels)
    weighting = 1. / class_count.float()
    weights = weighting[labels]
    sampler = WeightedRandomSampler(weights, len(labels))
    return sampler
