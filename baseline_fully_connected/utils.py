import os
import pickle

import numpy as np


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
        train, dev, test = pickle.load(open(datapath, 'rb'))
    else:
        cwd = os.getcwd()
        os.system(cwd + '/data_monolithic_mfcc.py')
        train, dev, test = pickle.load(open(datapath, 'rb'))
    return train, dev, test