"""
Save dataset under mel spectrogram transformation for convolutional network
"""
import multiprocessing as mp
import os
import xml.etree.ElementTree as ElementTree
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from librosa.util import frame
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def fetch_balanced_dataloaders(batch_size=64):
    files = '/mel_train.npz', '/mel_validation.npz', '/mel_test.npz'
    path = os.path.dirname(os.path.realpath(__file__))
    train_path, validation_path, test_path = [path + s for s in files]

    train = np.load(train_path, allow_pickle=True)
    validation = np.load(validation_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    x_train, y_train = train['audio'], train['label']
    x_validation, y_validation = validation['audio'], validation['label']
    x_test, y_test = test['audio'], test['label']

    # reshape for convolution
    x_train = np.expand_dims(x_train, axis=1)
    x_validation = np.expand_dims(x_validation, axis=1)
    x_test = np.expand_dims(x_test, axis=1)
    y_train = (np.expand_dims(y_train, axis=-1) > 0.125).astype(float)
    y_validation = (np.expand_dims(y_validation, axis=-1) > 0.125).astype(float)
    y_test = (np.expand_dims(y_test, axis=-1) > 0.125).astype(float)

    x_train, y_train = upsample_minority(x_train, y_train, y_threshold=0.125)
    print(sum(y_train > 0.125), len(y_train))
    print(sum(y_validation > 0.125), len(y_validation))
    print(x_train.shape)

    dl_args = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True}
    train_dl = DataLoader(TensorDataset(tensor(x_train), tensor(y_train).float()), **dl_args)
    validation_dl = DataLoader(TensorDataset(tensor(x_validation), tensor(y_validation).float()), **dl_args)
    test_dl = DataLoader(TensorDataset(tensor(x_test), tensor(y_test).float()), **dl_args)
    return train_dl, validation_dl, test_dl


def fetch_dummy_dataloader():
    train_dl = DataLoader(TensorDataset(torch.randn(200, 1, 40, 126), torch.rand(200, 1)), batch_size=5)
    validation_dl = DataLoader(TensorDataset(torch.randn(200, 1, 40, 126), torch.rand(200, 1)), batch_size=5)
    test_dl = DataLoader(TensorDataset(torch.randn(200, 1, 40, 126), torch.rand(200, 1)), batch_size=5)
    return train_dl, validation_dl, test_dl


class Normalizer:
    """
    computes and stores global mean and variance for a set of ndarrays.
    The dimensions are assumed as instance x height x length
    """

    def __init__(self):
        self.xm = np.array([])
        self.xv = np.array([])

    def fit(self, data):
        self.xm = data.mean(axis=0)
        self.xv = data.var(axis=0)
        return self.xm, self.xv

    def transform(self, data):
        data = data - self.xm[None, :, :]
        data = data / self.xv[None, :, :]
        return data

    def fit_transform(self, data):
        self.fit(data)
        data = self.transform(data)
        return data


def split(data, a=0.6, b=0.8):
    """
    Create a random train, validation, test split of a pandas data frame
    """
    a, b = int(a * len(data)), int(b * len(data))
    data_shuffled = data.sample(frac=1, random_state=1).reset_index(drop=True)
    train, validation, test = np.split(data_shuffled, [a, b])
    validation.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    return train, validation, test


def upsample_minority(x_train, y_train, y_threshold=0.25):
    """
    For binary classification returns a balanced dataset
    by repeatedly sampling from the minority class.
    For soft labels provide a split point y_threshold

    :param x_train: ndarray
    :param y_train: ndarray
    :param y_threshold: float
    :return: reampled_features, resampled_labels
    """
    pos_features = x_train[np.where(y_train > y_threshold)[0]]
    neg_features = x_train[np.where(y_train < y_threshold)[0]]
    pos_labels = y_train[np.where(y_train > y_threshold)[0]]
    neg_labels = y_train[np.where(y_train < y_threshold)[0]]

    ids = np.arange(len(pos_features))
    choices = np.random.choice(ids, len(neg_features))
    res_pos_features = pos_features[choices]
    res_pos_labels = pos_labels[choices]

    resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
    resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)
    order = np.arange(len(resampled_labels))
    np.random.shuffle(order)
    resampled_features = resampled_features[order]
    resampled_labels = resampled_labels[order]
    return resampled_features, resampled_labels



def mark_to_vec(marks_in_s, len_sequence):
    """
    convert the marks ins seconds into a time series label vector
    and track durations of the marked sections
    """
    mark_in_samp = []
    for mark in marks_in_s:
        start = round(float(mark[0]) * SAMPLE_RATE)
        end = round(float(mark[1]) * SAMPLE_RATE)
        mark_in_samp.append([start, end])

    label_vec = np.zeros(len_sequence)
    for mark in mark_in_samp:
        label_vec[mark[0]:mark[1]] = 1
    detection = np.alltrue(label_vec == 0)

    return label_vec, detection


def extract_aup(aup_path):
    """
    Extract audio and annotations from a single audacity project
    """
    # parse xml
    doc = ElementTree.parse(aup_path)
    root = doc.getroot()

    # load wavfile
    xml_wave = r'{http://audacity.sourceforge.net/xml/}wavetrack'
    name = root.find(xml_wave).attrib['name'] + '.wav'
    print(f'extracting data point {name}')
    audio = librosa.core.load(data_path + '/' + name, SAMPLE_RATE, mono=False)[0][0, :]
    audio_len = len(audio)

    # extract labels
    xml_label = r'{http://audacity.sourceforge.net/xml/}label'
    marks_in_s: List[Tuple[str, str]] = []
    for element in root.iter(xml_label):
        start = element.attrib['t']
        end = element.attrib['t1']
        marks_in_s.append((start, end))
    label_vec, detection = mark_to_vec(marks_in_s, audio_len)
    station_vec = np.repeat(station_id, audio_len)

    return station_vec, audio, label_vec, detection


def stmt(x):
    """
    returns the normalized log short term mel spectrogram transformation of signal vector x
    """
    x = librosa.feature.melspectrogram(x, sr=SAMPLE_RATE, n_fft=512, hop_length=STFT_HOP_LENGTH, n_mels=N_MELS)
    x = np.log(x + 1e-12)
    return x


def transform(dataset):
    X = dataset.audio
    S = dataset.station
    Y = dataset.label_vec

    print('starting transform')
    X = [frame(x, frame_length=FRAME_LENGTH, hop_length=FRAME_HOP_LENGTH) for x in X]
    Y = [frame(y, frame_length=FRAME_LENGTH, hop_length=FRAME_HOP_LENGTH) for y in Y]
    S = [frame(s, frame_length=FRAME_LENGTH, hop_length=FRAME_HOP_LENGTH) for s in S]

    X = np.concatenate(X, axis=-1).T
    X = [stmt(x) for x in tqdm(X)]
    X = np.stack(X)
    S = np.concatenate(S, axis=-1).T[:, 0]
    Y = np.concatenate(Y, axis=-1)
    Y = Y.mean(axis=0)

    return X, S, Y


if __name__ == '__main__':
    SAMPLE_RATE = 8_000  # 48_000
    FRAME_LENGTH = 16_000  # 96_000
    FRAME_HOP_LENGTH = 2_000  # 12_000
    STFT_HOP_LENGTH = 128  # 1024
    N_MELS = 40

    root = os.path.abspath('../data/')
    _, stations, _ = next(os.walk(root))
    print(f'found sub folders for station {stations}')

    data = []
    for station_id, station in enumerate(stations):
        print(f'loading station {station}')
        data_path = root + '/' + station
        files = os.listdir(data_path)
        audacity_projects = [f for f in files if f.endswith('.aup')]
        project_paths = [[os.path.join(data_path, aup)] for aup in audacity_projects]
        print(f'found {len(project_paths)} files')

        with mp.Pool(mp.cpu_count()) as p:
            station_data = p.starmap(extract_aup, project_paths)

        data.extend(station_data)

    print(len(data))
    data = pd.DataFrame(data, columns=['station', 'audio', 'label_vec', 'detection'])
    print('split data')
    train, validation, test = split(data)
    del data

    print('start feature extraction')
    normalizer = Normalizer()
    X_train, S_train, Y_train = transform(train)
    del train
    X_train = normalizer.fit_transform(X_train)
    X_validation, S_validation, Y_validation = transform(validation)
    del validation
    X_validation = normalizer.transform(X_validation)
    X_test, S_test, Y_test = transform(test)
    del test
    X_test = normalizer.transform(X_test)

    np.savez('mel_train.npz',
             audio=X_train,
             station=S_train,
             label=Y_train)

    np.savez('mel_validation.npz',
             audio=X_validation,
             station=S_validation,
             label=Y_validation)

    np.savez('mel_test.npz',
             audio=X_test,
             station=S_test,
             label=Y_test)
