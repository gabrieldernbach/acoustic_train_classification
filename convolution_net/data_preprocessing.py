"""
Save dataset under mel spectrogram transformation for convolutional network
"""
import multiprocessing as mp
import os
import pickle

import librosa
import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset


def fetch_balanced_dataloaders(batch_size=256, label_threshold=0.125):
    files = '/data_train.npz', '/data_validation.npz', '/data_test.npz'
    path = os.path.dirname(os.path.realpath(__file__))
    train_path, validation_path, test_path = [path + s for s in files]

    train = np.load(train_path, allow_pickle=True)
    validation = np.load(validation_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    x_train, y_train = train['audio'], train['target']
    x_validation, y_validation = validation['audio'], validation['target']
    x_test, y_test = test['audio'], test['target']

    # reshape for convolution
    x_train = np.expand_dims(x_train, axis=1)
    x_validation = np.expand_dims(x_validation, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    x_train, y_train = upsample_minority(x_train, y_train, y_threshold=label_threshold)
    print('defective trains in train set', sum(y_train > label_threshold), len(y_train))
    print('defecitve trains in validation set', sum(y_validation > label_threshold), len(y_validation))
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


def upsample_minority(x_train, y_train, y_threshold=0.25):
    """
    For binary classification returns a balanced dataset
    by repeatedly sampling from the minority class.
    For soft labels a split point y_threshold must be provided

    Parameters
    ----------
    x_train : np.ndarray
    y_train : np.ndarray
    y_threshold : float

    Returns
    -------
    resampled_feaures : np.ndarray
    resampled_labels : np.ndarray
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


class DataExtractor:

    def __init__(self, data_register, *, target_speed, fs, target_fs,
                 frame_length, hop_length, stft_hoplength, n_mels, speed_normalization=False):
        self.data_register = data_register
        self.target_speed = target_speed
        self.fs = fs
        self.target_fs = target_fs

        self.frame_length = frame_length
        self.hop_length = hop_length

        self.stft_hoplength = stft_hoplength
        self.n_mels = n_mels

        self.speed_normalization = speed_normalization

    def __call__(self, idx):
        """
        loads sample and target [idx] from data self.data_register
        Parameters
        ----------
        idx : int
            index of entry in self.data_register to be loaded,
            must be in range(len(data_register))

        Returns
        -------
        samples : np.ndarray ( k x m x t )
            k log short term mel spectrograms of
            m mel bands over t time steps
        targets : np.ndarray ( k x 1 )
            k hard targets {0, 1}
        """
        print(f'started processing {idx} of {self.__len__()}')
        row = self.data_register.iloc[idx]

        # determine resampling fs
        subsample = self.fs / self.target_fs
        if self.speed_normalization is True:
            speed = row.speed_kmh
            resampling_ratio = np.maximum(speed / self.target_speed, 0.25)
        else:
            resampling_ratio = 1.
        resampling_fs = int(self.fs * resampling_ratio / subsample)

        # load audio and targets in memory
        audio, _ = librosa.core.load(row.audio_path, self.fs, mono=False)
        audio = np.asfortranarray(audio[0])
        targets = self.mark_to_vec(row.label, self.fs, len(audio))

        # extract log mel spectrograms
        audio = librosa.core.resample(audio, self.fs, resampling_fs)
        audio = librosa.util.frame(audio, self.frame_length, self.hop_length)
        samples = np.stack([self.stmt(x) for x in audio.T])

        # pool targets
        targets = librosa.core.resample(targets, self.fs, resampling_fs)
        targets = librosa.util.frame(targets, self.frame_length, self.hop_length)
        targets = ((targets.sum(0) / self.frame_length) > 0.125)[:, None]

        return samples, targets

    def stmt(self, x):
        """
        short term mel spectrogram of series x with log compression
        """
        x = librosa.feature.melspectrogram(x, sr=self.target_fs,
                                           n_fft=512,
                                           hop_length=self.stft_hoplength,
                                           n_mels=self.n_mels)
        x = np.log(x + 1e-12)
        return x

    @staticmethod
    def mark_to_vec(marks_in_s, fs, len_sequence):
        """
        convert list of marks in seconds into a time series label vector
        marked regions are assigned to 1, unmarked regions are assigned to 0.

        Parameters
        ----------
        marks_in_s : list
            List of tuples denoting begin and end of marked sections.
            The values are assumed to be in seconds
        fs : int
            Sampling frequency of the annotated source material
        len_sequence : int
            Number of samples of the annotated source material

        Returns
        -------
        label_vec : np.ndarray
            Vector of length len_sequence that contains 1s and 0s according
            to the marked regions in mark_in_s
        detection : bool
            Indicates if at least one anomalous region was marked
        """
        mark_in_samp = []
        for mark in marks_in_s:
            start = round(float(mark[0]) * fs)
            end = round(float(mark[1]) * fs)
            mark_in_samp.append([start, end])

        label_vec = np.zeros(len_sequence)
        for mark in mark_in_samp:
            label_vec[mark[0]:mark[1]] = 1

        return label_vec

    def extract_all(self):
        """
        extracts all cases provided in data register
        Returns
        -------
        samples : np.ndarray
            tensor of n_cases x n_features
        targets : np.nadarray
            tensor of n_cases x 1

        """
        with mp.Pool(mp.cpu_count()) as p:
            data = p.map(self, range(self.__len__()))
        samples, targets = zip(*data)
        samples = np.concatenate(samples, axis=0)
        targets = np.concatenate(targets, axis=0)
        return samples, targets

    def __len__(self):
        return len(self.data_register)


def extraction_to_disk(data_registers, **kwargs):
    """
    extracts entries in the data_register as paremetrized by **kwargs

    Parameters
    ----------
    data_registers : dictionary
     {'train': 'path', 'validation': 'path', 'test': 'path'}}
    kwargs : dictionary
       the kwargs overwrite the defaults of the DataExtractor

    """
    train_register = pickle.load(open(data_registers['train'], 'rb'))
    validation_register = pickle.load(open(data_registers['validation'], 'rb'))
    test_register = pickle.load(open(data_registers['test'], 'rb'))

    X_train, Y_train = DataExtractor(train_register, **kwargs).extract_all()
    normalizer = Normalizer()
    X_train = normalizer.fit_transform(X_train)
    np.savez('data_train.npz', audio=X_train, target=Y_train)
    del X_train
    del Y_train

    X_validation, Y_validation = DataExtractor(validation_register, **kwargs).extract_all()
    X_validation = normalizer.transform(X_validation)
    np.savez('data_validation.npz', audio=X_validation, target=Y_validation)
    del X_validation
    del Y_validation

    X_test, Y_test = DataExtractor(test_register, **kwargs).extract_all()
    X_test = normalizer.transform(X_test)
    np.savez('data_test.npz', audio=X_test, target=Y_test)
    del X_test
    del Y_test


if __name__ == "__main__":
    data_register = dict(train='../data/data_register_train.pkl',
                         validation='../data/data_register_dev.pkl',
                         test='../data/data_register_test.pkl')
    extraction_arguments = dict(fs=48_000, target_fs=8_000,
                                frame_length=16_000, hop_length=2_000,
                                stft_hoplength=128, n_mels=40,
                                target_speed=50, speed_normalization=True)
    extraction_to_disk(data_register, **extraction_arguments)
