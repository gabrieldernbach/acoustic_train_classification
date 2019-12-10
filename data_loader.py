"""
Sets up the data loader with any pre processing applied.
The dataset must be specified by a register locating the files. (see data_build_register.py)
"""

import multiprocessing as mp

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from augmentations import Resize, Spectrogram
from utils import split


class AcousticSceneDataset(Dataset):

    def __init__(self,
                 data_register,
                 transform=None,
                 sr=24000,
                 frame_length=48000,
                 hop_length=12000):
        self.data_register = data_register
        self.frame_length = frame_length
        self.transform = transform
        self.hop_length = hop_length
        self.sr = sr

        self.audio, self.label, self.station = self.load_in_frames(data_register)

    def read_from_register(self, idx):
        print(f'read {idx}')
        # load and frame audio
        audio_path = self.data_register.audio_path[idx]
        audio_raw = librosa.core.load(audio_path,
                                      sr=self.sr,
                                      mono=False)[0][0, :]
        audio_raw = np.ascontiguousarray(audio_raw)
        audio_framed = librosa.util.frame(audio_raw,
                                          frame_length=self.frame_length,
                                          hop_length=self.hop_length)
        # load and frame labels
        label_vec = self.label_to_vec(self.data_register.label[idx], len(audio_raw))
        label_framed = librosa.util.frame(label_vec,
                                          frame_length=self.frame_length,
                                          hop_length=self.hop_length)
        return audio_framed, label_framed

    def load_in_frames(self, data_register):
        """
        loads instances listed in data_register
        applies framing to audio and labels
        stations are one hot encoded alphabetically
        """
        audio, label = [], []
        print('starting pool')
        with mp.Pool(mp.cpu_count()) as p:
            data_framed = p.map(self.read_from_register, range(len(self.data_register)))

        audio, labels = list(zip(*data_framed))
        print('concatenate audio')
        audio = np.concatenate(audio, axis=-1).T
        print('concatenate labels')
        labels = np.concatenate(label, axis=-1)
        print('perform one hot encoding')
        station = np.array(pd.get_dummies(train.station))

        return audio, labels, station

    def label_to_vec(self, label_in_seconds, len_sequence):
        """
        convert the labels in seconds into a time series label vector
        """
        mark_in_samp = []
        for mark in label_in_seconds:
            start = round(float(mark[0]) * self.sr)
            end = round(float(mark[1]) * self.sr)
            mark_in_samp.append([start, end])

        label_vec = np.zeros(len_sequence)
        for mark in mark_in_samp:
            label_vec[mark[0]:mark[1]] = 1

        return label_vec

    def __getitem__(self, item):
        sample = self.audio[item]
        target = self.label[item]

        # todo implement mixup

        if self.transform:
            sample = self.transform(sample)

        sample = torch.from_numpy(sample)
        target = torch.from_numpy(target)

        return sample, target

    def __len__(self):
        return len(self.audio)

    def __repr__(self):
        return f'{self.__class__.__name__}'


if __name__ == '__main__':
    print('read register')
    df = pd.read_pickle('data_register.pkl')
    # optionally condition on station
    # df = df[df.station['VHB']]
    print('split data')
    train, dev, test = split(df)
    print('load train set')
    composed = transforms.Compose([Spectrogram(nperseg=1024, noverlap=768),
                                   Resize(300, 300)])
    train = AcousticSceneDataset(train)
