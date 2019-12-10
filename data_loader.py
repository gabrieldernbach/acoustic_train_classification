"""
Sets up the data loader with any pre processing applied.
The dataset must be specified by a register locating the files. (see data_build_register.py)
"""

import librosa
import numpy as np
import pandas as pd
import scipy.signal
import torch
from skimage import transform
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from utils import split


class AcousticSceneDataset(Dataset):

    def __init__(self, data_register,
                 transform=None,
                 sr=24000, frame_length=48000,
                 hop_length=12000):
        self.frame_length = frame_length
        self.transform = transform
        self.hop_length = hop_length
        self.sr = sr

        self.audio, self.label, self.station = self.load_in_frames(data_register)

    def load_in_frames(self, data_register):
        """
        loads instances listed in data_register
        applies framing to audio and labels
        stations are one hot encoded alphabetically
        """
        audio, label = [], []
        for i in tqdm(range(len(data_register))):
            # load and frame audio
            audio_path = data_register.audio_path[i]
            audio_raw = librosa.core.load(audio_path,
                                          sr=self.sr,
                                          mono=False)[0][0, :]
            audio_raw = np.ascontiguousarray(audio_raw)
            audio_framed = librosa.util.frame(audio_raw,
                                              frame_length=self.frame_length,
                                              hop_length=self.hop_length)
            audio.append(audio_framed)

            # load and frame labels
            label_vec = self.label_to_vec(data_register.label[i], len(audio_raw))
            label_framed = librosa.util.frame(label_vec,
                                              frame_length=self.frame_length,
                                              hop_length=self.hop_length)
            label.append(label_framed)

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


class PitchShift(object):

    def __init__(self,
                 sr=48000,
                 n_steps=4,
                 bins_per_octave=24.,
                 res_type='kaiser_fast'):
        self.sr = sr
        self.n_steps = n_steps
        self.bins_per_octave = bins_per_octave
        self.res_type = res_type

    def __call__(self, sample):
        sample = librosa.effects.pitch_shift(self.sr,
                                             self.n_steps,
                                             self.bins_per_octave,
                                             self.res_type)
        return sample


class AdjustAmplitude(object):

    def __init__(self, offset_in_db):
        self.offset_in_db = offset_in_db
        self.factor = 10 ** (offset_in_db / 20)

    def __call__(self, sample):
        return self.factor * sample


class Spectrogram(object):

    def __init__(self, nperseg=1024, noverlap=768):
        self.nperseg = nperseg,
        self.noverlap = noverlap

    def __call__(self, sample):
        spec = scipy.signal.spectrogram(sample,
                                        nperseg=self.nperseg,
                                        noverlap=self.noverlap)
        return spec[2]


class PercussiveSeparation(object):

    def __init__(self, margin=3.0):
        self.margin = margin

    def __call__(self, sample):
        return librosa.effects.percussive(sample, self.margin)


class Resize(object):

    def __init__(self, x_length, y_length):
        self.x_length = x_length
        self.y_length = y_length

    def __call__(self, sample):
        # return cv2.resize(sample, (self.x_length, self.y_length))
        return transform.resize(sample, (self.x_length, self.y_length))


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
