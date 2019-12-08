import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import split


class AcousticSceneDataset(Dataset):

    def __init__(self, data_register, sr=24000, frame_length=48000, hop_length=12000):
        self.data_register = data_register
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.sr = sr

        self.audio, self.label, self.station = self.load_framed(data_register)

    def load_framed(self, data_register):
        """
        loads instances listed in data_register
        applies framing to audio and labels
        stations are one hot encoded alphabetically
        """
        # multi process by
        # paths = list(zip(data_register.audio_path, data_register.label))
        audio_path = data_register.audio_path
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
        # audio = np.concatenate(audio, axis=-1).T
        audio = np.hstack(audio).T
        print('concatenate labels')
        # labels = np.concatenate(label, axis=-1)
        labels = np.hstack(label)
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

        return self.audio[item], self.label[item]

    def __len__(self):
        return len(self.audio)

    def __repr__(self):
        return f'{self.__class__.__name__}'


if __name__ == '__main__':
    print('read register')
    df = pd.read_pickle('data_register.pkl')
    print('split data')
    train, dev, test = split(df)
    print('load train set')
    train = AcousticSceneDataset(train)
