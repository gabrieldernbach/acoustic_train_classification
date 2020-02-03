"""
This module defines classes for feature extraction from the raw data set.
And writes them to disk (numpy npz)

Either call this script by itself or
import the `extract_to_disk` function
"""

import multiprocessing as mp
import pickle

import librosa
import numpy as np


class LoadAudio:
    def __init__(self, fs=48000, channel=0):
        self.fs = fs
        self.channel = channel

    def __call__(self, _, register_row):
        audio_path = register_row.audio_path
        audio, _ = librosa.core.load(audio_path, self.fs, mono=False)
        audio = np.asfortranarray(audio[self.channel])
        return audio


class Resample:
    def __init__(self, fs=48000, target_fs=8000):
        self.fs = fs
        self.target_fs = target_fs

    def __call__(self, samples, _):
        return librosa.core.resample(samples, self.fs, self.target_fs)


class ResampleSpeedNormalization:
    def __init__(self, fs=48000, target_fs=8000, target_speed=50):
        self.fs = fs
        self.target_fs = target_fs
        self.target_speed = target_speed
        self.subsample = self.fs / self.target_fs

    def __call__(self, samples, register_row):
        speed = register_row.speed_kmh
        resampling_ratio = np.maximum(speed / self.target_speed, 0.25)
        self.resample_fs = int(self.fs * resampling_ratio / self.subsample)
        samples = librosa.core.resample(samples, self.fs, self.target_fs)
        return samples


class Frame:
    def __init__(self, frame_length=16000, hop_length=2000):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def __call__(self, samples, _):
        return librosa.util.frame(samples, self.frame_length, self.hop_length).T


class ShortTermMelTransform:
    def __init__(self, fs=8000, n_fft=512, hop_length=128, n_mels=40):
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, framed_samples, _):
        samples = np.stack([self.stmt(x) for x in framed_samples])
        return samples

    def stmt(self, samples):
        x = librosa.feature.melspectrogram(samples, sr=self.fs, n_fft=self.n_fft,
                                           hop_length=self.hop_length, n_mels=self.n_mels)
        x = np.log(x + 1e-12)
        return x


class LoadTargets:
    def __init__(self, fs=48000):
        self.fs = fs

    def __call__(self, _, register_row):
        audio_path = register_row.audio_path
        marks_in_s = register_row.target
        audio, _ = librosa.core.load(audio_path, self.fs, mono=False)
        len_sequence = len(audio[0])

        mark_in_samp = []
        for mark in marks_in_s:
            start = round(float(mark[0]) * self.fs)
            end = round(float(mark[1]) * self.fs)
            mark_in_samp.append([start, end])

        target_vec = np.zeros(len_sequence)
        for mark in mark_in_samp:
            target_vec[mark[0]:mark[1]] = 1

        return target_vec


class AvgPoolTargets:
    def __init__(self, threshold=0.125):
        self.threshold = threshold

    def __call__(self, targets, _):
        frame_length = targets.shape[1]
        targets = ((targets.sum(1) / frame_length) > self.threshold)[:, None]
        return targets


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
        data = data - self.xm
        data = data / self.xv
        return data

    def fit_transform(self, data):
        self.fit(data)
        data = self.transform(data)
        return data


class RegisterExtractor:
    def __init__(self, data_register, sample_tfs, target_tfs):
        self.data_register = pickle.load(open(data_register, 'rb'))
        self.sample_tfs = sample_tfs
        self.target_tfs = target_tfs

    def __getitem__(self, idx):
        print(f'start processing instance {idx} of {self.__len__()}')
        register_row = self.data_register.iloc[idx]

        samples = []
        for tfs in self.sample_tfs:
            samples = tfs(samples, register_row)

        targets = []
        for tfs in self.target_tfs:
            targets = tfs(targets, register_row)

        return samples, targets

    def __call__(self, idx):
        return self.__getitem__(idx)

    def extract_all(self):
        with mp.Pool(mp.cpu_count()) as p:
            data = p.map(self, range(self.__len__()))
        samples, targets = zip(*data)
        samples = np.concatenate(samples, axis=0)
        targets = np.concatenate(targets, axis=0)
        return samples, targets

    def __len__(self):
        return len(self.data_register)


def extract_to_disk(sample_tfs, target_tfs):
    data_registers = dict(train='../data/data_register_train.pkl',
                          validation='../data/data_register_dev.pkl',
                          test='../data/data_register_test.pkl')

    normalizer = Normalizer()

    for split in ['train', 'validation', 'test']:
        register_extractor = RegisterExtractor(data_registers[split], sample_tfs, target_tfs)
        samples, targets = register_extractor.extract_all()
        if split == 'train':
            samples = normalizer.fit_transform(samples)
        else:
            samples = normalizer.transform(samples)
        np.save(f'data_{split}_samples.npy', samples)
        np.save(f'data_{split}_targets.npy', targets)
        del samples
        del targets


if __name__ == "__main__":
    sample_tfs = [LoadAudio(fs=48000),
                  ResampleSpeedNormalization(target_fs=8000, target_speed=50),
                  Frame(frame_length=16000, hop_length=4000),
                  ShortTermMelTransform(fs=8000, n_fft=512, hop_length=128, n_mels=40)]
    target_tfs = [LoadTargets(fs=48000),
                  ResampleSpeedNormalization(target_fs=8000, target_speed=50),
                  Frame(frame_length=16000, hop_length=4000),
                  AvgPoolTargets(threshold=0.125)]

    extract_to_disk(sample_tfs, target_tfs)
