"""
Sets up the data loader with customizable pre processing.
The dataset must be specified by a register locating the files. (see data_build_register.py)
"""

import multiprocessing as mp

import librosa
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, WeightedRandomSampler


class MelDataset(Dataset):

    def __init__(self, datapath, transform=None):
        self.datapath = datapath
        self.transform = transform

        data = np.load(datapath, allow_pickle=True)
        self.inputs = torch.from_numpy(data['audio']).float()
        self.station = torch.from_numpy(data['station']).long()
        self.labels = torch.from_numpy(data['label']).float()

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        context = self.station[idx]  # todo: need to implement context reading
        labels = self.labels[idx]

        if self.transform:
            inputs = self.transform(inputs)

        inputs = inputs.unsqueeze(0)
        labels = labels.unsqueeze(0)

        return inputs, labels

    def __len__(self):
        return len(self.inputs)

    def __repr__(self):
        return f'{self.__class__.__name__}'


class RawDataset(Dataset):

    def __init__(self,
                 data_register,
                 transform=None,
                 sr=24000,
                 frame_length=48000,
                 hop_length=12000):
        self.data_register = data_register
        self.data_register['station'] = self.data_register['station'].factorize()[0]
        self.frame_length = frame_length
        self.transform = transform
        self.hop_length = hop_length
        self.sr = sr

        # checkpoint_path = '../data/subsampeled.pkl'
        # if os.path.exists(checkpoint_path):
        #     print('reading resampeled data from previous checkpoint')
        #     self.audio, self.context, self.label = pickle.load(open(checkpoint_path, 'rb'))
        # else:
        #     print('extracting and resampling from raw dataset')
        #     self.audio, self.context, self.label = self.load_in_frames(data_register)
        #     print(f'saving resampled files to {checkpoint_path}')
        #     pickle.dump((self.audio, self.context, self.label),
        #                 open(checkpoint_path, 'wb'), protocol=4)
        self.audio, self.context, self.label = self.load_in_frames(data_register)

    def read_from_register(self, idx):
        print(f'reading entry {idx}')
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
        # load an frame context identifier
        context = self.data_register.station[idx]
        context_framed = np.repeat(context, label_framed.shape[1])

        return audio_framed, context_framed, label_framed

    def load_in_frames(self, data_register):
        """
        loads instances listed in data_register,
        applies framing to audio and labels,
        stations are one hot encoded alphabetically
        """
        print(f'starting pool with {8} workers')
        self.data_register.reset_index(inplace=True)
        with mp.Pool(8) as p:
            data_framed = p.map(self.read_from_register, range(len(self.data_register)))
        # data_framed = [self.read_from_register(i) for i in range(len(self.data_register))]

        audio, context, label = list(zip(*data_framed))
        print('concatenate audio')
        audio = np.concatenate(audio, axis=-1).T
        print('concatenate context')
        context = np.concatenate(context, axis=-1).T
        print('concatenate labels')
        labels = np.concatenate(label, axis=-1).T

        return audio, context, labels

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

    def __getitem__(self, idx):
        sample = self.audio[idx]
        target = self.label[idx]
        context = self.context[idx]

        # convert labels
        target = np.array(target.sum() / self.frame_length).astype('float32')
        target = torch.from_numpy(target).long()
        context = torch.tensor(context).long()

        # todo implement mixup

        if self.transform:
            sample = self.transform(sample)

        return sample, context, target

    def __len__(self):
        return len(self.audio)

    def __repr__(self):
        return f'{self.__class__.__name__}'


def split(data, a=0.6, b=0.8):
    """
    Create a random titemrain, dev, test split of a pandas data frame
    """
    a, b = int(a * len(data)), int(b * len(data))
    data_shuffled = data.sample(frac=1, random_state=1).reset_index(drop=True)
    train, validation, test = np.split(data_shuffled, [a, b])
    validation.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    return train, validation, test


def class_imbalance_sampler(labels, threshold=0.35):
    """
    Takes integer class labels and returns the torch sampler
    for balancing the class prior distribution
    """
    labels = (labels > threshold).long()
    class_count = torch.bincount(labels)
    weighting = 1. / class_count.float()
    weights = weighting[labels]
    sampler = WeightedRandomSampler(weights, len(labels))
    return sampler


def evaluate_model(model, subset):
    inputs, contexts, labels = subset
    # predictions = F.softmax(model(inputs, contexts), dim=1)[:, 1].detach().numpy()  # multiclass
    predictions = model(inputs, contexts).detach().numpy()
    labels = labels.detach().numpy() > 0.35
    roc = roc_auc_score(labels, predictions)
    f1 = f1_score(labels, predictions > 0.5)
    confmat = confusion_matrix(labels, predictions > 0.5)
    return roc, f1, confmat


if __name__ == '__main__':
    # test RawDataset
    # print('read register')
    # df = pd.read_pickle('../data/data_register.pkl')
    # print('split data')
    # df = df[:3]
    # train, dev, test = split(df)
    # print('load train set')
    # composed = transforms.Compose([Spectrogram(nperseg=1024, noverlap=768),
    #                                Resize(224, 224)])
    # train = RawDataset(train, transform=composed)
    # print(train[1])
    # print('test success')

    # test MelDataset
    dev_path = 'mel_dev.npz'
    devset = MelDataset(dev_path)
    print(devset[1])
