from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import tensor
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from convolution_net.augment import Normalizer


def path2entry(path):
    entry = {
        'audio_path': path,
        'target_path': Path(str(path).replace('_audio.npy', '_target.npy')),
        'file_name': path.parent,
        'speed_bucket': path.parent.parent.name,
        'station': path.parent.parent.parent.name,
    }
    return entry


def build_register(root):
    root = Path(root)
    source_paths = list(root.rglob('*audio.npy'))
    print('indexing dataset')
    register = pd.DataFrame([path2entry(p) for p in tqdm(source_paths)])

    print('label encode')
    register['station_id'] = register.station.astype('category').cat.codes
    register['speed_id'] = register.speed_bucket.astype('category').cat.codes
    return register


def group_split(register, random_state, group='file_name'):
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=random_state)
    split = list(gss.split(register, groups=register[group].values))[0]
    return register.iloc[split[0]], register.iloc[split[1]]


def train_dev_test(register, subset_fraction=1.0, random_state=5):
    remain, test = group_split(register, random_state=random_state)
    remain = remain.sample(frac=subset_fraction)
    train, dev = group_split(remain, random_state=random_state)
    return {'train': train, 'dev': dev, 'test': test}


class RailWatchDataset(Dataset):
    def __init__(self, register, transform=None):
        self.register = register
        self.transform = transform

    def __getitem__(self, item):
        row = self.register.iloc[item]
        out = {}

        audio = np.load(row.audio_path, allow_pickle=True)
        target = np.load(row.target_path, allow_pickle=True)
        out['audio'] = self.transform['audio'](audio)
        out['target'] = self.transform['target'](target)

        out['station_id'] = torch.tensor(row.station_id).long()
        out['speed_id'] = torch.tensor(row.speed_id).long()

        return out

    def __len__(self):
        return len(self.register)


def class_imbalance_sampler(targets, segmentation_threshold):
    if len(targets.shape) > 1:  # if posed as segmentation task
        targets = targets.sum(axis=1) / targets.shape[1]
        targets = targets > segmentation_threshold

    targets = tensor(targets).long().squeeze()
    class_count = torch.bincount(targets)
    weighting = tensor(1.) / class_count.float()
    weights = weighting[targets]
    sampler = WeightedRandomSampler(weights, len(targets))
    return sampler


def infer_stats(data_loader, slide_treshold):
    mean = 0.
    std = 0.
    nb_samples = 0.
    targets = []
    for batch in tqdm(data_loader, desc='infer data stats'):
        samples = batch['audio']
        batch_samples = samples.size(0)
        samples = samples.view(batch_samples, samples.size(1), -1)
        mean += samples.mean(2).sum(0)
        std += samples.std(2).sum(0)
        nb_samples += batch_samples
        targets.append(batch['target'].numpy())

    mean = mean / nb_samples
    std = std / nb_samples
    normalizer = Normalizer(mean=mean, std=std)

    targets = np.concatenate(targets, axis=0)
    sampler = class_imbalance_sampler(targets, slide_treshold)
    return normalizer, sampler


def fetch_dataloaders(registers, dl_args, train_tfs, dev_tfs, slide_threshold):
    # infer normalization
    tmp_set = RailWatchDataset(registers['train'], transform=train_tfs)
    normalizer, sampler = infer_stats(DataLoader(tmp_set, **dl_args), slide_threshold)
    train_tfs['audio'].transforms.append(normalizer)
    dev_tfs['audio'].transforms.append(normalizer)

    # setup data set
    train_set = RailWatchDataset(registers['train'], transform=train_tfs)
    dev_set = RailWatchDataset(registers['dev'], transform=dev_tfs)
    test_set = RailWatchDataset(registers['test'], transform=dev_tfs)

    # build dataloaders
    train_dl = DataLoader(train_set, **dl_args, sampler=sampler)
    dev_dl = DataLoader(dev_set, **dl_args)
    test_dl = DataLoader(test_set, **dl_args)

    return {'train': train_dl, 'dev': dev_dl, 'test': test_dl}
