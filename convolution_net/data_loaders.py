"""
Save dataset under mel spectrogram transformation for convolutional network
"""
import os

import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import WeightedRandomSampler


def fetch_dummy():
    """
    light weight data loaders for debugging new models
    Returns
    -------
    train_dl, validation_dl, test_dl : Torch DataLoader

    """
    train_dl = DataLoader(TensorDataset(torch.randn(200, 1, 40, 126), torch.rand(200, 1)), batch_size=5)
    validation_dl = DataLoader(TensorDataset(torch.randn(200, 1, 40, 126), torch.rand(200, 1)), batch_size=5)
    test_dl = DataLoader(TensorDataset(torch.randn(200, 1, 40, 126), torch.rand(200, 1)), batch_size=5)
    return train_dl, validation_dl, test_dl


def class_imbalance_sampler(data_set, threshold=0.125):
    """
    infer class imbalance and setup weighted sampler

    Parameters
    ----------
    data_set : torch Dataset
        class indicator expected in data_set.targets
    threshold : int
        threshold to be applied on soft labels

    Returns
    -------
    sampler : torch object

    """
    targets = data_set.targets
    targets = targets > threshold
    targets = tensor(targets).long().squeeze()
    class_count = torch.bincount(targets)
    weighting = tensor(1.) / class_count.float()
    weights = weighting[targets]
    sampler = WeightedRandomSampler(weights, len(targets))
    return sampler


class AcousticFlatSpotDataset(Dataset):
    """
    setup data loader from extracted npz files,
    allows for input transformations

    Parameters
    ----------
    phase : ['train', 'validation', 'test']
        determines the split to be loaded
    transforms: torch.transforms, function, None
        data augmentations to be applied
    """

    def __init__(self, phase, memmap, transforms=None):
        assert phase in ['train', 'validation', 'test']
        file_name = f'/data_{phase}'
        parent_path = os.path.dirname(os.path.realpath(__file__))
        data_path = parent_path + file_name

        print(f'loading {file_name}')
        mmap_mode = 'r' if memmap else None
        self.samples = np.load(f'{data_path}_samples.npy', mmap_mode=mmap_mode)
        self.targets = np.load(f'{data_path}_targets.npy', mmap_mode=mmap_mode)

        self.transforms = transforms

    def __getitem__(self, idx):
        sample, target = self.samples[idx], self.targets[idx]

        sample = np.expand_dims(sample, axis=0)
        if self.transforms:
            sample = self.transforms(sample)

        target = torch.from_numpy(target).float()
        return sample, target

    def __len__(self):
        return self.samples.shape[0]


def fetch_dataloaders(batch_size, num_workers, memmap, train_tfs, validation_tfs):
    dl_args = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}

    train_set = AcousticFlatSpotDataset('train', memmap, transforms=train_tfs)
    validation_set = AcousticFlatSpotDataset('validation', memmap, transforms=validation_tfs)
    test_set = AcousticFlatSpotDataset('test', memmap, transforms=validation_tfs)

    sampler = class_imbalance_sampler(train_set)
    train_dl = DataLoader(train_set, **dl_args, sampler=sampler)
    validation_dl = DataLoader(validation_set, **dl_args)
    test_dl = DataLoader(test_set, **dl_args)

    return train_dl, validation_dl, test_dl


if __name__ == "__main__":
    train_dl, validation_dl, test_dl = fetch_dataloaders(64, None)
    print(next(iter(train_dl)))
