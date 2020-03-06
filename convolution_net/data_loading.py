"""
Save dataset under mel spectrogram transformation for convolutional network
"""
import os

import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from data_augmentation import Normalizer


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
    if len(targets.shape) > 1:
        targets = targets.sum(axis=1) / targets.shape[1]
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
        mmap_mode = 'r+' if memmap else None
        self.samples = np.load(f'{data_path}_samples.npy', mmap_mode=mmap_mode)
        self.targets = np.load(f'{data_path}_targets.npy', mmap_mode=mmap_mode)

        self.transforms = transforms

    def __getitem__(self, idx):
        sample, target = self.samples[idx], self.targets[idx]

        if self.transforms['sample']:
            sample = self.transforms['sample'](sample)

        if self.transforms['target']:
            target = self.transforms['target'](target)

        return sample, target

    def __len__(self):
        return self.samples.shape[0]


def fetch_dataloaders(batch_size, num_workers, train_tfs, validation_tfs, memmap=False, normalize=True):
    """
    Fetch data loaders and equip them with necessary normalization and imbalance sampler

    Parameters
    ----------
    batch_size : int
        Batch Size
    num_workers : int
        Recommended to be 4 per GPU, 1 for debugging
    train_tfs : Composed Transformations
        Training Transformations, possibly including augmentation
    validation_tfs : Composed Transformations
        Validataion / Test Transformations without data augmentation
    memmap : bool
        If True data sets are lazy loaded from disk which largly decreases memory footprint
        but requires a fast hard drive (ssd)

    Returns
    -------
        train_dl : Torch DataLoader
        validation_dl : Torch DataLoader
        test_dl : Torch DataLoader
        normalizer : transformation
    """
    dl_args = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}

    train_set = AcousticFlatSpotDataset('train', memmap, transforms=train_tfs)
    validation_set = AcousticFlatSpotDataset('validation', memmap, transforms=validation_tfs)
    test_set = AcousticFlatSpotDataset('test', memmap, transforms=validation_tfs)

    if normalize:
        normalizer = Normalizer(DataLoader(train_set, **dl_args))
        train_set.transforms['sample'].transforms.append(normalizer)
        validation_set.transforms['sample'].transforms.append(normalizer)
        ### do not append to test set, they share the transformation already ###
    else:
        normalizer = None

    sampler = class_imbalance_sampler(train_set)
    train_dl = DataLoader(train_set, **dl_args, sampler=sampler)
    validation_dl = DataLoader(validation_set, **dl_args)
    test_dl = DataLoader(test_set, **dl_args)

    return train_dl, validation_dl, test_dl, normalizer


def plot_network_input(data_loader, n_samples):
    """
    Parameters
    ----------
    data_loader : troch DataLoader
    n_samples : int
    """
    import matplotlib.pyplot as plt
    samples, targets = next(iter(data_loader))
    for i in range(n_samples):
        plt.imshow(samples[i][0], vmin=-3, vmax=3)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    train_dl, validation_dl, test_dl, normalizer = fetch_dataloaders(batch_size=64,
                                                                     num_workers=1,
                                                                     memmap=True,
                                                                     train_tfs=None,
                                                                     validation_tfs=None)
    print(next(iter(train_dl)))
