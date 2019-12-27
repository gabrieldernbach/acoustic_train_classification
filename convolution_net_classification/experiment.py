import pickle

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from convolution_net_classification.augmentations import Spectrogram, Resize
from convolution_net_classification.data_loader import AcousticSceneDataset
from convolution_net_classification.model import ResNet
from convolution_net_classification.trainer import Trainer


def split(data, a=0.6, b=0.8):
    """
    Create a random train, dev, test split of a pandas data frame
    """
    a, b = int(a * len(data)), int(b * len(data))
    data_shuffled = data.sample(frac=1).reset_index(drop=True)
    return np.split(data_shuffled, [a, b])


if __name__ == '__main__':
    print('read data set')
    data_register = pickle.load(open('../data/data_register.pkl', 'rb'))
    data_register = data_register[:5]
    train, validation, test = split(data_register)

    transform = transforms.Compose([
        Spectrogram(),  # todo needs normalization
        Resize(224, 224),
        # ExpandDim(),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(AcousticSceneDataset(train, transform=transform), batch_size=50)
    validation_loader = DataLoader(AcousticSceneDataset(validation, transform=transform), batch_size=50)
    test_loader = DataLoader(AcousticSceneDataset(test, transform=transform), batch_size=50)

    model = ResNet

    trainer = Trainer(model=ResNet,
                      criterion=nn.CrossEntropyLoss(),
                      optimizer=optim.Adam(ResNet.parameters(), lr=0.01, betas=(0.9, 0.999)),
                      epochs=400,
                      callback=None,
                      early_stop_patience=20,
                      early_stop_verbose=True
                      )

    trainer.fit(train_loader, validation_loader)
