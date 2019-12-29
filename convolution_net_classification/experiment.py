import pickle

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from convolution_net_classification.augmentations import Spectrogram, Resize
from convolution_net_classification.data_loader import AcousticSceneDataset, split
from convolution_net_classification.model import ResNet, squeezenet
from convolution_net_classification.trainer import Trainer

if __name__ == '__main__':
    print('read data set')
    data_register = pickle.load(open('../data/data_register.pkl', 'rb'))
    # data_register = data_register.sample(n=30).reset_index(drop=True)
    train, validation, test = split(data_register)

    transform = transforms.Compose([
        Spectrogram(),  # todo needs normalization
        Resize(224, 224),
        # ExpandDim(),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(AcousticSceneDataset(train, transform=transform), batch_size=50, num_workers=2)
    validation_loader = DataLoader(AcousticSceneDataset(validation, transform=transform), batch_size=50, num_workers=2)
    test_loader = DataLoader(AcousticSceneDataset(test, transform=transform), batch_size=50, num_workers=2)

    print('start training')
    trainer = Trainer(model=squeezenet,
                      criterion=nn.CrossEntropyLoss(),
                      optimizer=optim.Adam(ResNet.parameters(), lr=0.01, betas=(0.9, 0.999)),
                      epochs=400,
                      callback=None,
                      early_stop_patience=20,
                      early_stop_verbose=True
                      )

    trainer.fit(train_loader, validation_loader)
