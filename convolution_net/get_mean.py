import pickle

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from convolution_net.data_augmentations import Resize, MelSpectrogram
from convolution_net.data_loader import AcousticSceneDataset, split

data_register = pickle.load(open('../data/data_register.pkl', 'rb'))
data_register = data_register.sample(n=6).reset_index(drop=True)
train, dev, test = split(data_register)

transform = transforms.Compose([
    MelSpectrogram(),
    Resize(224, 224)  # todo needs normalization
])

train_loader = DataLoader(AcousticSceneDataset(train),
                          batch_size=50,
                          num_workers=4)

labels = []
for i, (sample, context, label) in enumerate(train_loader):
    labels.append(label.numpy())
labels = np.concatenate(labels)
labels = (labels != 0) * 1
ratio = np.bincount(labels)
weights = 1. / ratio
sample_weights = weights[labels]

# hot =label
# print(f'plot image {i}')
# plt.clf()
# plt.imshow(data[0][0, :, :])
# plt.colorbar()
# plt.savefig(f'plots/spectrogram_{i}.png')
