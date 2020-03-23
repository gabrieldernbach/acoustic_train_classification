import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from main.extract import ResampleTrainSpeed, Frame
from main.extract import load_audio, load_target, load_axle, load_train_speed, load_wheel_diameter

"""
Collect Observation
    * load wav
    * load csv
    * load aup
"""


def load_file(path):
    f = {}
    path = pathlib.Path(path)
    f['file_name'] = path.with_suffix('').name
    f['station'] = path.parent.name
    f['train_speed'] = load_train_speed(path)
    f['wheel_diameter'] = load_wheel_diameter(path)
    f['axle'] = load_axle(path)
    f['audio'] = load_audio(path)
    f['target'] = load_target(path, len(f['audio']))
    return f


class NumpyDataset(Dataset):
    def __init__(self, sample, transforms):
        self.sample = sample.astype('float32')
        self.transforms = transforms

    def __getitem__(self, item):
        x = self.sample[item]
        x = self.transforms(x)
        return x

    def __len__(self):
        return len(self.sample)


"""
Define Predicitor
    * load model from path
    * transform input data
    * apply model
    * undo input transformation
"""


class Predictor:
    def __init__(self, model_path, resampler, framer, batch_size, num_workers):
        checkpoint = torch.load(model_path)
        from main.models.unet import TinyUnet
        self.model = TinyUnet()
        self.model.load_state_dict(checkpoint['model_parameters'])
        self.model.eval()

        self.resampler = resampler
        self.framer = framer
        self.transforms = checkpoint['transforms']['audio']
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __call__(self, f):
        audio = self.resampler.down(f['audio'], f)
        frames = self.framer.split(audio)

        dl = DataLoader(NumpyDataset(frames, self.transforms),
                        batch_size=self.batch_size,
                        num_workers=self.num_workers)

        with torch.no_grad():
            outs = []
            for d in tqdm(dl):
                out = self.model({'audio': d})
                outs.append(out['target'])

            out = torch.cat(outs)
            out = out.detach().numpy()
            out = out.mean(axis=-1)

        out = self.resampler.up(self.framer.join(out, len(audio)), f)
        return out


"""
Visualize Results
"""


def plot_batch(samples):
    samples = samples.numpy()
    for i in range(len(samples)):
        plt.imshow(samples[i, 0, :, :], vmin=-5, vmax=+5)
        plt.show()


def plot(f):
    n = len(f['target'])
    x = np.linspace(0, n / 48000, n) / 60
    plt.fill_between(x, 0, f['target'], color='C1', alpha=0.5)
    plt.ylim(0, 1)
    plt.axhline(0.5, color='C0', alpha=0.5, lw=0.5)

    n = len(f['out'])
    x = np.linspace(0, n / 48000, n) / 60
    plt.plot(x, f['out'])
    plt.show()


if __name__ == "__main__":
    root = '/Users/gabrieldernbach/git/acoustic_train_class/data/'
    root = pathlib.Path(root)
    paths = list(root.rglob('*.aup'))

    predictor = Predictor(model_path='model.pt',
                          resampler=ResampleTrainSpeed(target_fs=8192, target_train_speed=14),
                          framer=Frame(frame_length=16384, hop_length=4096),
                          batch_size=16,
                          num_workers=4)

    for i in range(10):
        print('predicting train', i)
        f = load_file(paths[i + 100])
        f['out'] = predictor(f)
        plot(f)
