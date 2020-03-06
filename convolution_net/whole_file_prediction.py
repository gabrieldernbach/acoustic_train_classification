import pathlib
import xml.etree.ElementTree as ElementTree

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

"""
Collect Observation
    * load wav, csv, aup
"""


def load_audio(path):
    audio_path = path.with_suffix('.wav')
    audio, _ = librosa.core.load(audio_path, 48_000, mono=False)
    audio = np.asfortranarray(audio[0])
    return audio


def load_target(path, audio_len):
    aup_path = path.with_suffix('.aup')
    doc = ElementTree.parse(aup_path)
    root = doc.getroot()
    xml_target = r'{http://audacity.sourceforge.net/xml/}label'
    marks_sec = []
    for element in root.iter(xml_target):
        start = element.attrib['t']
        end = element.attrib['t1']
        marks_sec.append((start, end))

    sec2samp = lambda x: round(float(x) * 48_000)
    marks_samp = []
    for m in marks_sec:
        start = sec2samp(m[0])
        end = sec2samp(m[1])
        marks_samp.append([start, end])

    target_vec = np.zeros(audio_len, dtype='float32')
    for m in marks_samp:
        target_vec[m[0]:m[1]] = 1

    return target_vec


def load_train_speed(path):
    csv_path = path.with_suffix('.csv')
    file = pd.read_csv(csv_path, sep=';', decimal=',', dtype=np.float32)
    speeds = file.speedInMeterPerSeconds
    speed_kmh = speeds.mean()
    return speed_kmh


def load_wheel_diameter(path):
    csv_path = path.with_suffix('.csv')
    file = pd.read_csv(csv_path, sep=';', decimal=',', dtype=np.float32)
    diameter = file.DiameterInMM / 1_000  # convert to meter
    for _ in range(5):
        diameter = diameter[diameter != diameter.max()]
    assert not np.any(diameter == 0), f'zero diameter encountered in {path}'
    assert not np.any(diameter == np.NaN), f'nan diameter encountered in {path}'
    return diameter.mean()


def load_axle(path):
    csv_path = path.with_suffix('.csv')
    file = pd.read_csv(csv_path, sep=';', decimal=',', dtype=np.float32)
    axle = file[['axleNumber', 'WaveTimestampInSeconds']]
    return axle


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


"""
shift representation:
    * resample
    * frame
"""


class Resample:
    def __init__(self, target_fs=8_192):
        self.target_fs = target_fs

    def down(self, sequence):
        sequence = librosa.core.resample(sequence, 48_000, self.target_fs)
        return sequence

    def up(self, sequence):
        sequence = librosa.core.resample(sequence, self.target_fs, 48_000)
        return sequence


class ResampleTrainSpeed:
    def __init__(self, target_fs=8_192, target_train_speed=14):
        self.target_fs = target_fs
        self.subsample_ratio = self.target_fs / 48_000
        self.train_speed_target = target_train_speed

    def down(self, sequence, train_speed):
        resample_fs = self.normalized_fs(train_speed)
        sequence = librosa.core.resample(sequence, 48_000, resample_fs)
        return sequence

    def up(self, sequence, train_speed):
        resample_fs = self.normalized_fs(train_speed)
        sequence = librosa.core.resample(sequence, resample_fs, 48_000)
        return sequence

    def normalized_fs(self, train_speed):
        normalization_ratio = np.maximum(train_speed / self.train_speed_target, 0.25)
        resample_fs = int(48_000 * normalization_ratio * self.subsample_ratio)
        return resample_fs


class ResampleBeatFrequency:
    def __init__(self, target_fs=8_192, target_freq=8):
        self.target_freq = target_freq
        self.target_fs = target_fs
        self.subsample_ratio = self.target_fs / 48_000

    def down(self, sequence, train_speed, wheel_diameter):
        resample_fs = self.normalized_fs(train_speed, wheel_diameter)
        sequence = librosa.core.resample(sequence, 48_000, resample_fs)
        return sequence

    def up(self, sequence, train_speed, wheel_diameter):
        resample_fs = self.normalized_fs(train_speed, wheel_diameter)
        sequence = librosa.core.resample(sequence, resample_fs, 48_000)
        return sequence

    def normalized_fs(self, train_speed, wheel_diameter):
        beat_freq = train_speed / (np.pi * wheel_diameter)
        normalization_ratio = beat_freq / self.target_freq
        resample_fs = int(48_000 * normalization_ratio * self.subsample_ratio)
        return resample_fs


class Frame:
    def __init__(self, frame_length=16384, hop_length=8192):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def split(self, sequence):
        frames = librosa.util.frame(sequence, frame_length=self.frame_length, hop_length=self.hop_length)
        return frames.T  # n x d

    def join(self, frames, clip_length):
        sequence = np.zeros(len(frames) * self.hop_length + self.frame_length)
        for i in range(len(frames)):
            start = i * self.hop_length
            end = start + self.frame_length
            sequence[start:end] += frames[i]

        overlap_ratio = self.hop_length / self.frame_length
        sequence *= overlap_ratio

        sequence = sequence[:clip_length]
        return sequence


"""
collect predictions:
    * setup Numpy DataLoader
    * Call model on Audio
"""

import torch
from torch.utils.data import Dataset, DataLoader


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


class Predictor:
    def __init__(self, model_path, resampler, framer, batch_size, num_workers):
        checkpoint = torch.load(model_path)
        self.model = checkpoint['model']()
        self.model.load_state_dict(checkpoint['model_parameters'])
        self.model.eval()

        self.resampler = resampler
        self.framer = framer
        self.transforms = checkpoint['transforms']['sample']
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __call__(self, f):
        audio = self.resampler.down(f['audio'], f['train_speed'], f['wheel_diameter'])
        frames = self.framer.split(audio)

        dl = DataLoader(NumpyDataset(frames, self.transforms), batch_size=self.batch_size, num_workers=self.num_workers)
        with torch.no_grad():
            out = torch.cat([self.model(batch) for batch in tqdm(dl)])
            out = out.detach().numpy()

        out = self.resampler.up(self.framer.join(out, len(audio)), f['train_speed'], f['wheel_diameter'])
        return out


"""
average prediction:
    * shift by offset and add
    * divide by 2
"""

"""
plot 
    * melspectrogram
    * predict_proba
    * true label
    * axis
    
"""


def plot_batch(samples):
    samples = samples.numpy()
    for i in range(len(samples)):
        plt.imshow(samples[i, 0, :, :], vmin=-5, vmax=+5)
        plt.show()


def plot_with_spec(f):
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.imshow(stmt(f['audio']))

    n = len(f['out'])
    x = np.linspace(0, n / 48000, n)
    ax2.plot(x, f['out'])

    n = len(f['target'])
    x = np.linspace(0, n / 48000, n)
    ax2.plot(x, f['target'])
    ax2.set_ylim(0, 1)

    plt.show()


def plot_frames(f):
    framer = Frame(frame_length=48_000 * 5, hop_length=48_000 * 3)

    audio = framer.split(f['audio'])
    target = framer.split(f['target'])
    out = framer.split(f['out'])

    for i in range(len(target)):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.imshow(stmt(audio[i]))

        n = len(target[i])
        x = np.linspace(0, n / 48000, n)
        ax2.fill_between(x, 0, target[i], color='C1', alpha=0.5)

        n = len(out[i])
        x = np.linspace(0, n / 48000, n)
        ax2.plot(x, out[i])

        ax2.set_ylim(0, 1)
        ax2.axhline(0.5, color='C0', alpha=0.5, lw=0.5)

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

    # axle = f['axle']['WaveTimestampInSeconds']
    # axle_id = f['axle']['axleNumber'].int()
    # plt.vlines(axle, ymin=[0], ymax=[1], label=axle_id)

    plt.show()


# def melspecs(instance):
#     frame_length = 5 * 8192
#     frame = Frame(frame_length=frame_length, hop_length=frame_length)
#     sample = frame(instance['sample'])
#     stmt = librosa.feature.melspectrogram(sample, sr=8192, n_fft=512, hop_length=128, n_mels=40)

def stmt(sample):
    ms = librosa.feature.melspectrogram(sample, sr=48000, n_fft=2048, hop_length=512, n_mels=40)
    ms = np.log(ms + 1e-12)
    return ms


root = '/Users/gabrieldernbach/git/acoustic_train_class/data/'
root = pathlib.Path(root)
paths = list(root.rglob('*.aup'))

predictor = Predictor(model_path='model.pt',
                      # resampler=Resample(target_fs=8192),
                      resampler=ResampleBeatFrequency(target_fs=8_192, target_freq=8),
                      framer=Frame(frame_length=16384, hop_length=4096),
                      batch_size=16,
                      num_workers=4)

for i in range(10):
    print('predicting train', i)
    f = load_file(paths[i + 100])
    f['out'] = predictor(f)
    plot(f)
# f = load_file(paths[53])
# f['out'] = predictor(f)
# plot(f)
