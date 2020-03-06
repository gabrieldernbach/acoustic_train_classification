import pathlib
import xml.etree.ElementTree as ElementTree
from uuid import uuid4

import librosa
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
    f['audio_path'] = path.with_suffix('.wav')
    f['target_path'] = path.with_suffix('.aup')
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


root = '/Users/gabrieldernbach/git/acoustic_train_class/data/'
root = pathlib.Path(root)
paths = list(root.rglob('*.aup'))


def load_meta(path):
    f = {}
    path = pathlib.Path(path)
    f['file_name'] = path.with_suffix('').name
    f['station'] = path.parent.name
    f['train_speed'] = load_train_speed(path)
    f['wheel_diameter'] = load_wheel_diameter(path)
    f['axle'] = load_axle(path)
    f['audio_path'] = path.with_suffix('.wav')
    f['target_path'] = path.with_suffix('.aup')
    return f


resampler = ResampleTrainSpeed()
framer = Frame()
for path in paths:
    meta = load_meta(path)

    audio = load_audio(path)
    target = load_target(path, len(audio))

    audio = framer.split(resampler.down(audio, meta['train_speed']))
    target = framer.split(resampler.down(target, meta['train_speed']))

    write_dir = root.parent / 'framed' / meta['station'] / meta['file_name']
    for i in range(len(audio)):
        uid = str(uuid4())
        audio_path = write_dir / f'{uid}_audio'
        target_path = write_dir / f'{uid}_target'
        np.save(audio_path, audio[i])
        np.save(target_path, target[i])

# register.append(write(audio, target))


register = []
for path in paths:
    f = load_file(path)
    f['audio'] = framer.split(resampler.down(f['audio'], f['train_speed']))
    f['target'] = framer.split(resampler.down(f['target'], f['train_speed']))

    write_dir = root.parent / 'data_framed' / f['station'] / f['file_name']

    write_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(len(f['audio']))):
        id = str(uuid4())
        paudio = write_dir / f'{id}_audio'
        ptarget = write_dir / f'{id}_target'
        np.save(paudio, f['audio'][i])
        np.save(ptarget, f['target'][i])
