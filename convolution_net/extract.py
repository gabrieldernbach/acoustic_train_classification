import xml.etree.ElementTree as ElementTree
from pathlib import Path
from uuid import uuid4

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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
    diameter[diameter == 0] = np.NaN  # treat 0 as nan
    diameter.ffill(inplace=True)  # forward fill previous
    return diameter.mean()


def load_axle(path):
    csv_path = path.with_suffix('.csv')
    file = pd.read_csv(csv_path, sep=';', decimal=',', dtype=np.float32)
    axle = file[['axleNumber', 'WaveTimestampInSeconds']]
    return axle


"""
shift representation:
    * resample
    * frame
"""


class Resample:
    def __init__(self, target_fs=8_192, **kwargs):
        self.target_fs = target_fs

    def down(self, sequence, register_row):
        sequence = librosa.core.resample(sequence, 48_000, self.target_fs)
        return sequence

    def up(self, sequence):
        sequence = librosa.core.resample(sequence, self.target_fs, 48_000)
        return sequence


class ResampleTrainSpeed:
    def __init__(self, target_fs=8_192, target_train_speed=14, **kwargs):
        self.target_fs = target_fs
        self.subsample_ratio = self.target_fs / 48_000
        self.train_speed_target = target_train_speed

    def down(self, sequence, register_row):
        train_speed = register_row['train_speed']
        resample_fs = self.normalized_fs(train_speed)
        sequence = librosa.core.resample(sequence, 48_000, resample_fs)
        return sequence

    def up(self, sequence, register_row):
        train_speed = register_row['train_speed']
        resample_fs = self.normalized_fs(train_speed)
        sequence = librosa.core.resample(sequence, resample_fs, 48_000)
        return sequence

    def normalized_fs(self, train_speed):
        normalization_ratio = np.maximum(train_speed / self.train_speed_target, 0.25)
        resample_fs = int(48_000 * normalization_ratio * self.subsample_ratio)
        return resample_fs


class ResampleBeatFrequency:
    def __init__(self, target_fs=8_192, target_freq=8, **kwargs):
        self.target_freq = target_freq
        self.target_fs = target_fs
        self.subsample_ratio = self.target_fs / 48_000

    def down(self, sequence, register_row):
        train_speed = register_row['train_speed']  # in meter per second
        wheel_diameter = register_row['wheel_diameter']  # in meter
        resample_fs = self.normalized_fs(train_speed, wheel_diameter)
        sequence = librosa.core.resample(sequence, 48_000, resample_fs)
        return sequence

    def up(self, sequence, register_row):
        train_speed = register_row['train_speed']
        wheel_diameter = register_row['wheel_diameter']
        resample_fs = self.normalized_fs(train_speed, wheel_diameter)
        sequence = librosa.core.resample(sequence, resample_fs, 48_000)
        return sequence

    def normalized_fs(self, train_speed, wheel_diameter):
        beat_freq = train_speed / (np.pi * wheel_diameter)
        normalization_ratio = np.maximum(beat_freq / self.target_freq, 0.25)
        resample_fs = int(48_000 * normalization_ratio * self.subsample_ratio)
        return resample_fs


class Frame:
    def __init__(self, frame_length=16384, hop_length=8192, **kwargs):
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


def load_meta(path):
    m = {}
    path = Path(path)
    m['file_name'] = path.with_suffix('').name
    m['station'] = path.parent.name
    m['train_speed'] = load_train_speed(path)
    m['wheel_diameter'] = load_wheel_diameter(path)
    m['audio_path'] = path.with_suffix('.wav')
    m['target_path'] = path.with_suffix('.aup')
    return m


class Extractor:
    def __init__(self, destination, resampler, framer):
        self.root = destination
        self.destination = destination
        self.resampler = resampler
        self.framer = framer

    def __call__(self, r):
        audio = load_audio(r['audio_path'])
        target = load_target(r['target_path'], len(audio))

        audio = self.framer.split(self.resampler.down(audio, r))
        target = self.framer.split(self.resampler.down(target, r))

        write_dir = (
                self.root.parent
                / self.destination
                / r['station']
                / str(r['speed_bucket'])
                / r['file_name']
        )

        write_dir.mkdir(parents=True, exist_ok=True)

        for i in range(len(audio)):
            fname = str(write_dir / str(uuid4()))
            np.save(fname + '_audio', audio[i])
            np.save(fname + '_target', target[i])


def create_dataset(source_path, destination_path, resampler, framer):
    source = Path(source_path)
    destination = Path(destination_path)

    # import shutil
    # if destination.exists():
    #     print('clean up destination')
    #     shutil.rmtree(destination)

    print('indexing source files')
    source_paths = list(source.rglob('*.aup'))
    register = pd.DataFrame([load_meta(p) for p in tqdm(source_paths)])
    register['speed_bucket'] = pd.cut(register.train_speed, bins=10, labels=False)

    print('extracting to disk')
    extractor = Extractor(destination, resampler, framer)
    Parallel(4, verbose=10)(delayed(extractor)(r) for _, r in register.iterrows())


if __name__ == "__main__":
    sr = 8192
    # resampler = Resample(sr)
    # framer = Frame(sr * 5, sr * 5)
    # create_dataset('data_resample_sub', resampler, framer)

    resampler = ResampleTrainSpeed(sr, target_train_speed=14)
    framer = Frame(sr * 5, sr * 5)
    create_dataset('data_resample_train_5s', resampler, framer)

    resampler = ResampleTrainSpeed(sr, target_train_speed=14)
    framer = Frame(sr * 2, int(sr * 0.5))
    create_dataset('data_resample_train_5s', resampler, framer)

    # resampler = ResampleBeatFrequency(sr, target_freq=8)
    # framer = Frame(sr * 5, sr * 5)
    # create_dataset('data_resample_freq', resampler, framer)
