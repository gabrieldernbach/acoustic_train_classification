import multiprocessing as mp
import os
import pickle
import xml.etree.ElementTree as ElementTree
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
from librosa.util import frame
from tqdm import tqdm

from utils import split


def mark_to_vec(marks_in_s, len_sequence):
    '''
    convert the marks ins seconds into a time series label vector
    and track durations of the marked sections
    '''
    durations = []
    mark_in_samp = []
    for mark in marks_in_s:
        durations.append(float(mark[1]) - float(mark[0]))
        start = round(float(mark[0]) * 8000)
        end = round(float(mark[1]) * 8000)
        mark_in_samp.append([start, end])

    label_vec = np.zeros(len_sequence)
    for mark in mark_in_samp:
        label_vec[mark[0]:mark[1]] = 1
    detection = np.alltrue(label_vec == 0)

    return label_vec, durations, detection


def extract_aup(aup_path):
    '''
    Extract audio and annotations from a single audacity project
    :param aup_path: abolute file path to audacity project
    :return: station(global var), name, audio, marks_in_s, label_vec, durations
    '''
    # parse xml
    doc = ElementTree.parse(aup_path)
    root = doc.getroot()

    # load wavfile
    xml_wave = r'{http://audacity.sourceforge.net/xml/}wavetrack'
    name = root.find(xml_wave).attrib['name'] + '.wav'
    print(f'extracting data point {name}')
    audio = librosa.core.load(data_path + '/' + name, 8000, mono=False)[0][0, :]
    audio_len = len(audio)

    # extract labels
    xml_label = r'{http://audacity.sourceforge.net/xml/}label'
    marks_in_s: List[Tuple[str, str]] = []
    for element in root.iter(xml_label):
        start = element.attrib['t']
        end = element.attrib['t1']
        marks_in_s.append((start, end))
    label_vec, durations, detection = mark_to_vec(marks_in_s, audio_len)

    return station, name, audio, marks_in_s, label_vec, durations, detection


def mfcc_vec(x):
    def mfcc(x):
        return librosa.feature.mfcc(x, sr=8000).flatten()

    return np.apply_along_axis(mfcc, 0, x)


def transform(dataset):
    X = dataset.audio
    S = dataset.station
    Y = dataset.label_vec

    print('starting transform')
    X = [frame(x, frame_length=8000, hop_length=1000) for x in X]
    Y = [frame(y, frame_length=8000, hop_length=1000) for y in Y]

    print('start mfcc')
    X = [mfcc_vec(x) for x in tqdm(X)]

    X = np.concatenate(X, axis=-1).T
    Y = np.concatenate(Y, axis=-1)
    Y = (Y.sum(axis=0) / 8000)  # do not flatten for segmentation

    return X, S, Y


if __name__ == '__main__':
    cwd = os.getcwd()
    stations = os.listdir(cwd + '/data')
    stations = [s for s in stations if not s.startswith('.')]
    print(f'found sub folders for station {stations}')

    data = []
    for station in stations:
        print(f'loading station {station}')
        data_path = cwd + '/data/' + station
        files = os.listdir(data_path)
        audacity_projects = [f for f in files if f.endswith('.aup')]
        project_paths = [[os.path.join(data_path, aup)] for aup in audacity_projects]
        print(f'found {len(project_paths)} files')

        with mp.Pool(mp.cpu_count()) as p:
            station_data = p.starmap(extract_aup, project_paths)

        data.extend(station_data)

    print(len(data))
    data = pd.DataFrame(data, columns=['station', 'name',
                                       'audio', 'mark_in_s',
                                       'label_vec', 'durations', 'detection'])
    # print('safe file to "data.pkl"')
    # data.to_pickle('data.pkl')
    # data = pd.read_pickle('data.pkl')
    print('split data')
    train, dev, test = split(data)

    print('start feature extraction')
    train = transform(train)
    dev = transform(dev)
    test = transform(test)
    data = (train, dev, test)
    pickle.dump(data, open('data_split_mfcc_station.pkl', 'wb'))
