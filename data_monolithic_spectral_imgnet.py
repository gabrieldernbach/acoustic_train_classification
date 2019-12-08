import multiprocessing as mp
import os
import xml.etree.ElementTree as ElementTree
from typing import List, Tuple

import cv2
import librosa
import numpy as np
import pandas as pd
from scipy.signal import spectrogram


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


def resized_spectrogram(x):
    def single_call(x):
        # 1024 is 20ms frame with 1/4 overlap, corresponds to 4x oversampling in time domain
        x = spectrogram(x, nperseg=1024, noverlap=768)[2]
        # x = np.abs(stft(x, nperseg=1024, noverlap=786)[2])**2
        x = cv2.resize(x, (300, 300))  # alternatively skimage.transform.resize (brighter but worse contrast?)
        return x

    return np.apply_along_axis(single_call, 1, x)
    # return [single_call(x[i]) for i in range(len(x))]


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
    audio = librosa.core.load(data_path + '/' + name, 48000, mono=False)[0][0, :]
    audio = np.ascontiguousarray(audio)
    audio_len = len(audio)
    audio = librosa.util.frame(audio, frame_length=96000, hop_length=24000).T
    audio = resized_spectrogram(audio)

    # extract labels
    xml_label = r'{http://audacity.sourceforge.net/xml/}label'
    marks_in_s: List[Tuple[str, str]] = []
    for element in root.iter(xml_label):
        start = element.attrib['t']
        end = element.attrib['t1']
        marks_in_s.append((start, end))
    label_vec, durations, detection = mark_to_vec(marks_in_s, audio_len)
    label_vec = librosa.util.frame(label_vec, frame_length=96000, hop_length=24000)
    labels = label_vec.sum(axis=1) / 96000

    return station, name, audio, labels, detection


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
                                       'audio', 'labels', 'detection'])
    print('safe file to "data.pkl"')
    data.to_pickle('data_image_classify.pkl')
    print('finished')
