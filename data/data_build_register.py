"""
Creates a register that locates files in given subdirectories
and extracts the targets from provided aup projects.

By default data is assumed to live in:
    data/{station}/{*.aup}

"""

import os
import xml.etree.ElementTree as ElementTree

import numpy as np
import pandas as pd


def split(data, a=0.6, b=0.8):
    """
    Create a random train, dev, test split of a pandas data frame
    """
    a, b = int(a * len(data)), int(b * len(data))
    data_shuffled = data.sample(frac=1, random_state=1).reset_index(drop=True)
    train, validation, test = np.split(data_shuffled, [a, b])
    validation.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    return train, validation, test


def extract_aup(aup_path, data_path, station, verbose=1):
    # parse xml
    doc = ElementTree.parse(aup_path)
    root = doc.getroot()

    # load wave file
    xml_wave = r'{http://audacity.sourceforge.net/xml/}wavetrack'
    name = root.find(xml_wave).attrib['name']
    if verbose > 0:
        print(f'extracting data point {name}')
    audio_path = f'{data_path}/{name}.wav'

    # extract targets
    xml_target = r'{http://audacity.sourceforge.net/xml/}label'
    marks = []
    for element in root.iter(xml_target):
        start = element.attrib['t']
        end = element.attrib['t1']
        marks.append((start, end))
    detection = (len(marks) > 0)

    # extract speed
    path_csv = f'{data_path}/{name}.csv'
    file = pd.read_csv(path_csv, sep=';', decimal=',', dtype=np.float32)
    speeds = file.speedInMeterPerSeconds
    speed_kmh = (speeds * 3.6).mean()

    return station, audio_path, marks, detection, speed_kmh


if __name__ == '__main__':
    cwd = os.getcwd()
    stations = filter(os.path.isdir, os.listdir(cwd))
    stations = [f for f in stations if not f.startswith('.')]
    print(f'found sub folders for station {stations}')

    data = []
    for station in stations:
        print(f'loading station {station}')
        data_path = f'{cwd}/{station}'
        files = os.listdir(data_path)
        audacity_projects = [f for f in files if f.endswith('.aup')]
        project_paths = [os.path.join(data_path, aup) for aup in audacity_projects]
        print(f'found {len(project_paths)} files')

        station_data = [extract_aup(i, data_path, station) for i in project_paths]

        data.extend(station_data)

    print(len(data))
    data = pd.DataFrame(data, columns=['station', 'audio_path',
                                       'target', 'detection', 'speed_kmh'])
    print('safe file to "data_register.pkl"')
    data.to_pickle('data_register.pkl')

    train, dev, test = split(data)

    train.to_pickle('data_register_train.pkl')
    print(f'train data ratio of well / defective trains \n{train.detection.value_counts(normalize=True).values}')
    dev.to_pickle('data_register_dev.pkl')
    print(f'dev data ratio of well / defective trains \n{dev.detection.value_counts(normalize=True).values}')
    test.to_pickle('data_register_test.pkl')
    print(f'test data ratio of well / defective trains \n{test.detection.value_counts(normalize=True).values}')
