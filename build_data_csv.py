import os
import xml.etree.ElementTree as ElementTree

import pandas as pd


def extract_aup(aup_path):
    # parse xml
    doc = ElementTree.parse(aup_path)
    root = doc.getroot()

    # load wavfile
    xml_wave = r'{http://audacity.sourceforge.net/xml/}wavetrack'
    name = root.find(xml_wave).attrib['name'] + '.wav'
    print(f'extracting data point {name}')
    audio_path = f'{data_path}/{name}'

    # extract labels
    xml_label = r'{http://audacity.sourceforge.net/xml/}label'
    marks = []
    for element in root.iter(xml_label):
        start = element.attrib['t']
        end = element.attrib['t1']
        marks.append((start, end))
    detection = (len(marks) > 0)

    return station, audio_path, marks, detection


if __name__ == '__main__':
    cwd = os.getcwd()
    stations = os.listdir(cwd + '/data')
    stations = [s for s in stations if not s.startswith('.')]
    print(f'found sub folders for station {stations}')

    data = []
    for station in stations:
        print(f'loading station {station}')
        data_path = f'{cwd}/data/{station}'
        files = os.listdir(data_path)
        audacity_projects = [f for f in files if f.endswith('.aup')]
        project_paths = [os.path.join(data_path, aup) for aup in audacity_projects]
        print(f'found {len(project_paths)} files')

        station_data = [extract_aup(i) for i in project_paths]

        data.extend(station_data)

    print(len(data))
    data = pd.DataFrame(data, columns=['station', 'audio_path',
                                       'label', 'detection'])
    print('safe file to "data_register.pkl"')
    data.to_pickle('data_register.pkl')
    print('finished')
