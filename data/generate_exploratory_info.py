"""
Generate exploratory information about the provided dataset,
such as how many files, their play time, and the extent to which we can find labels.
"""

import os

from librosa.core import get_duration

from data.data_build_register import extract_aup

# collect all wav files
cwd = os.getcwd()
stations = filter(os.path.isdir, os.listdir(cwd))
stations = [f for f in stations if not f.startswith('.')]

audio_paths = []
for station in stations:
    data_path = f'{cwd}/{station}'
    files = os.listdir(data_path)
    wav_names = [f for f in files if f.endswith('.wav')]
    wav_paths = [f'{data_path}/{f}' for f in wav_names]
    audio_paths.extend(wav_paths)

print(f'we found {len(audio_paths)} audio files in total')
lens = [get_duration(filename=f) for f in audio_paths]
print(f'their total play time amounts to {sum(lens) / 60 / 60:.2f} hours')

# collect all aup files
labeled_data = []
for station in stations:
    data_path = f'{cwd}/{station}'
    files = os.listdir(data_path)
    audacity_projects = [f for f in files if f.endswith('.aup')]
    project_paths = [f'{data_path}/{aup}' for aup in audacity_projects]
    station_data = [extract_aup(i, data_path, station, verbose=0) for i in project_paths]
    labeled_data.extend(station_data)

print(f'{len(labeled_data)} instances have labels provided')

# infer amount of detections
labeled_lens = [get_duration(filename=f[1]) for f in labeled_data]
print(f"their total play time amounts to {sum(labeled_lens) / 60 / 60}")
number_detections = sum([f[3] for f in labeled_data])
print(f"{number_detections / len(labeled_data):.2f} % of labeled data show at least one flat spot")
