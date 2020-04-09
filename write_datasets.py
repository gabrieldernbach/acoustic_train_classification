# generate representation

# load data

# initialize model
sr = 8192
from pathlib import Path

from convolution_net.extract import ResampleTrainSpeed, Resample, ResampleBeatFrequency, Frame, create_dataset

source_path = Path('/Users/gabrieldernbach/git/acoustic_train_class_data/data')
destination_path = Path('/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/')

sr = 8192
resampler_catalogue = {
    'subsample': Resample(target_fs=sr),
    'trainspeed': ResampleTrainSpeed(target_fs=sr, target_train_speed=14),
    'beatfrequency': ResampleBeatFrequency(target_fs=sr, target_freq=8)
}

framer_catalogue = {
    # '2sec': Frame(2 * sr, 2 * sr),
    # '5sec': Frame(5 * sr, 5 * sr),
    '7sec': Frame(7 * sr, 7 * sr),
}

for sampler_name, resampler in resampler_catalogue.items():
    for framer_name, framer in framer_catalogue.items():
        create_dataset(source_path=source_path,
                       destination_path=destination_path / f'{sampler_name}_{framer_name}',
                       resampler=resampler,
                       framer=framer)
