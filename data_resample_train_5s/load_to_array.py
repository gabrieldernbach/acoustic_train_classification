from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


def path2entry(path):
    entry = {
        'audio_path': path,  # numpy array of 5s at 8192 Fs
        'target_path': Path(str(path).replace('_audio.npy', '_target.npy')),  # numpy array of 5s at 8192 Fs
        'file_name': path.parent,  # file name used for grouped split
        'speed_bucket': path.parent.parent.name,  # on of 10 bins
        'station': path.parent.parent.parent.name,  # one of bhv, brl, vld
    }
    return entry


def build_register(root):
    print('indexing dataset')
    root = Path(root)
    source_paths = list(root.rglob('*audio.npy'))
    register = pd.DataFrame([path2entry(p) for p in tqdm(source_paths)])

    print('label encode')
    register['station_id'] = register.station.astype('category').cat.codes
    register['speed_id'] = register.speed_bucket.astype('category').cat.codes
    return register


def group_split(register, random_state=5, group='file_name'):
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=random_state)
    split = list(gss.split(register, groups=register[group].values))[0]
    return register.iloc[split[0]], register.iloc[split[1]]


def split_train_dev_test(register, subset_fraction=1.0, random_state=5):
    remain, test = group_split(register, random_state)
    remain = remain.sample(frac=subset_fraction)
    train, dev = group_split(remain, random_state)
    return {'train': train, 'dev': dev, 'test': test}


def load_to_np(register):
    audio = Parallel(n_jobs=4, verbose=10)([delayed(np.load)(p) for p in register.audio_path.values])
    target = Parallel(n_jobs=4, verbose=10)([delayed(np.load)(p) for p in register.target_path.values])
    return np.array(audio), np.array(target)


if __name__ == "__main__":
    root = Path.cwd()
    print(root)
    register = build_register(root)
    print(register.keys())

    # register = register['station'=='BHV' or 'station'=='BRL']
    # register = register['speed_id' < 5]

    registers = split_train_dev_test(register, subset_fraction=0.3, random_state=5)

    X, Y = load_to_np(registers['train'])
    print(X.shape, Y.shape)
