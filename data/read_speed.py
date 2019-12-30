import os

import librosa
import numpy as np
import pandas as pd

cwd = os.getcwd()
stations = filter(os.path.isdir, os.listdir(cwd))
sr = 48000

stations = [f for f in stations if not f.startswith('.')]

file = pd.read_csv('./BHV/2019-09-12--10-30-57-2019-09-12--10-35-56.csv', sep=';', decimal=',', dtype=np.float32)
file['smpl_indices'] = np.diff(file.WaveTimestampInSeconds * sr)

librosa.core.load('./BHV/2019-09-12--10-30-57-2019-09-12--10-35-56.csv', sr=48000)
