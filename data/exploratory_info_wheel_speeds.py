import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

root = pathlib.Path.cwd()
paths = list(root.rglob('*.aup'))


def beat_frequency(row):
    velocity = row.speedInMeterPerSeconds
    diameter = row.DiameterInMM / 1_000
    if np.any(diameter == 0):
        return np.NaN

    freq = velocity / (np.pi * diameter)
    return freq


def read_csv(p):
    df = pd.read_csv(p.with_suffix('.csv'), sep=';', decimal=',', dtype=np.float32)
    df['path'] = p
    df['file'] = p.with_suffix('').name
    df['station'] = p.parent.name
    return df


dfs = [read_csv(p) for p in tqdm(paths)]
df = pd.concat(dfs, axis=0, ignore_index=True)
df['beat_freq'] = df.apply(lambda row: beat_frequency(row), axis=1)
df['beat_freq'] = df['beat_freq'].fillna(method='ffill')

# sns.distplot(df.speedInMeterPerSeconds.values)
# plt.title('speeds')
# plt.xlabel('speed in $s$')
# plt.show()
#
# sns.distplot(df.DiameterInMM.values / 1_000)
# plt.xlabel('diameter of wheel in $m$')
# plt.show()
#
# sns.distplot(df.beat_freq.fillna(method='ffill').values)
# plt.xlabel('freq in $Hz$')
# plt.show()


brl = df.loc[df['station'] == 'BRL']
vld = df.loc[df['station'] == 'VLD']
bhv = df.loc[df['station'] == 'BHV']

sns.distplot(brl.speedInMeterPerSeconds, label='brl')
sns.distplot(vld.speedInMeterPerSeconds, label='vld')
sns.distplot(bhv.speedInMeterPerSeconds, label='bhv')
plt.legend()
plt.show()

sns.distplot(brl.DiameterInMM, label='brl')
sns.distplot(vld.DiameterInMM, label='vld')
sns.distplot(bhv.DiameterInMM, label='bhv')
plt.legend()
plt.show()

sns.distplot(brl.beat_freq, label='brl')
sns.distplot(vld.beat_freq, label='vld')
sns.distplot(bhv.beat_freq, label='bhv')
plt.legend()
plt.show()
