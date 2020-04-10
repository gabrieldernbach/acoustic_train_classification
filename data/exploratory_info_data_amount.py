"""
Generate exploratory information about the provided dataset,
such as how many files, their play time, and the extent to which we can find labels.
"""

import xml.etree.ElementTree as ElementTree
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from librosa.core import get_duration
from tqdm import tqdm

sns.set_context('paper', font_scale=1.5)


def read_aup(aup_path):
    audio_path = aup_path.with_suffix('.wav')
    csv_path = aup_path.with_suffix('.csv')
    doc = ElementTree.parse(aup_path)
    root = doc.getroot()

    # load wave file
    xml_wave = r'{http://audacity.sourceforge.net/xml/}wavetrack'
    name = root.find(xml_wave).attrib['name']

    # extract targets
    xml_target = r'{http://audacity.sourceforge.net/xml/}label'
    marks = []
    for element in root.iter(xml_target):
        start = element.attrib['t']
        end = element.attrib['t1']
        marks.append((start, end))
    detection = (len(marks) > 0)

    # extract speed and diameter
    file = pd.read_csv(csv_path, sep=';', decimal=',', dtype=np.float32)

    speed = file.speedInMeterPerSeconds.mean()

    diameter = file.DiameterInMM / 1_000  # convert to meter
    diameter[diameter == 0] = np.NaN
    diameter.ffill(inplace=True)
    diameter = diameter.mean()

    entry = {
        'marks': marks,
        'speed': speed,
        'detection': detection,
        'diameter': diameter,
    }

    return entry


# collect all wav files
# cwd = os.getcwd()
data_path = Path('/Users/gabrieldernbach/git/acoustic_train_class_data/data')
project_paths = list(data_path.rglob('*.aup'))
print(f'found {len(project_paths)} train passings with annotation')

fig, axes = plt.subplots(3, 1, figsize=(7, 7))

durations = [get_duration(filename=str(p.with_suffix('.wav'))) for p in project_paths]
print(f'their joint play time amounts to {sum(durations) / 60 / 60:.2f}')
chart = sns.distplot(
    pd.Series(durations, name='distribution of drive by durations in minutes') / 60,
    rug=True,
    kde=True,
    hist=False,
    kde_kws={'shade': True},
    rug_kws={'alpha': 0.5, 'linewidth': 0.5, 'height': 0.05},
    ax=axes[0])
chart.set(ylabel='density')
chart.set(yticklabels=[])
# plt.savefig('drivebydurations.svg', format='svg', dpi=300)
# plt.savefig('drivebydurations.pdf', format='pdf', dpi=300)
# plt.show()

meta = pd.DataFrame([read_aup(p) for p in tqdm(project_paths)])
print('number of train passings with at least one mark', sum(meta.detection))
print('number of marked regions', sum([len(m) for m in meta.marks]))

mark_durs = [float(m[1]) - float(m[0]) for marks in meta.marks for m in marks]
print('total duration of annotated flat spot', sum(mark_durs) / 60)
chart = sns.distplot(pd.Series(
    mark_durs, name='distribution of flat spot durations in seconds'),
    rug=True,
    kde=True,
    hist=False,
    kde_kws={'shade': True},
    rug_kws={'alpha': 0.5, 'linewidth': 0.5, 'height': 0.05},
    ax=axes[1])
chart.set(ylabel='density')
chart.set(yticklabels=[])
# plt.savefig('flatspotdurations.svg', format='svg', dpi=300)
# plt.savefig('flatspotdurations.pdf', format='pdf', dpi=300)
# plt.show()

chart = sns.distplot(pd.Series(meta.speed, name='distribution of speed of passing trains in km/h') * 3.6,
                     rug=True,
                     kde=True,
                     hist=False,
                     kde_kws={'shade': True},
                     rug_kws={'alpha': 0.5, 'linewidth': 0.5, 'height': 0.05},
                     ax=axes[2])
chart.set(ylabel='density')
chart.set(yticklabels=[])
# plt.savefig('trainspeeds.svg', format='svg', dpi=300)
# plt.savefig('trainspeeds.pdf', format='pdf', dpi=300)
# plt.show()

plt.tight_layout()
plt.savefig('datastats.svg', format='svg', dpi=300)
plt.savefig('datastats.pdf', format='pdf', dpi=300)
plt.show()
