from pathlib import Path

import pandas as pd

# pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)

files = list(Path.cwd().glob('*.csv'))

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True, sort=True)

df = df[df.phase == 'test']
df = df[df.max_epoch > 50]

print(df.sort_values(by='f1pos', ascending=False))

import matplotlib.pyplot as plt
import seaborn as sns

sns.catplot(x='model_name', y='aps', hue='resample_mode', col='mixup', kind='box', data=df)
sns.catplot(x='model_name', y='f1pos', hue='resample_mode', col='mixup', kind='box', data=df)
plt.show()
