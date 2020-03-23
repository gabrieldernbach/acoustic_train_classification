from pathlib import Path

import pandas as pd

# pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)

files = list(Path.cwd().glob('experiment_runs/*.csv'))

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True, sort=True)

df = df[df.phase == 'test']

print(df.sort_values(by='f1pos', ascending=False))

import matplotlib.pyplot as plt
import seaborn as sns

params = [
    'bilinear', 'freq_cutout', 'learning_rate', 'log_compression',
    'loss_ratio', 'mixup_ratio', 'num_filters',
    'subset_fraction', 'time_cutout', 'weight_decay', 'f1pos'
]

df = df[params]
# sns.catplot(x='val', y='f1pos', col='col', data=df, col_wrap=4, sharex=False)

# dfa = df.melt('f1pos', var_name='col', value_name='val')
# chart = sns.catplot(x='val', y='f1pos', col='col', data=dfa, col_wrap=4, sharex=False)
# g = sns.FacetGrid(dfa, col='col', sharex=False, sharey=True, col_wrap=4)
# g.map(sns.stripplot, 'val', 'f1pos', dfa)

plt.show()

# g = sns.FacetGrid(dfa, col='col')
# g.map(sns.scatter(x='f1pos', y='val', data=dfa))


chart = sns.catplot(y='f1pos', x='bilinear', kind='box', data=df)
chart.set(ylim=(0.6, None))
plt.show()

chart = sns.catplot(y='f1pos', x='num_filters', kind='box', data=df)
chart.set_xticklabels(rotation=65, horizontalalignment='right')
chart.set(ylim=(0.6, None))
plt.show()

seg = ['subset_fraction', 'freq_cutout', 'time_cutout',
       'log_compression', 'learning_rate', 'weight_decay',
       'loss_ratio', 'mixup_ratio']
for s in seg:
    chart = sns.regplot(y='f1pos', x=s, data=df, lowess=True)
    chart.set(ylim=(0.6, None))
    plt.show()

# for key in params:
#     chart = sns.catplot(y='f1pos', x=key, data=df)
#     chart.set_xticklabels(rotation=65, horizontalalignment='right')
#     plt.show()
# print('end')
