from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option('display.expand_frame_repr', False)

# result_path = Path('experiment_runs/')
# result_path = Path('experiment_runs_old/')
result_path = Path('experiment_runs/')
data = pd.concat([pd.read_csv(p) for p in result_path.glob('*.csv')])

data['dataset'] = data['dataset'].replace('trainpseed_5sec', 'trainspeed_5sec')
data = data.sort_values('dataset')
data.dataset = pd.Categorical(data.dataset, [
    "subsample_2sec",
    "trainspeed_2sec",
    "beatfrequency_2sec",
    "subsample_5sec",
    "trainspeed_5sec",
    "beatfrequency_5sec"
])

sns.boxplot(x='dataset', y='f1pos', data=data[data['phase'] == 'dev'])
plt.xticks(rotation=30, horizontalalignment='right', fontweight='light')
plt.show()
# print(data)

sns.boxplot(x='dataset', y='f1pos', data=data[data['phase'] == 'test'])
plt.xticks(rotation=30, horizontalalignment='right', fontweight='light')
plt.show()

print(data[data.phase == 'test'].sort_values('f1pos'))
top = data[data.phase == 'test']
top = top[top.dataset == 'beatfrequency_5sec'].sort_values('f1pos')
print(top)
