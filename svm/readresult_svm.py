from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option('display.expand_frame_repr', False)

result_path = Path('/Users/gabrieldernbach/git/acoustic_train_class_data/experiment_runs/svm/')
data = pd.concat([pd.read_csv(p) for p in result_path.glob('*best*.csv')])

# sort and fix typo
data['dataset'] = data['dataset'].replace('trainpseed_5sec', 'trainspeed_5sec')
data = data.sort_values('dataset')
print(data)

data.dataset = pd.Categorical(data.dataset, [
    "subsample_2sec",
    "trainspeed_2sec",
    "beatfrequency_2sec",
    "subsample_5sec",
    "trainspeed_5sec",
    "beatfrequency_5sec"
])

chart = sns.boxplot(x='dataset', y='f1', data=data)
plt.xticks(rotation=30, horizontalalignment='right', fontweight='light')
chart.set_ylim(0.3, 0.85)

plt.title('svm with c=1, gamma=0.001')
plt.show()

# result_path = Path('/Users/gabrieldernbach/git/acoustic_train_class_data/experiment_runs/svm/')
# data = pd.concat([pd.read_csv(p) for p in result_path.glob('*.csv')])
# print(data)
#
# chart = sns.swarmplot(x='dataset_name', y='mean_test_score', data=data)
# plt.xticks(rotation=30, horizontalalignment='right', fontweight='light')
# plt.show(
#
# data['param_svc__class_weight'] = data['param_svc__class_weight'].astype('category').cat.codes
#
# chart = sns.swarmplot(x='param_quantile', y='mean_test_score', hue='dataset_name', data=data)
# plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize='xx-small')
# plt.show()
#
# import numpy as np
# chart = sns.regplot(x='param_svc__C', y='mean_test_score', x_bins=20, data=data)
# plt.show()
# chart = sns.regplot(x='param_svc__gamma', y='mean_test_score', x_bins=20, data=data)
# chart.set(xscale='log')
# plt.show()
#
# print('results')
# print('finish')
