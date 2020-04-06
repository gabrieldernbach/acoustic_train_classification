from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option('display.expand_frame_repr', False)

result_path = Path('/Users/gabrieldernbach/git/acoustic_train_class_data/experiment_runs/sgd/')
data = pd.concat([pd.read_csv(p) for p in result_path.glob('*exp*.csv')])
print(data)

chart = sns.swarmplot(x='dataset_name', y='mean_test_score', data=data)
plt.xticks(rotation=30, horizontalalignment='right', fontweight='light')
plt.show()

data['param_sgd__class_weight'] = data['param_sgd__class_weight'].astype('category').cat.codes

chart = sns.swarmplot(x='param_quantile', y='mean_test_score', hue='dataset_name', data=data)
plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize='xx-small')
plt.show()

chart = sns.swarmplot(x='param_sgd__loss', y='mean_test_score', hue='dataset_name', data=data)
plt.show()
chart = sns.scatterplot(x='param_sgd__max_iter', y='mean_test_score', hue='dataset_name', data=data)
plt.show()
chart = sns.swarmplot(x='param_sgd__class_weight', y='mean_test_score', hue='dataset_name', data=data)
plt.show()
chart = sns.scatterplot(x='param_sgd__alpha', y='mean_test_score', hue='dataset_name', data=data)
chart.set(xscale='log')
plt.show()

result_path = Path('/Users/gabrieldernbach/git/acoustic_train_class_data/experiment_runs/sgd/')
data = pd.concat([pd.read_csv(p) for p in result_path.glob('*test*.csv')])
sns.boxplot(y='f1', data=data)
print(data)

print('results')
print('finish')
