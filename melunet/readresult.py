from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option('display.expand_frame_repr', False)

result_path = Path('experiment_runs/')
data = pd.concat([pd.read_csv(p) for p in result_path.glob('*.csv')])
data['dataset'] = data['dataset'].replace('trainpseed_5sec', 'trainspeed_5sec')
data = data.sort_values('dataset')
print(data)

# sns.catplot(x='dataset', y='f1pos', hue='random_state', data=data[data['phase']=='dev'])
# plt.title('melunet - dev set f1')
# plt.xticks(rotation=30, horizontalalignment='right', fontweight='light')
# plt.show()

# sns.catplot(x='dataset', y='f1pos', hue='random_state', data=data[data['phase'] == 'test'])
sns.boxplot(x='dataset', y='f1pos', data=data[data['phase'] == 'test'])
plt.title('melunet - test set f1')
plt.xticks(rotation=30, horizontalalignment='right', fontweight='light')
plt.show()

print(data[data['phase'] == 'test'].sort_values('f1pos').iloc[-1])
