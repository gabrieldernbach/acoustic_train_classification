from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option('display.expand_frame_repr', False)

result_path = Path('experiment_runs/')
data = pd.concat([pd.read_csv(p) for p in result_path.glob('*.csv')])

sns.catplot(x='dataset', y='f1pos', data=data[data['phase'] == 'dev'])
plt.xticks(rotation=30, horizontalalignment='right', fontweight='light')
plt.show()
# print(data)

sns.catplot(x='dataset', y='f1pos', data=data[data['phase'] == 'test'])
plt.xticks(rotation=30, horizontalalignment='right', fontweight='light')
plt.show()

print(data[data.phase == 'test'].sort_values('f1pos'))
