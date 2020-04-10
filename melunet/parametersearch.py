import hashlib
import json
from pathlib import Path
from random import choice

import pandas as pd
from numpy.random import uniform

from melunet.experiment import experiment

# datasets = {
#     'trainspeed_2sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/trainspeed_2sec',
#     'trainpseed_5sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/trainspeed_5sec',
#     'subsample_2sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/subsample_2sec',
#     'subsample_5sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/subsample_5sec',
#     'beatfrequency_2sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/beatfrequency_2sec',
#     'beatfrequency_5sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/beatfrequency_5sec',
# }
#
#
datasets = {
    'trainspeed_2sec': '/home/gdernbach/projects/acoustic_train_class_data/data_processed/trainspeed_2sec',
    'trainpseed_5sec': '/home/gdernbach/projects/acoustic_train_class_data/data_processed/trainspeed_5sec',
    'subsample_2sec': '/home/gdernbach/projects/acoustic_train_class_data/data_processed/subsample_2sec',
    'subsample_5sec': '/home/gdernbach/projects/acoustic_train_class_data/data_processed/subsample_5sec',
    'beatfrequency_2sec': '/home/gdernbach/projects/acoustic_train_class_data/data_processed/beatfrequency_2sec',
    'beatfrequency_5sec': '/home/gdernbach/projects/acoustic_train_class_data/data_processed/beatfrequency_5sec',
}


def gen_params():
    params = {
        'dataset': 'beatfrequency_5sec',  # choice(list(datasets.keys())),
        'subset_fraction': 1.0,
        'random_state': 2,  # choice([0, 1, 2]),
        'learning_rate': 0.0028,  # uniform(0.0001, 0.01),
        'weight_decay': 0.000469,  # uniform(1e-7, 1e-3),
        'mixup_ratio': 0.20,  # uniform(0.01, 0.4),
        'reduce_plateau_patience': 10,
        'early_stop_patience': 20,
        'max_epoch': 200,
        'dropout_ratio': 0.2,  # uniform(0.0, 0.5),

        'loss_ratio': uniform(0.1, 0.9),
        'num_filters': choice([
            [16, 32, 64],
        ])
    }
    params['data_path'] = datasets[params['dataset']]

    print(json.dumps({**params}, indent=4))
    return params


# random search -- fully parallel
# for _ in range(1000):
#     params = gen_params()
#     uid = hashlib.md5(json.dumps({**params}).encode()).hexdigest()
#     params['uid'] = uid
#     fpath = Path(f'experiment_runs/{uid}.csv')
#
#     res = experiment(**params)
#     res = [{**r, **params} for r in res]
#     pd.DataFrame(res).to_csv(fpath)


# three fold cross validation
for dataset, dataset_path in datasets.items():
    for random_state in [0, 1, 2]:
        params = gen_params()
        params['dataset'] = dataset
        params['data_path'] = dataset_path
        params['random_state'] = random_state

        uid = hashlib.md5(json.dumps({**params}).encode()).hexdigest()
        params['uid'] = uid
        fpath = Path(f'experiment_runs/{uid}.csv')

        res = experiment(**params)
        res = [{**r, **params} for r in res]
        pd.DataFrame(res).to_csv(fpath)
