import hashlib
import json
from pathlib import Path
from random import choice

import pandas as pd
from numpy.random import uniform

from melcnn.experiment import experiment

datasets = {
    'trainspeed_2sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/trainspeed_2sec',
    'trainpseed_5sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/trainspeed_5sec',
    'subsample_2sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/subsample_2sec',
    'subsample_5sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/subsample_5sec',
    'beatfrequency_2sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/beatfrequency_2sec',
    'beatfrequency_5sec': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/beatfrequency_5sec',
}


def gen_params():
    params = {
        'dataset': choice(list(datasets.keys())),
        'subset_fraction': 1.0,
        'random_state': choice([0, 1, 2]),
        'learning_rate': uniform(0.0001, 0.01),
        'weight_decay': uniform(1e-7, 1e-3),
        'mixup_ratio': uniform(0.01, 0.4),
        'reduce_plateau_patience': 10,
        'early_stop_patience': 20,
        'max_epoch': 200,
        'dropout_ratio': uniform(0.0, 0.5),
    }
    params['data_path'] = datasets[params['dataset']]

    print(json.dumps({**params}, indent=4))
    return params


for _ in range(1000):
    params = gen_params()
    uid = hashlib.md5(json.dumps({**params}).encode()).hexdigest()
    params['uid'] = uid
    fpath = Path(f'experiment_runs/{uid}.csv')

    res = experiment(**params)
    res = [{**r, **params} for r in res]
    pd.DataFrame(res).to_csv(fpath)
