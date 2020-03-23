import hashlib
import json
from pathlib import Path
from random import choice

import pandas as pd
from numpy.random import uniform

from hyper_search_waveunet.experiment import experiment


def gen_params():
    params = dict(
        dropout_ratio=uniform(0.0, 0.5),
        loss_ratio=uniform(0.0, 1.0),
        num_filters=choice([[2, 4, 8, 16, 32, 64, 128, 256],
                            [2, 4, 8, 16, 32, 64, 128],
                            [2, 4, 8, 16, 32, 64],
                            [4, 8, 16, 32]]),
        mixup_ratio=uniform(0.0, 0.4),
        # pitch_shift_range_cent=randint(0, 1200),
        learning_rate=uniform(0.0001, 0.1),
        weight_decay=uniform(1e-5, 1e-1),
        subset_fraction=uniform(0.7, 1.0),

        max_epoch=100,
        reduce_plateau_patience=10,
        early_stop_patience=20,
        model_name='WaveUnet',  # TFUNet

        target_fs=8192,
        frame_length=8192 * 5,
        hop_length=8192 * 5,
        resample_mode='train_speed',

        data_path=str(Path.cwd().parent / 'data_resample_train_5s'),
        data_set_in_memory=False,
    )
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
