import hashlib
import json
from pathlib import Path
from random import choice

import pandas as pd
from numpy.random import uniform, randint

from hyper_search_unet.experiment import experiment


def gen_params():
    params = dict(
        # dropout_ratio=uniform(0.0, 0.5),
        loss_ratio=uniform(0.0, 1.0),
        num_filters=choice([
            [4, 8, 16],
            [8, 16, 32],
            [16, 32, 64],
            [32, 64, 128],
            [8, 16, 32, 64],
            [16, 32, 64, 128],
            [32, 64, 128, 256],
        ]),
        mixup_ratio=uniform(0.0, 0.4),
        log_compression=uniform(0.5, 10),
        freq_cutout=randint(1, 40),
        time_cutout=randint(1, 40),
        learning_rate=uniform(0.0001, 0.1),
        weight_decay=uniform(1e-5, 1e-2),
        subset_fraction=uniform(0.5, 1.0),
        random_state=randint(0, 200),
        bilinear=choice([True, False]),

        max_epoch=100,
        reduce_plateau_patience=10,
        early_stop_patience=20,
        model_name='TinyUnet',  # TFUNet

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
