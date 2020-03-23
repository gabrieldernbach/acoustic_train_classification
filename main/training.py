"""
Specify Model Training Run
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch.backends.cudnn
from torchaudio import transforms

from main.augment import ToTensor, Compose, TorchUnsqueeze, LogCompress, ShortTermAverageTransform
from main.callback import SaveCheckpoint, Mixup, SchedulerWrap, EarlyStopping
from main.extract import Frame, Resample
from main.learner import Learner
from main.load import fetch_dataloaders, build_register, train_dev_test

# environment
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)

train_tfs = {
    'audio': Compose([
        ToTensor(),
        transforms.MelSpectrogram(sample_rate=8192, n_fft=512, hop_length=128, n_mels=40),
        LogCompress(ratio=1),
        transforms.TimeMasking(4),
        transforms.FrequencyMasking(4),
        TorchUnsqueeze()
    ]),
    'target': Compose([
        ShortTermAverageTransform(frame_length=512, hop_length=128, threshold=0.5),  # use for segmentation
        # ThresholdPoolSequence(0.001),  # use for classification
        ToTensor()
    ])
}
dev_tfs = {
    'audio': Compose([
        ToTensor(),
        transforms.MelSpectrogram(sample_rate=8192, n_fft=512, hop_length=128, n_mels=40),
        LogCompress(ratio=1),
        TorchUnsqueeze(),
    ]),
    'target': Compose([
        ShortTermAverageTransform(frame_length=512, hop_length=128, threshold=0.5),  # use for segmentation
        # ThresholdPoolSequence(0.001),  # use for classification
        ToTensor()
    ]),
}

from main.models.conv_net import TinyCNN
from main.models.temporal_timbre_net import TinyTemporalTimbreCNN
from main.models.sample_net import TinySampleCNN
from main.models.concat_net import TinyConcatCNN
from main.models.unet import TinyUnet

model_catalogue = {
    'TinyCNN': TinyCNN,
    'TinyTemporalTimbreCNN': TinyTemporalTimbreCNN,
    'TinySampleCNN': TinySampleCNN,
    'TinyConcatCNN': TinyConcatCNN,
    'TinyUnet': TinyUnet
}


def main(*, model_name, mixup, subset_fraction, max_epoch, early_stop_patience, learning_rate, **kwargs):
    register = build_register(Path.cwd().parent / 'data_resample_train')  # _5s
    registers = train_dev_test(register, subset_fraction=subset_fraction)

    dl_args = {'batch_size': 64, 'num_workers': 4, 'pin_memory': True}
    dl = fetch_dataloaders(registers, dl_args, train_tfs=train_tfs, dev_tfs=dev_tfs,
                           slide_threshold=0.001)

    print('init model')
    model = model_catalogue[model_name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)

    ckpt_path = 'ckpt_local.pt'
    callbacks = [
        Mixup(alpha=mixup),
        SchedulerWrap(scheduler),
        EarlyStopping(patience=early_stop_patience),
        SaveCheckpoint(ckpt_path),
        model.metric(),
    ]

    learner = Learner(model, optimizer, callbacks=callbacks)
    learner.fit(dl['train'], dl['dev'], max_epoch=max_epoch)
    learner.resume(ckpt_path)

    print('collecting results')
    result = []
    for key in dl.keys():
        r = learner.validate(dl[key])
        r['phase'] = key
        result.append(r)

    # safe model to disk
    torch.save({'model_parameters': model.state_dict(), 'transforms': dev_tfs}, 'model.pt')

    return result


if __name__ == "__main__":

    def dict2hash(config):
        cstring = json.dumps(config)
        uid = hashlib.md5(cstring.encode()).hexdigest()
        return uid


    extract_config = {
        'target_fs': 8192,
        'frame_length': 16384,
        'hop_length': 8192,
        'resample_mode': 'train_speed',
    }
    resampler = Resample(**extract_config)
    framer = Frame(**extract_config)

    for mixup in [0.001, 0.2, 0.4]:
        for cv_iter in range(5):
            for model_name in ['TinyCNN', 'TinyTemporalTimbreCNN', 'TinyConcatCNN']:
                learn_config = {'mixup': mixup,
                                'model_name': model_name,
                                'subset_fraction': 0.2,
                                'max_epoch': 100,
                                'early_stop_patience': 20,
                                'learning_rate': 0.01,
                                'cv_iter': cv_iter}

                uid = dict2hash({**extract_config, **learn_config})
                fpath = Path(f'experiment_runs/{uid}.csv')
                if fpath.exists():
                    print('experiment results found, skipping run')
                    continue

                res = main(**learn_config)
                config = {**extract_config, **learn_config}
                res = [{**r, **config} for r in res]
                pd.DataFrame(res).to_csv(fpath)
