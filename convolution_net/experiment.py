import hashlib
import json
from pathlib import Path

import torch.backends.cudnn
from torchaudio import transforms

from augment import ToTensor, Compose, TorchUnsqueeze, LogCompress, ShortTermAverageTransform
from callback import SaveCheckpoint, Mixup, SchedulerWrap, EarlyStopping
from extract import Frame, Resample
from learner import Learner
from load import fetch_dataloaders, build_register, train_dev_test

# environment
# torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)
# np.random.seed(0)

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
        ShortTermAverageTransform(frame_length=512, hop_length=128, threshold=0.5),
        # ThresholdPoolSequence(0.001),  # was 0.125
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
        ShortTermAverageTransform(frame_length=512, hop_length=128, threshold=0.5),
        # ThresholdPoolSequence(0.001),
        ToTensor()
    ]),
}

from models.conv_net import TinyCNN
from models.temporal_timbre_net import TinyTemporalTimbreCNN
from models.sample_net import TinySampleCNN
from models.concat_net import TinyConcatCNN
from models.unet import TinyUnet

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
                           segmentation_threshold=0.001, load_in_memory=False)

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

    result = []
    for key in dl.keys():
        r = learner.validate(dl[key])
        r['phase'] = key
        result.append(r)

    # safe model to disk
    torch.save({'model_parameters': model.state_dict(), 'transforms': dev_tfs}, 'model.pt')

    return result


"""
export model with
    * extraction tfs
    * loading tfs
    * model
"""

"""
model = ['TinyeNet', 'TfNet', 'ConcatNet']
mixup = [0.0, 0.2, 0.4]
subset_ratio = [0.2, 0.5, 1.0]
resample_speed = ['sub', 'speed', 'freq']
window_length = [1, 2, 5] # in seconds
"""

"""
Cross Validation
"""

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
    # create_dataset(resampler, framer)

    # for mixup in [0.001, 0.2, 0.4]:
    for mixup in [0.2]:
        for cv_iter in range(5):
            # for model_name in ['TinyCNN', 'TinyTemporalTimbreCNN', 'TinyConcatCNN']:
            for model_name in ['TinyUnet']:
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
                # pd.DataFrame(res).to_csv(fpath)
