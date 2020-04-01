"""
Specify Model Training Run
"""

import torch.backends.cudnn
from torchaudio import transforms

from main.augment import ToTensor, Compose, TorchUnsqueeze, LogCompress, ShortTermAverageTransform
from main.callback import SaveCheckpoint, Mixup, SchedulerWrap, EarlyStopping
from main.extract import ResampleTrainSpeed, Frame
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
        LogCompress(ratio=7),
        transforms.TimeMasking(5),
        transforms.FrequencyMasking(16),
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
from main.models.multiresunet import MultiResUnet
from main.models.unet_temporal_timbre import TTUNet

model_catalogue = {
    'TinyCNN': TinyCNN(),
    'TinyTemporalTimbreCNN': TinyTemporalTimbreCNN(),
    'TinySampleCNN': TinySampleCNN(),
    'TinyConcatCNN': TinyConcatCNN(),
    'TinyUnet': TinyUnet([8, 16, 32]),
    'MultiResUnet': MultiResUnet([16, 32, 64]),
    'TTUNet': TTUNet([8, 16, 32])
}


def main(**cfg):
    register = build_register(cfg['data_path'])
    registers = train_dev_test(register, subset_fraction=cfg['subset_fraction'])

    dl_args = {'batch_size': 64, 'num_workers': 4, 'pin_memory': True}
    dl = fetch_dataloaders(registers, dl_args, train_tfs=train_tfs, dev_tfs=dev_tfs,
                           slide_threshold=0.001)

    print('init model')
    model = model_catalogue[cfg['model_name']]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=0.004)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)

    ckpt_path = 'model_ckpt.pt'
    callbacks = [
        Mixup(alpha=cfg['mixup']),
        SchedulerWrap(scheduler),
        EarlyStopping(patience=cfg['early_stop_patience']),
        SaveCheckpoint(ckpt_path),
        model.metric(),
    ]

    learner = Learner(model, optimizer, callbacks=callbacks)
    learner.fit(dl['train'], dl['dev'], max_epoch=cfg['max_epoch'])
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

    """
    Build dataset
    """

    source = '/Users/gabrieldernbach/git/acoustic_train_class_data/data'
    destination = '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/data_resample_train_10s'
    fs = 8192

    resampler = ResampleTrainSpeed(fs, target_train_speed=14)
    framer = Frame(frame_length=5 * fs, hop_length=5 * fs)
    # create_dataset(source, destination, framer=framer, resampler=resampler)

    """
    Configure and train model
    """


    learn_config = {
        'data_path': '/Users/gabrieldernbach/git/acoustic_train_class_data/data_processed/data_resample_train_10s',
        'mixup': 0.2,
        'model_name': 'TTUNet',
        'subset_fraction': 0.8,
        'max_epoch': 100,
        'early_stop_patience': 20,
        'learning_rate': 0.0001
    }

    res = main(**learn_config)
    print(res)
