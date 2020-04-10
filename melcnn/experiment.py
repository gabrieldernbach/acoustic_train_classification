import os

import torch.backends.cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchaudio import transforms

from convolution_net.augment import ToTensor, Compose, TorchUnsqueeze, LogCompress, ThresholdPoolSequence
from convolution_net.callback import SaveCheckpoint, Mixup, EarlyStopping
from convolution_net.learner import Learner
from convolution_net.load import fetch_dataloaders, build_register, train_dev_test
from melcnn.model import MelCNN

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)

train_tfs = {
    'audio': Compose([
        ToTensor(),
        transforms.MelSpectrogram(sample_rate=8192, n_fft=512, hop_length=128, n_mels=40),
        LogCompress(ratio=7),
        transforms.TimeMasking(4),
        transforms.FrequencyMasking(4),
        TorchUnsqueeze()
    ]),
    'target': Compose([
        ThresholdPoolSequence(0.05),
        ToTensor()
    ])
}
dev_tfs = {
    'audio': Compose([
        ToTensor(),
        transforms.MelSpectrogram(sample_rate=8192, n_fft=512, hop_length=128, n_mels=40),
        LogCompress(ratio=7),
        TorchUnsqueeze(),
    ]),
    'target': Compose([
        ThresholdPoolSequence(0.05),
        ToTensor()
    ]),
}


def experiment(**kwargs):
    register = build_register(kwargs['data_path'])
    registers = train_dev_test(register,
                               subset_fraction=kwargs['subset_fraction'],
                               random_state=kwargs['random_state'])

    dl_args = {'batch_size': 64, 'num_workers': 4, 'pin_memory': True}
    dl = fetch_dataloaders(registers, dl_args, train_tfs=train_tfs, dev_tfs=dev_tfs, slide_threshold=0.05)

    print('init model')
    model = MelCNN(kwargs['dropout_ratio'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs['learning_rate'], weight_decay=kwargs['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=kwargs['reduce_plateau_patience'],
                                  verbose=True)

    ckpt_path = f'experiment_runs/{kwargs["uid"]}.pt'
    callbacks = [
        Mixup(alpha=kwargs['mixup_ratio']),
        EarlyStopping(patience=kwargs['early_stop_patience']),
        SaveCheckpoint(ckpt_path),
        model.metric(),
    ]

    learner = Learner(model, optimizer, scheduler, callbacks=callbacks)
    learner.fit(dl['train'], dl['dev'], max_epoch=kwargs['max_epoch'])
    learner.resume(ckpt_path)

    result = []
    for key in dl.keys():
        r = learner.validate(dl[key])
        r['phase'] = key
        result.append(r)
    os.remove(ckpt_path)

    return result
