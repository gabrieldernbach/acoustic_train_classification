import os

import torch
import torch.backends.cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from convolution_net.augment import ToTensor, Compose, TorchUnsqueeze
from convolution_net.callback import EarlyStopping, SaveCheckpoint, Mixup
from convolution_net.learner import Learner
from convolution_net.load import build_register, train_dev_test, fetch_dataloaders
from sampleunet.model import SampleUnet

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

train_tfs = {
    'audio': Compose([
        ToTensor(),
        TorchUnsqueeze()
    ]),
    'target': Compose([
        # ThresholdPoolSequence(0.05),  # was 0.125
        ToTensor()
    ])

}
dev_tfs = {
    'audio': Compose([
        ToTensor(),
        TorchUnsqueeze(),
    ]),
    'target': Compose([
        # ThresholdPoolSequence(0.05),  # was 0.125
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
    model = SampleUnet(num_filters=kwargs['num_filters'],
                       loss_ratio=kwargs['loss_ratio'],
                       dropout=kwargs['dropout_ratio'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs['learning_rate'], weight_decay=kwargs['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=kwargs['reduce_plateau_patience'],
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
