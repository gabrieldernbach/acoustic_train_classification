import os

import torch.backends.cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from convolution_net.augment import ToTensor, Compose, TorchUnsqueeze
from convolution_net.callback import SaveCheckpoint, Mixup, SchedulerWrap, EarlyStopping
from convolution_net.learner import Learner
from convolution_net.load import fetch_dataloaders, build_register, train_dev_test

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_num_threads(2)


def construct_transforms(**kwargs):
    train_tfs = {
        'audio': Compose([
            # PitchShift(**kwargs),
            ToTensor(),
            TorchUnsqueeze()
        ]),
        'target': Compose([
            ToTensor()
        ])
    }
    dev_tfs = {
        'audio': Compose([
            ToTensor(),
            TorchUnsqueeze(),
        ]),
        'target': Compose([
            ToTensor()
        ]),
    }
    return train_tfs, dev_tfs


from convolution_net.models.conv_net import TinyCNN
from convolution_net.models.temporal_timbre_net import TinyTemporalTimbreCNN
from convolution_net.models.sample_net import TinySampleCNN
from convolution_net.models.concat_net import TinyConcatCNN
from convolution_net.models.unet import TinyUnet
from convolution_net.models.tf_unet import TFUNet
from convolution_net.models.waveunet import WaveUnet

model_catalogue = {
    'TinyCNN': TinyCNN,
    'TinyTemporalTimbreCNN': TinyTemporalTimbreCNN,
    'TinySampleCNN': TinySampleCNN,
    'TinyConcatCNN': TinyConcatCNN,
    'TinyUnet': TinyUnet,
    'TFUNet': TFUNet,
    'WaveUnet': WaveUnet
}


def experiment(**kwargs):
    register = build_register(kwargs['data_path'])  # _5s
    registers = train_dev_test(register, subset_fraction=kwargs['subset_fraction'])

    train_tfs, dev_tfs = construct_transforms(**kwargs)
    dl_args = {'batch_size': 512, 'num_workers': 4, 'pin_memory': True}
    dl = fetch_dataloaders(registers, dl_args, train_tfs=train_tfs, dev_tfs=dev_tfs,
                           slide_threshold=0.05, load_in_memory=kwargs['data_set_in_memory'])

    print('init model')
    model = model_catalogue[kwargs['model_name']](**kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs['learning_rate'], weight_decay=kwargs['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=kwargs['reduce_plateau_patience'],
                                  verbose=True)

    ckpt_path = f'experiment_runs/{kwargs["uid"]}.pt'
    callbacks = [
        Mixup(alpha=kwargs['mixup_ratio']),
        SchedulerWrap(scheduler),
        EarlyStopping(patience=kwargs['early_stop_patience']),
        SaveCheckpoint(ckpt_path),
        model.metric(),
    ]

    learner = Learner(model, optimizer, callbacks=callbacks)
    learner.fit(dl['train'], dl['dev'], max_epoch=kwargs['max_epoch'])
    learner.resume(ckpt_path)

    result = []
    for key in dl.keys():
        r = learner.validate(dl[key])
        r['phase'] = key
        result.append(r)
    os.remove(ckpt_path)

    # safe model to disk
    # torch.save({'model_parameters': model.state_dict(), 'transforms': dev_tfs}, 'model.pt')

    return result
