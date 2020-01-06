# from fastai.callbacks import OneCycleScheduler
# from fastai.vision import *
#
# path = untar_data(URLs.MNIST_SAMPLE)
# data = ImageDataBunch.from_folder(path)
#
# model = simple_cnn((3, 16, 16, 2))
# learn = Learner(data, model)
#
# cb = OneCycleScheduler(learn, lr_max=0.01)
# learn.fit(1, callbacks=cb)
#
#

import os

from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.metrics import ConfusionMatrix, auc_roc_score
from torch.utils.data import DataLoader


from conv_models import VGGNet
from data_set_custom import class_imbalance_sampler, MelDataset
import torch.nn as nn

files = '/mel_train.npz', '/mel_validation.npz', '/mel_test.npz'
path = os.path.dirname(os.path.realpath(__file__))
train_path, validation_path, test_path = [path + s for s in files]

train_set = MelDataset(train_path)

sampler = class_imbalance_sampler(train_set.labels)
train_loader = DataLoader(train_set,
                          sampler=sampler,
                          batch_size=20,
                          num_workers=4,
                          pin_memory=True)
validation_loader = DataLoader(MelDataset(validation_path),
                               batch_size=20,
                               num_workers=4,
                               shuffle=True,
                               pin_memory=True)
test_loader = DataLoader(MelDataset(test_path),
                         batch_size=20,
                         num_workers=4,
                         shuffle=True,
                         pin_memory=True)

data = DataBunch(train_dl=train_loader,
                 valid_dl=validation_loader,
                 test_dl=test_loader)

model = VGGNet(1, [32, 64, 64, 64, 64, 64, 64], 1)

learn = Learner(data, model)
learn.loss_func = nn.BCEWithLogitsLoss()
learn.metrics = [auc_roc_score]

learn.fit(1)
