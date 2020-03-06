import numpy as np
import torch.backends.cudnn
from torchaudio.transforms import MelSpectrogram

from data_augmentation import ToTensor, Compose, TorchUnsqueeze, ShortTermAverageTransform, ThresholdPoolSequence, \
    LogCompress
from data_extraction import LoadAudio, Frame, ResampleSpeedNormalization, extract_to_disk
from data_extraction import LoadTargets
from data_loading import fetch_dataloaders
from models.convolution_net import TinyCNN
from torch_callbacks import BinaryClassificationMetrics, SaveCheckpoint, Mixup, SchedulerWrap
from torch_learner import Learner

# reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(0)

# feature pre processing
sample_extraction_tfs = [
    LoadAudio(fs=48000),
    # Resample(target_fs=8192),
    ResampleSpeedNormalization(fs=48000, target_fs=8192, target_speed=50),
    Frame(frame_length=16384, hop_length=2048),
]
target_extraction_tfs = [
    LoadTargets(fs=48000),
    # Resample(target_fs=8192),
    ResampleSpeedNormalization(fs=48000, target_fs=8192, target_speed=50),
    Frame(frame_length=16384, hop_length=2048),
]
extract_to_disk(sample_extraction_tfs, target_extraction_tfs)
# exit()

train_tfs = {
    'sample': Compose([ToTensor(),
                       MelSpectrogram(sample_rate=8192, n_fft=512, hop_length=128, n_mels=40),
                       LogCompress(ratio=1),
                       # AmplitudeToDB(stype='power', top_db=80),
                       # TimeMasking(4),
                       # FrequencyMasking(4),
                       TorchUnsqueeze()
                       ]),
    'target': Compose([ShortTermAverageTransform(frame_length=512, hop_length=128, threshold=0.5),
                       ThresholdPoolSequence(0.125),
                       ToTensor()
                       ])
}
validation_tfs = {
    'sample': Compose([ToTensor(),
                       MelSpectrogram(sample_rate=8192, n_fft=512, hop_length=128, n_mels=40),
                       LogCompress(ratio=1),
                       # AmplitudeToDB(stype='power', top_db=80),
                       TorchUnsqueeze(),
                       ]),
    'target': Compose([ShortTermAverageTransform(frame_length=512, hop_length=128, threshold=0.5),
                       ThresholdPoolSequence(0.125),
                       ToTensor()
                       ]),
}

train_dl, validation_dl, test_dl, normalizer = fetch_dataloaders(
    batch_size=64,
    num_workers=1,
    memmap=True,
    train_tfs=train_tfs,
    validation_tfs=validation_tfs,
    normalize=True
)


class SmoothSegmentationLoss:
    def __init__(self):
        self.criterion = torch.nn.BCELoss()

    def __call__(self, outs, targets):
        outs = torch.nn.functional.softmax(outs, dim=1)[:, 1, :]
        targets = targets.float()
        return self.criterion(outs, targets)


class PooledSegmentationLoss:
    def __init__(self, llambda=0.5):
        self.criterion = torch.nn.BCELoss()
        self.llambda = llambda

    def __call__(self, outs, targets):
        outs = torch.nn.functional.softmax(outs, dim=1)[:, 1, :]
        targets = targets.float()
        segmentation_loss = self.criterion(outs, targets)

        classification_loss = self.criterion(
            (outs.mean(dim=-1) > 0.125).float(),
            (targets.mean(dim=-1) > 0.125).float()
        )

        loss = self.llambda * segmentation_loss + (1 - self.llambda) * classification_loss
        return loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('init model')
# model = Unet(channels=1, classes=2, bilinear=False)
model = TinyCNN()
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCELoss()
# criterion = PooledSegmentationLoss(llambda=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)

callbacks = [
    # SegmentationMetrics(),
    BinaryClassificationMetrics(),
    SchedulerWrap(scheduler),
    SaveCheckpoint('ckpt_local.pt'),
    Mixup(alpha=0.4),
]

learner = Learner(model, criterion, optimizer, callbacks=callbacks)
# learner.resume('ckpt_speedresamp.pt')
learner.fit(train_dl, validation_dl, max_epoch=100)
learner.resume('ckpt_speedresamp.pt')

# for threshold in [0.5, 0.3, 0.2, 0.1]:
#     learner.cb = CallbackHandler([BinaryClassificationMetrics(threshold=threshold)])
# learner.validate(test_dl)

"""
export model with
    * extraction tfs
    * loading tfs
    * model
"""

torch.save({'model': TinyCNN, 'model_parameters': model.state_dict(), 'transforms': validation_tfs}, 'model.pt')
