import torch.backends.cudnn

from data_augmentations import ToTensor, Compose
from data_feature_extraction import LoadAudio, ResampleSpeedNormalization, Frame, ShortTermMelTransform
from data_feature_extraction import LoadTargets, AvgPoolTargets
from data_feature_extraction import extract_to_disk
from data_loaders import fetch_dataloaders
from models.convolution_net import TinyCNN
from torch_callbacks import BinaryClassificationMetrics, Mixup, SchedulerWrap, SaveCheckpoint, CallbackHandler
from torch_learner import Learner

# feature pre processing

sample_extraction_tfs = [
    LoadAudio(fs=48000),
    ResampleSpeedNormalization(target_fs=8000, target_speed=50),
    Frame(frame_length=16000, hop_length=2000),
    ShortTermMelTransform(fs=8000, n_fft=512, hop_length=128, n_mels=40)]
target_extraction_tfs = [
    LoadTargets(fs=48000),
    ResampleSpeedNormalization(target_fs=8000, target_speed=50),
    Frame(frame_length=16000, hop_length=2000),
    AvgPoolTargets(threshold=0.125)]
extract_to_disk(sample_extraction_tfs, target_extraction_tfs, normalize=True)

#
# data loading
train_tfs = Compose([
    ToTensor(),
    # Scatter1D(),
    # MelSpectrogram(sample_rate=8192, n_fft=512, hop_length=128, n_mels=40),
    # AmplitudeToDB(stype='power', top_db=80),
    # TimeMasking(time_mask_param=20),
    # FrequencyMasking(freq_mask_param=4),
])
validation_tfs = Compose([
    ToTensor(),
    # Scatter1D()
    # MelSpectrogram(sample_rate=8192, n_fft=512, hop_length=128, n_mels=40),
    # AmplitudeToDB(stype='power', top_db=80)
])

train_dl, validation_dl, test_dl = fetch_dataloaders(
    batch_size=16,
    num_workers=4,
    memmap=False,
    train_tfs=train_tfs,
    validation_tfs=validation_tfs
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

model = TinyCNN()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)

callbacks = [BinaryClassificationMetrics(),
             SchedulerWrap(scheduler),
             SaveCheckpoint('ckpt.pt'),
             Mixup(alpha=0.3)]

learner = Learner(model, criterion, optimizer, callbacks=callbacks)
# learner.resume('ckpt.pt')
learner.fit(train_dl, validation_dl, max_epoch=200)

for threshold in [0.5, 0.3, 0.2, 0.1]:
    learner.cb = CallbackHandler([BinaryClassificationMetrics(threshold=threshold)])
    learner.validate(test_dl)
