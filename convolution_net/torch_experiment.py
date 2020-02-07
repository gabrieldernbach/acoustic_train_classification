import torch.backends.cudnn
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from data_augmentation import ToTensor, Compose
from data_feature_extraction import LoadAudio, Frame
from data_feature_extraction import LoadTargets, AvgPoolTargets, Resample
from data_feature_extraction import RegisterExtractor
from data_feature_extraction import extract_to_disk
from data_loading import fetch_dataloaders
from data_show_results import ShowResult
from models.convolution_net import TinyCNN
from torch_callbacks import BinaryClassificationMetrics, Mixup, SchedulerWrap, SaveCheckpoint, CallbackHandler
from torch_learner import Learner

# feature pre processing
sample_extraction_tfs = [
    LoadAudio(fs=48000),
    Resample(target_fs=8192),
    Frame(frame_length=16384, hop_length=4096)]
target_extraction_tfs = [
    LoadTargets(fs=48000),
    Resample(target_fs=8192),
    Frame(frame_length=16384, hop_length=4096),
    AvgPoolTargets(threshold=0.125)]
extract_to_disk(sample_extraction_tfs, target_extraction_tfs)

train_tfs = Compose([
    ToTensor(),
    MelSpectrogram(sample_rate=8192, n_fft=512, hop_length=128, n_mels=40),
    AmplitudeToDB(stype='power', top_db=80),
])
validation_tfs = Compose([
    ToTensor(),
    MelSpectrogram(sample_rate=8192, n_fft=512, hop_length=128, n_mels=40),
    AmplitudeToDB(stype='power', top_db=80)
])

train_dl, validation_dl, test_dl = fetch_dataloaders(
    batch_size=256,
    num_workers=1,
    memmap=False,
    train_tfs=train_tfs,
    validation_tfs=validation_tfs
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

print('init model')
model = TinyCNN()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)

callbacks = [BinaryClassificationMetrics(),
             SchedulerWrap(scheduler),
             SaveCheckpoint('ckpt.pt'),
             Mixup(alpha=0.3)]

print('setup learner')
learner = Learner(model, criterion, optimizer, callbacks=callbacks)
learner.fit(train_dl, validation_dl, max_epoch=1)
learner.resume('ckpt.pt')

for threshold in [0.5, 0.3, 0.2, 0.1]:
    learner.cb = CallbackHandler([BinaryClassificationMetrics(threshold=threshold)])
    learner.validate(test_dl)

# show results
sample_extraction_tfs = [
    LoadAudio(fs=48000),
    # ResampleSpeedNormalization(target_fs=8000, target_speed=50),
    Resample(target_fs=8192),
    Frame(frame_length=16384, hop_length=16384)]
# ShortTermMelTransform(fs=8000, n_fft=512, hop_length=128, n_mels=40)]
extractor = RegisterExtractor(data_register='../data/data_register_dev.pkl',
                              sample_tfs=sample_extraction_tfs,
                              target_tfs=target_extraction_tfs)
show_result = ShowResult((extractor, validation_tfs, model))
show_result.all()
