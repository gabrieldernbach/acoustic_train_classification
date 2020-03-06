import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from data_augmentation import ToTensor, Compose, TorchUnsqueeze
from data_extraction import LoadAudio, Frame, ResampleSpeedNormalization
from data_extraction import LoadTargets, AvgPoolTargets
from models.convolution_net import TinyCNN

# export results
sample_extraction_tfs = [
    LoadAudio(fs=48000),
    ResampleSpeedNormalization(target_fs=8192),
    Frame(frame_length=16384, hop_length=16384)]
# ShortTermMelTransform(fs=8000, n_fft=512, hop_length=128, n_mels=40)]
target_extraction_tfs = [
    LoadTargets(fs=48000),
    ResampleSpeedNormalization(target_fs=8192),
    Frame(frame_length=16384, hop_length=16384),
    AvgPoolTargets(threshold=0.125)]
train_tfs = Compose([
    ToTensor(),
    MelSpectrogram(sample_rate=8192, n_fft=512, hop_length=128, n_mels=40),
    AmplitudeToDB(stype='power', top_db=80),
    TorchUnsqueeze(),
])
validation_tfs = Compose([
    ToTensor(),
    MelSpectrogram(sample_rate=8192, n_fft=512, hop_length=128, n_mels=40),
    AmplitudeToDB(stype='power', top_db=80),
    TorchUnsqueeze(),
])

from data_loading import fetch_dataloaders

_, _, test_ld, normalizer = fetch_dataloaders(512, num_workers=4, train_tfs=train_tfs, validation_tfs=validation_tfs,
                                              memmap=True, normalize=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TinyCNN()

checkpoint = torch.load('ckpt2.pt', map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
#
# extractor = RegisterExtractor(data_register='../data/data_register_dev.pkl',
#                               sample_tfs=sample_extraction_tfs,
#                               target_tfs=target_extraction_tfs)
# show_result = ShowResult((extractor, validation_tfs, model), device=device)
# show_result.all()

from torch_learner import Learner
import torch.nn as nn
from torch_callbacks import BinaryClassificationMetrics

learner = Learner(model=model, criterion=nn.BCELoss(), optimizer=torch.optim.Adam(params=model.parameters()),
                  callbacks=[BinaryClassificationMetrics(threshold=0.5)])
# starting validation
learner.validate(test_ld)
