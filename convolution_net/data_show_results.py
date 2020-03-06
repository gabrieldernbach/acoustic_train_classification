import matplotlib.pyplot as plt
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from data_augmentation import ToTensor, Compose, TorchUnsqueeze
from data_extraction import *
from models.convolution_net import TinyCNN


class ShowResult:
    def __init__(self, pipeline, device):
        self.extraction, self.transform, self.model = pipeline
        self.device = device

    def predict(self, idx):
        audios, targets = self.extraction(idx)
        samples = torch.stack([self.transform(audio) for audio in audios])
        predictions = self.model(samples.to(self.device)).cpu().detach().numpy()
        return audios, samples, targets, predictions

    def __call__(self, idx):
        audios, samples, targets, predictions = self.predict(idx)

        for i in range(len(samples)):
            false_positive = (targets[i] == 0) and (predictions[i] > 0.9)
            false_negative = (targets[i] == 1) and (predictions[i] < 0.2)

            if false_positive:
                case_id = self.extraction.data_register.iloc[idx].audio_path
                case_id = case_id.split('/')[-1].split('.')[0]
                fname = f'{case_id}_frame_{i}'
                print('saving', fname)
                # save audio
                librosa.output.write_wav(f'results/false_positive/{fname}.wav', audios[i], sr=8192)
                # save plot
                plt.imshow(samples[i].squeeze())
                plt.title(f'{fname}')
                plt.savefig(f'results/false_positive/{fname}.png')

            # if false_negative:
            #     case_id = self.extraction.data_register.iloc[idx].audio_path
            #     case_id = case_id.split('/')[-1].split('.')[0]
            #     fname = f'{case_id}_frame_{i}'
            #     print('saving', fname)
            #     # save audio
            #     librosa.output.write_wav(f'results/false_negative/{fname}.wav', audios[i], sr=8192)
            #     # save plot
            #     plt.imshow(samples[i].squeeze())
            #     plt.title(f'{fname}')
            #     plt.savefig(f'results/false_negative/{fname}.png')

    def all(self):
        for i in range(len(self)):
            self(i)

    def __len__(self):
        return len(self.extraction.data_register)


if __name__ == "__main__":
    sample_extraction_tfs = [
        LoadAudio(fs=48000),
        Resample(target_fs=8192),
        Frame(frame_length=16384, hop_length=16384)]
    target_extraction_tfs = [
        LoadTargets(fs=48000),
        Resample(target_fs=8192),
        Frame(frame_length=16384, hop_length=16384),
        AvgPoolTargets(threshold=0.125)]

    extractor = RegisterExtractor(data_register='../data/data_register_dev.pkl',
                                  sample_tfs=sample_extraction_tfs,
                                  target_tfs=target_extraction_tfs)

    test_tf = Compose([
        ToTensor(),
        MelSpectrogram(sample_rate=8192, n_fft=512, hop_length=128, n_mels=40),
        AmplitudeToDB(stype='power', top_db=80),
        TorchUnsqueeze(0),
    ])

    model = TinyCNN()
    show_result = ShowResult((extractor, test_tf, model))
    show_result.all()
