import numpy as np
from librosa.effects import percussive
from librosa.effects import pitch_shift
from librosa.feature.spectral import melspectrogram
from scipy.signal import spectrogram
from skimage.transform import resize


class PitchShift(object):

    def __init__(self,
                 sr=48000,
                 n_steps=4,
                 bins_per_octave=24.,
                 res_type='kaiser_fast'):
        self.sr = sr
        self.n_steps = n_steps
        self.bins_per_octave = bins_per_octave
        self.res_type = res_type

    def __call__(self, sample):
        sample = pitch_shift(self.sr,
                             self.n_steps,
                             self.bins_per_octave,
                             self.res_type)
        return sample


class MelSpectrogram(object):

    def __init__(self, sr=48000, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, sample):
        sample = np.ascontiguousarray(sample)
        spec = melspectrogram(y=sample,
                              sr=self.sr,
                              n_fft=self.n_fft,
                              hop_length=self.hop_length)
        logspec = np.log(spec + 1e-9)
        logspec -= np.mean(logspec)
        return logspec


class AdjustAmplitude(object):

    def __init__(self, offset_in_db):
        self.offset_in_db = offset_in_db
        self.factor = 10 ** (offset_in_db / 20)

    def __call__(self, sample):
        return self.factor * sample


class Spectrogram(object):

    def __init__(self, nperseg=1024, noverlap=768):
        self.nperseg = nperseg
        self.noverlap = noverlap

    def __call__(self, sample):
        spec = spectrogram(sample,
                           nperseg=self.nperseg,
                           noverlap=self.noverlap)
        return spec[2]


class PercussiveSeparation(object):

    def __init__(self, margin=3.0):
        self.margin = margin

    def __call__(self, sample):
        return percussive(sample, margin=self.margin)


class Resize(object):

    def __init__(self, x_length=300, y_length=300):
        self.x_length = x_length
        self.y_length = y_length

    def __call__(self, sample):
        # return cv2.resize(sample, (self.x_length, self.y_length))
        return resize(sample, (self.x_length, self.y_length))


class ExpandDim(object):

    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, sample):
        return np.expand_dims(sample, self.axis)
