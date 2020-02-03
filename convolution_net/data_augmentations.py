import numpy as np
import torch
from kymatio import Scattering1D
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


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, array):
        for t in self.transforms:
            array = t(array)
        return array

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    def __call__(self, array):
        """
        Args: np.ndarray
            array to be converted to tensor.

        Returns:
            Tensor: Converted array.
        """
        return torch.from_numpy(array)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Scatter1D:
    def __init__(self, J=6, T=2 ** 14, Q=7):
        self.scattering = Scattering1D(J, T, Q)
        self.log_eps = 1e-6

    def __call__(self, samples):
        Sx = self.scattering.forward(samples)
        return torch.log(Sx + self.log_eps)
