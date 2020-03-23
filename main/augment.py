import librosa
import numpy as np
import torch
from kymatio import Scattering1D
from librosa.effects import percussive
from librosa.effects import pitch_shift


class PitchShift(object):
    def __init__(self,
                 sr=8_192,
                 pitch_shift_range_cent=200,
                 bins_per_octave=1_200,
                 res_type='kaiser_fast',
                 **kwargs):
        """

        Parameters
        ----------
        sr : int
        n_steps : [int, int]
            fractional steps to shift by (width determined by bins per octave)
            the values get sampled uniformly in the given domain [low, high]
        bins_per_octave : float
            determines the step width of n_step
        res_type : str
            resampling type
        """
        self.sr = sr
        self.bins_per_octave = bins_per_octave
        self.res_type = res_type
        self.n_steps = pitch_shift_range_cent

    def __call__(self, sample):
        n_steps = np.random.randint(-self.n_steps, self.n_steps)

        sample = pitch_shift(sample, self.sr, n_steps, self.bins_per_octave, self.res_type)
        return sample


class LogCompress(object):
    """
    apply logarithmic compression of `ratio`
    """

    def __init__(self, ratio=1., eps=1e-9):
        self.ratio = ratio
        self.eps = eps

    def __call__(self, sample):
        return np.log(sample * self.ratio + self.eps)


class PercussiveSeparation(object):
    """
    separates and returns the percussive part of an audio recording
    """

    def __init__(self, margin=3.0):
        self.margin = margin

    def __call__(self, sample):
        return percussive(sample, margin=self.margin)


class TorchUnsqueeze(object):
    """
    adds singleton dimension representing the channels
    """

    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, sample):
        return sample.unsqueeze(dim=self.dim)


class ThresholdPoolSequence(object):
    """
    infers the ratio of true labels of a sequence
    and returns if the ratio is bigger `threshold`
    """

    def __init__(self, threshold=0.125):
        self.threshold = threshold

    def __call__(self, sample):
        non_zero_ratio = (sample > 0).sum() / len(sample)
        label = np.array(non_zero_ratio > self.threshold).astype('float32')
        label = np.expand_dims(label, 0)
        return label


class ShortTermAverageTransform(object):
    """
    subsamples a target time series by taking the average in the
    same frame and hop specs as the a respective STFT would
    """

    def __init__(self, frame_length, hop_length, threshold=0.5):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.threshold = threshold

    def __call__(self, sample):
        sample = np.pad(sample, self.frame_length // 2, mode='reflect')
        framed = librosa.util.frame(sample,
                                    frame_length=self.frame_length,
                                    hop_length=self.hop_length)
        pooled = np.mean(framed, axis=0)

        thresholded = (pooled > self.threshold).astype('int')
        return thresholded


class Compose(object):
    """Composes several transforms in to a single chain.

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
    """
    converts np.ndarray and returns torch.tensor
    """

    def __call__(self, array):
        return torch.from_numpy(array)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalizer(object):
    """
    applies channel wise standard normalization
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, audio):
        audio -= self.mean
        audio /= self.std
        return audio

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Scatter1D:
    """
    Alternative to Melspectrogram:
    see https://arxiv.org/pdf/1304.6763.pdf for details
    """

    def __init__(self, J=6, T=2 ** 14, Q=7, device='cuda'):
        self.scattering = Scattering1D(J, T, Q)
        self.log_eps = 1e-6

        if device is 'cuda':
            self.scattering = self.scattering.cuda()

    def __call__(self, samples):
        Sx = self.scattering.forward(samples)
        return torch.log(Sx + self.log_eps)[:, 1:, :]
