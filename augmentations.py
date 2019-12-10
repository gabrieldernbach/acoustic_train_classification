from librosa.effects import pitch_shift, percussive
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

    def __init__(self, x_length, y_length):
        self.x_length = x_length
        self.y_length = y_length

    def __call__(self, sample):
        # return cv2.resize(sample, (self.x_length, self.y_length))
        return resize(sample, (self.x_length, self.y_length))