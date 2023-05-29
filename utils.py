import numpy as np
import scipy as sp
import librosa
from config import *
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
import scipy
import random
import analysis as anal
# from processing import *
from helpers import *
from sklearn.metrics import mean_squared_error
import soundfile as sf

def rms(y):
    """
    Root mean square value of signal
    :param y:
    :return:
    """
    return np.sqrt(np.mean(y ** 2, axis=-1))

# tfs feature extraction
def entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return scipy.stats.entropy(counts, base=base)

def power(y):
    """
    Calculate power over each window [J/s]
    :param y:
    :return:
    """
    return sp.sum(y ** 2, 0) / y.size


def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def read_audio(filename):
    # x, fs = librosa.load(filename, sr=target_fs) # was producing error is a few datasets
    if '.flac' in filename[-5:]:
        x, fs = sf.read(filename)
    elif '.wav' in filename[-5:]:
        fs, x = read(filename)
    else:
        pass
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError('Error: NaN or Inf value. File path: %s.' % (filename))
    if not (isinstance(x[0], np.float32) or isinstance(x[0], np.float64)):
        x = x.astype('float32') / np.power(2, 15)
    if target_fs is not None and fs != target_fs:
        x = librosa.resample(x, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        # print('resampling...')
    return fs, x

def write_audio(filename, fs, wav):
    if isinstance(wav[0], np.float32) or isinstance(wav[0], np.float64):
        wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int16)
    write(filename, fs, wav)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def rmsle(y_true, y_pred):
    """
    sklearn function for RMSLE

    :param y_true:
    :param y_pred:
    :return:
    """

    return mean_squared_error(y_true, y_pred) ** 0.5


def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')


def remove_silent_frames(x, y, fs, dyn_range=30, framelen=4096, hop=1024):
    """
    Remove silent frames of x and y based on x
    A frame is excluded if its energy is lower than max(energy) - dyn_range
    The frame exclusion is based solely on x, the clean speech signal
    # Arguments :
        x : array, original speech wav file
        y : array, denoised speech wav file
        dyn_range : Energy range to determine which frame is silent. Speech dynamic range is around 40 dB
        framelen : Window size for energy evaluation
        hop : Hop size for energy evaluation
    # Returns :
        x without the silent frames
        y without the silent frames (aligned to x)
    """
    EPS = np.finfo("float").eps
    # Compute Mask
    w = sp.hanning(framelen + 2)[1:-1]

    x_frames = np.array(
        [w * x[i:i + framelen] for i in range(0, len(x) - framelen, hop)])
    y_frames = np.array(
        [w * y[i:i + framelen] for i in range(0, len(x) - framelen, hop)])

    # Compute energies in dB
    x_energies = 20 * np.log10(np.linalg.norm(x_frames, axis=1) + EPS)

    # Find boolean mask of energies lower than dynamic_range dB
    # with respect to maximum clean speech energy frame
    mask = (np.max(x_energies) - dyn_range - x_energies) < 0

    # Remove silent frames by masking
    x_frames = x_frames[mask]
    y_frames = y_frames[mask]

    # init zero arrays to hold x, y with silent frames removed
    x_sil = np.zeros(x_frames.shape[0] * hop + framelen) # np.zeros((mask.shape[0] - 1) * hop + framelen)
    y_sil = np.zeros(x_frames.shape[0] * hop + framelen) # np.zeros((mask.shape[0] - 1) * hop + framelen)
    for i in range(x_frames.shape[0]):
        x_sil[range(i * hop, i * hop + framelen)] += x_frames[i, :]
        y_sil[range(i * hop, i * hop + framelen)] += y_frames[i, :]

    return x_sil / 2., y_sil / 2.



def upsample(s_ds, factor, axis=-1):
    """
    upsample 1D signals by factor given.
    Upsampling is done in seq length dimension for all filters and time steps.
    :param s_ds: Numpy array. Input shape (time_steps, seq_length, filters)
    :param factor: Integer. Upsampling factor fs_new / fs_ds
    :return: signal with shape (time_steps, seq_length * factor, filters)
    """
    seq_length = s_ds.shape[axis]
    filters = s_ds.shape[axis-1]

    t_ds = np.array(range(seq_length))
    t_up = np.linspace(0, seq_length, num=int(seq_length * factor))

    s_up = np.zeros(
        (filters, seq_length * factor),
        dtype='float32')

    for f in range(filters):
        # create the interpolating function
        current_env = s_ds[f, :]
        f_exp = interp1d(t_ds, current_env, kind='cubic', bounds_error=False, fill_value='extrapolate')
        current_env_up = f_exp(t_up)

        for s in range(seq_length * factor):
            s_up[f, s] = current_env_up[s]

        if np.isnan(s_up).any():
            raise ValueError('Encountered NaN in upsampled singal')

    return s_up
