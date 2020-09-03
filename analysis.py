from scipy.signal import hilbert, butter, filtfilt, decimate
import scipy
from scipy.interpolate import interp1d
from pambox.inner import GammatoneFilterbank, hilbert_envelope, lowpass_env_filtering, erb_bandwidth
from utils import *

# constants
cutoff = 32

cf_default = np.asarray([63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
                          800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
                          6300, 8000])


def erbspace(low, high, N, earQ=9.26449, minBW=24.7, order=1):
    '''
    Returns the centre frequencies on an ERB scale.

    ``low``, ``high``
        Lower and upper frequencies
    ``N``
        Number of channels
    ``earQ=9.26449``, ``minBW=24.7``, ``order=1``
        Default Glasberg and Moore parameters.
    '''
    low = float(low)
    high = float(high)
    cf = -(earQ * minBW) + np.exp((np.arange(N)) * (-np.log(high + earQ * minBW) + np.log(low + earQ * minBW)) / (N - 1)) * (high + earQ * minBW)
    cf = cf[::-1]
    return cf

center_freq = erbspace(80,8000, 64)


def low_pass_filter(signal, fs, cutoff):
    B, A = butter(1, cutoff / (fs / 2), btype='low')
    filtered_signal = filtfilt(B, A, signal, axis=-1)
    return filtered_signal


def high_pass_filter(signal, fs, cutoff):
    B, A = butter(1, cutoff / (fs / 2), btype='high')
    filtered_signal = filtfilt(B, A, signal, axis=-1)
    return filtered_signal

# input: speech signal, samling frequency
# output: temporal fine structure, instantaneous frequencies
def get_tfs(speech, fs):
    # speech is 1d shape: (time,)
    # speech is 2d shape: (freq, time)
    analytic_signal = hilbert(speech)  # form the analytical signal
    inst_phase = np.unwrap(np.angle(analytic_signal))
    inst_freq = np.diff(inst_phase) / (2 * np.pi) * fs  # inst frequency
    # Regenerate the carrier from the instantaneous phase
    tfs = np.cos(inst_phase)

    return tfs, inst_freq


def get_env(x, fs, filt=True):
    """
    get filtered temporal envelope of signal x

    input: signal, sampling frequency
    output:
    :param x:
    :param fs:
    :return:
    """
    x_env = hilbert_envelope(x)

    if filt: return lowpass_env_filtering(x_env, cutoff=cutoff, n=4, fs=fs)
    else: return x_env


def get_octave_bands(min_freq=None, max_freq=None):
    if min_freq is None and max_freq is None:
        fcenter = 2. ** np.arange(-6, 5, 1) * (10 ** 3)
    else:
        n = np.ceil(np.log2(max_freq / min_freq)) + 1.
        fcenter = min_freq * 2. ** np.arange(0., n, 1.)
    fd = 2. ** (1. / 2.)
    fupper = fcenter * fd
    flower = fcenter / fd
    return flower, fupper

def erb_frilter_bank(x, fcoefs):
    a0 = fcoefs[:, 0]
    a11 = fcoefs[:, 1]
    a12 = fcoefs[:, 2]
    a13 = fcoefs[:, 3]
    a14 = fcoefs[:, 4]
    a2 = fcoefs[:, 5]
    b0 = fcoefs[:, 6]
    b1 = fcoefs[:, 7]
    b2 = fcoefs[:, 8]
    gain = fcoefs[:, 9]

    output = np.zeros((np.size(gain, 0), np.size(x, 0)))

    for chan in range(np.size(gain, 0)):
        y1 = lfilter(np.array([a0[chan] / gain[chan], a11[chan] / gain[chan], a2[chan] / gain[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), x)
        y2 = lfilter(np.array([a0[chan], a12[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y1)
        y3 = lfilter(np.array([a0[chan], a13[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y2)
        y4 = lfilter(np.array([a0[chan], a14[chan], a2[chan]]),
                     np.array([b0[chan], b1[chan], b2[chan]]), y3)

        output[chan, :] = y4
    return output


def freq_to_aud(freq, audscale):
    aud = freq
    if audscale is 'mel':
        aud = 1000 / np.log(17 / 7) * np.sign(freq) * np.log(1 + np.abs(freq) / 700)
    if audscale is 'mel1000':
        aud = 1000 / np.log(2) * np.sign(freq) * np.log(1 + np.abs(freq) / 1000)
    if audscale is 'erb':
        aud = 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)
    if audscale is 'freq':
        aud = freq
    return aud

def aud_to_freq(aud, audscale):
    freq = aud
    if audscale is 'mel':
        freq = 700 * np.sign(aud) * (np.exp(np.abs(aud) * np.log(17 / 7) / 1000) - 1)
    if audscale is 'mel1000':
        freq = 1000 * np.sign(aud) * (np.exp(np.abs(aud) * np.log(2) / 1000) - 1)
    if audscale is 'erb':
        freq = (1 / 0.00437) * np.sign(aud) * (np.exp(np.abs(aud) / 9.2645) - 1)
    if audscale is 'freq':
        freq = aud
    return freq


def audspace(fmin, fmax, n, auscale='erb'):
    """
    Python Wrapper for matlab's auditory sclae conversion.
    Equidistantly spaced points on auditory scale
    %   Usage: y=audspace(fmin,fmax,n,scale);
    %   AUDSPACE(fmin,fmax,n,scale) computes a vector of length n*
    %   containing values equidistantly scaled on the selected auditory scale
    %   between the frequencies fmin and fmax. All frequencies are
    %   specified in Hz.
    %   Url: http://ltfat.github.io/doc/auditory/audspace.html

    """
    assert fmin >=0, fmax > fmin
    low_limit, upper_limit = freq_to_aud([fmin, fmax], auscale)
    y = aud_to_freq(np.linspace(low_limit, upper_limit, n), auscale)
    bw = (upper_limit - low_limit) / (n-1)
    y[0], y[-1] = fmin, fmax
    return y, bw


