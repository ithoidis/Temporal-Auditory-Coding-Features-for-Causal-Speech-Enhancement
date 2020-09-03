from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import *
from matplotlib import pyplot as plt
from scipy.signal import stft
import scipy
from config import *
import librosa
import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
from pambox.inner import GammatoneFilterbank
from analysis import audspace, low_pass_filter, high_pass_filter, get_tfs, get_env, upsample
from pambox.inner import hilbert_envelope


def Mc_mask(x, y, method='naive', dyn_range=40):
    """
    IRM = ((signal_power) / ((signal_power)+(noise_power))) ^ (β), where β = 1/2 usually
    Safe implementation here.
    :param x:  Magnitudes 3-dim array (BATCH, freq, time)
    :param y:  Magnitudes 3-dim array (BATCH, freq, time)
    :return: IRM
    """
    assert np.amin(x) >= 0.0
    assert np.amin(y) >= 0.0
    beta = 0.5
    x_new = np.where(x < y, y, x)
    mask =  np.power((y**2) / (x_new**2+ np.finfo(float).eps), beta)
    mask = np.clip(mask, a_min=0.0, a_max=1.0)
    return mask


def apply_mask(x, y, method='naive', dyn_range=40):
    y = np.clip(y, a_min=0.0, a_max=1.0)
    return x * y

def inv_magphase(mag, phase_angle):
  phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
  return mag * phase

def specgram(audio, n_fft=512, hop_length=None, window='hann', center=True, mask=False, log_mag=False, re_im=False, dphase=False, mag_only=False):
  """Spectrogram using librosa.

  Args:
    audio: 1-D array of float32 sound samples.
    n_fft: Size of the FFT.
    hop_length: Stride of FFT. Defaults to n_fft/2.
    mask: Mask the phase derivative by the magnitude.
    log_mag: Use the logamplitude.
    re_im: Output Real and Imag. instead of logMag and dPhase.
    dphase: Use derivative of phase instead of phase.
    mag_only: Don't return phase.

  Returns:
    specgram: [n_fft/2 + 1, audio.size / hop_length, 2]. The first channel is
      the logamplitude and the second channel is the derivative of phase.
  """
  if not hop_length:
    hop_length = int(n_fft / 2.)

  fft_config = dict( n_fft=n_fft, win_length=n_fft, hop_length=hop_length, center=center, window=window)

  spec = librosa.stft(audio, **fft_config)

  if re_im:
    re = spec.real[:, :, np.newaxis]
    im = spec.imag[:, :, np.newaxis]
    spec_real = np.concatenate((re, im), axis=2)
  else:
        mag, phase = librosa.core.magphase(spec)
        phase_angle = np.angle(phase)

        if dphase:
          #  Derivative of phase
          phase_unwrapped = np.unwrap(phase_angle)
          p = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
          p = np.concatenate([phase_unwrapped[:, 0:1], p], axis=1) / np.pi
        else:
          # Normal phase
          p = phase_angle / np.pi
        # Mask the phase
        if log_mag and mask:
          p = mag * p
        # Return Mag and Phase
        p = p.astype(np.float32)[:, :, np.newaxis]
        mag = mag.astype(np.float32)[:, :, np.newaxis]
        if mag_only:
          spec_real = mag[:, :, np.newaxis]
        else:
          spec_real = np.concatenate((mag, p), axis=2)
  return spec_real


def ispecgram(spec, n_fft=512, hop_length=None, window='hann', center=True, mask=False, log_mag=False, re_im=False, dphase=False, mag_only=False, normalize=False, num_iters=1000):
  """Inverse Spectrogram using librosa.

  Args:
    spec: 3-D specgram array [freqs, time, (mag_db, dphase)].
    n_fft: Size of the FFT.
    hop_length: Stride of FFT. Defaults to n_fft/2.
    mask: Reverse the mask of the phase derivative by the magnitude.
    log_mag: Use the logamplitude.
    re_im: Output Real and Imag. instead of logMag and dPhase.
    dphase: Use derivative of phase instead of phase.
    mag_only: Specgram contains no phase.
    num_iters: Number of griffin-lim iterations for mag_only.

  Returns:
    audio: 1-D array of sound samples. Peak normalized to 1.
  """
  if not hop_length:
    hop_length = n_fft // 2

  ifft_config = dict(win_length=n_fft, hop_length=hop_length, center=center, window=window)

  if mag_only:
    mag = spec[:, :, 0]
    phase_angle = np.pi * np.random.rand(*mag.shape)
  elif re_im:
    #
    spec_real = spec[:, :, 0] + 1.j * spec[:, :, 1]
  else:
    mag, p = spec[:, :, 0], spec[:, :, 1]
    if mask and log_mag:
      p /= (mag + 1e-13 * np.random.randn(*mag.shape))
    if dphase:
      # Roll up phase
      phase_angle = np.cumsum(p * np.pi, axis=1)
    else:
      phase_angle = p * np.pi

  phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
  spec_real = mag * phase

  if mag_only:
    audio = griffin_lim(mag, phase_angle, n_fft, hop_length, num_iters=num_iters)
  else:
    audio = librosa.core.istft(spec_real, **ifft_config)

  if normalize:
      return np.squeeze(audio) / audio.max()
  else:
      return np.squeeze(audio)


def envelope_spectrogram(x, fs, fmin, fmax, nbands, tau_ms=8, use_hilbert=False, order=4, q=9.26):
    # Cochlear filtering
    fmin = 80 # corresponds to 4.55 filters per band at nsgt erb, 2.5 for 64 filters
    nsgt = NSGT_ERB(fs, len(x), 4.55, cutoff_frequency=6000, plot=False)# 82.1 to 6000 Hz with 128 filters
    xf = np.real(nsgt.forward_full_temp(x))
    xf = xf[13:, :]

    if use_hilbert:
        env = hilbert_envelope(xf)  # (Bands, Time)
        env = low_pass_filter(env, fs, cutoff=50)
    else:
        env = np.maximum(xf, 0)
        env = low_pass_filter(env, fs, cutoff=50)
        env = np.maximum(env, 1e-9)
        # match energy
        scale_factor = np.sqrt(np.sum(np.square(xf), axis=-1) / np.sum(np.square(env), axis=-1))
        env = np.multiply(env, scale_factor[:, np.newaxis])

    # Integration
    tau = int(tau_ms / 1000 * fs)  # 8ms
    win_slope =  1
    window = np.exp(-np.linspace(0, win_slope * tau - 1, tau) / tau)
    window = np.transpose(np.repeat(window[:, np.newaxis], nbands, axis=1))  # (frequency, time)
    y = np.transpose([np.sqrt(np.sum(window * env[:, i:i + tau]**2, axis=-1)) for i in range(0, env.shape[1] - tau, tau)])
    return y ** 0.5

def ienvelope_spectrogram(S, xn, fs, fmin, fmax, nbands, tau_ms=8, use_hilbert=False, order=4, q=9.26):
    # Cochlear filtering
    fmin = 80
    nsgt = NSGT_ERB(fs, len(xn), 4.55, cutoff_frequency=fmax, plot=False)  # 82.1 to 6000 Hz with 128 filters
    xf = np.real(nsgt.forward_full_temp(xn))
    xf = xf[13:, :]

    # calculate TFS
    tfs, _ = get_tfs(xf, fs)
    fs_ds = int(1 / (tau_ms / 1000))

    # upsample S envelope and modulate tfs
    S = np.square(S)
    S_up = upsample(S, fs // fs_ds)
    S_up = np.maximum(S_up, 1e-12)
    S_up = low_pass_filter(S_up, fs, cutoff=50)

    # trim original sound
    tfs = tfs[:,:S_up.shape[1]]
    S_up = S_up[:,:tfs.shape[1]]
    y = np.multiply(S_up, tfs)

    y = np.sum(y, axis=0)

    return y

def tfs_spectrogram(x, fs, fmin, fmax, nbands, tau_ms=8, use_hilbert=False):
    # Cochlear filtering
    fmin = 125
    nsgt = NSGT_ERB(fs, len(x), 4.55, cutoff_frequency=fmax, plot=False)  # 82.1 to 6000 Hz with 128 filters
    xf = np.real(nsgt.forward_full_temp(x))
    xf = xf[13:72, :]
    # TFS-route
    tfs = np.heaviside(xf, 0)
    tfs = low_pass_filter(tfs, fs, cutoff=2000)
    # Lateral Inhibitory Network #
    # derivative along the tonotopic axis.
    tfss = tfs[:-1, :] - tfs[1:, :]
    tfss = np.concatenate((tfss, [0.0*tfs[-1, :]]), axis=0)  #
    # half-wave rectification
    tfs_rect = np.maximum(tfss, 0)
    tfs_lin = low_pass_filter(tfs_rect, fs, cutoff=2000) # was 10 original
    # Integration
    tau = int(tau_ms / 1000 * fs)  # 8ms
    tfs_shape = (tfs_lin.shape[0], tfs_lin.shape[1] // tau - 1)
    tfs_out = np.zeros(tfs_shape)
    for index, i in enumerate(range(0, tfs_lin.shape[1] - tau - 1, tau)): # (freq, time)
        for f in range(tfs_shape[0]):
            # Differential excitation / fine structure adaptation / Phase-lock
            tfs_out[f, index] = np.sum(np.maximum(np.diff(tfs_lin[f, i:i + tau]),0)) / np.sqrt(f+1)
    return np.array(tfs_out)


def preemphasis(y, coef=0.97, zi=None, return_zf=False):
    '''Pre-emphasize an audio signal with a first-order auto-regressive filter:
        y[n] -> y[n] - coef * y[n-1]
    '''
    return scipy.signal.lfilter([1.0, -coef], [1.0], y)

def deemphasis(y, coef=0.97, zi=None, return_zf=False):
    '''Restore the Pre-emphasize effect of an audio signal with a first-order auto-regressive filter:
            y[n] -> y[n] + coef * y[n-1]
    '''
    return scipy.signal.lfilter([1], [1, -coef], y)
