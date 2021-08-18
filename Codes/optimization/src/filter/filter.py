import numpy as np
from numpy.random import randn
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter, unscented_transform
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import math
from scipy.signal import butter, lfilter
from scipy.signal import freqs
import sympy as sp

from lmfit import Parameters
from ..solver import Solver_jac


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def lowpass_filter(data):
    # Filter requirements.
    order = 6
    fs = 10  # sample rate, Hz
    cutoff = 2  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)

    # Plot the frequency response.
    # w, h = freqz(b, a, worN=8000)
    # plt.subplot(2, 1, 1)
    # plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    # plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    # plt.axvline(cutoff, color='k')
    # plt.xlim(0, 0.5*fs)
    # plt.title("Lowpass Filter Frequency Response")
    # plt.xlabel('Frequency [Hz]')
    # plt.grid()

    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    T = 5.0  # seconds
    n = int(T * fs)  # total number of samples
    # t = np.linspace(0, T, n, endpoint=False)
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    # data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + \
    #     0.5*np.sin(12.0*2*np.pi*t)
    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(data, cutoff, fs, order)
    return y


def mean_filter(data, win=3):
    result = data.copy()
    size = data.size
    for i in range(1, size - win + 1):
        result[i] = np.mean(result[i:i + win])
    return result


def median_filter(data, win=3):
    result = data.copy()
    size = data.size
    for i in range(1, size - win + 1):
        result[i] = np.median(result[i:i + win])
    return result
