from ..periodogram import amplitude_spectrum, psd, power_spectrum

import numpy as np
from astropy.timeseries import LombScargle

#  A sine wave with amplitude A, should have a height of A in the amplitude spectrum, A^2 in the power spectrum, A^2 * Tobs in the power density spectrum.

amplitude = 5
frequency = 0.1
x = np.arange(0, 100, 0.1)
y = amplitude * np.sin(2 * np.pi * frequency * x)


def test_amplitude_normalization():
    freq, amp = amplitude_spectrum(x, y)
    assert np.isclose(np.max(amp), amplitude, rtol=1e-3)


def test_psd_normalization():
    tmax = x.max()
    tmin = x.min()
    df = 1.0 / (tmax - tmin)
    fmin = df
    fmax = 0.5 / np.median(np.diff(x))  # *nyq_mult

    N = len(x)

    freq_window = np.arange(fmin, fmax, df / 1)
    nu = 0.5 * (fmin + fmax)
    power_window = (
        LombScargle(x, np.sin(2 * np.pi * nu * x)).power(
            freq_window, normalization="psd"
        )
        / N
        * 4.0
    )
    Tobs = 1.0 / np.sum(np.median(freq_window[1:] - freq_window[:-1]) * power_window)

    freq, p = psd(x, y, fmin=fmin, fmax=fmax)
    assert np.isclose(p.max(), amplitude**2 * Tobs, rtol=1e-3)


def test_power_normalization():
    freq, power = power_spectrum(x, y)
    assert np.isclose(np.max(power), amplitude**2, rtol=1e-3)
