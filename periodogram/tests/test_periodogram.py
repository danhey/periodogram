from ..normalization import amplitude_spectrum, psd, power_spectrum

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
    freq, p = psd(x, y)
    assert np.isclose(p.max(), amplitude**2 * (x[-1] - x[0]), rtol=1e-3)


def test_power_normalization():
    freq, power = power_spectrum(x, y)
    assert np.isclose(np.max(power), amplitude**2, rtol=1e-3)
