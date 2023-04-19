from __future__ import division, print_function

import numpy as np
from astropy.timeseries import LombScargle


def amplitude_spectrum(
    t: np.array,
    y: np.array,
    fmin: float = None,
    fmax: float = None,
    oversample_factor: float = 5.0,
    freq: np.array = None,
) -> tuple:
    """Calculates the amplitude spectrum of input data.

    Args:
        t (np.array): Input times
        y (np.array): Input fluxes
        fmin (float, optional): Minimum frequency. Defaults to None.
        fmax (float, optional): Maximum frequency. Defaults to None.
        oversample_factor (float, optional): The oversampling rate. Defaults to 5.0.
        freq (np.array, optional): Optional custom frequency. Defaults to None.

    Returns:
        tuple: Frequency and amplitude
    """
    # t, y = self.time, self.residual
    if freq is None:
        tmax = t.max()
        tmin = t.min()
        df = 1.0 / (tmax - tmin)

        if fmin is None:
            fmin = df
        if fmax is None:
            fmax = 0.5 / np.median(np.diff(t))  # *nyq_mult

        freq = np.arange(fmin, fmax, df / oversample_factor)
    model = LombScargle(t, y)
    sc = model.power(freq, method="fast", normalization="psd")

    fct = np.sqrt(4.0 / len(t))
    amp = np.sqrt(sc) * fct

    return freq, amp


def psd(
    t: np.array,
    y: np.array,
    fmin: float = None,
    fmax: float = None,
    oversample_factor: int = 1,
) -> tuple:
    """Calculates the power spectral density spectrum of input data.
    Note that the time is assumed to be in `days'. The output frequency is converted to uHz.

    Args:
        t (np.array): Input times
        y (np.array): Input fluxes
        fmin (float, optional): Minimum frequency. Defaults to None.
        fmax (float, optional): Maximum frequency. Defaults to None.
        oversample_factor (float, optional): The oversampling rate. Defaults to 5.0.

    Returns:
        tuple: Frequency and power spectral density
    """
    tmax = t.max()
    tmin = t.min()
    df = 1.0 / (tmax - tmin)
    if fmin is None:
        fmin = df
    if fmax is None:
        fmax = 0.5 / np.median(np.diff(t))  # *nyq_mult

    N = len(t)
    freq = np.arange(fmin, fmax, df / oversample_factor)

    nu = 0.5 * (fmin + fmax)
    freq_window = np.arange(fmin, fmax, df / oversample_factor)
    power_window = (
        LombScargle(t, np.sin(2 * np.pi * nu * t)).power(
            freq_window, normalization="psd"
        )
        / N
        * 4.0
    )
    Tobs = 1.0 / np.sum(np.median(freq_window[1:] - freq_window[:-1]) * power_window)
    p = (LombScargle(t, y).power(freq, normalization="psd") / N * 4.0) * Tobs

    return freq, p


def power_spectrum(*args, **kwargs) -> tuple:
    """A simple wrapper to return the power spectrum

    Returns:
        tuple: Frequency and power
    """
    freq, amp = amplitude_spectrum(*args, **kwargs)
    return freq, amp**2


def psd_muHz(*args, **kwargs) -> tuple:
    freq, p = psd(*args, **kwargs)

    freq = freq / (24 * 3600) * 1e6  # c/d to muHz
    p = p * (24 * 3600) * 1e-6  #  # ppm^2/(c/d) to ppm^2/muHz
    return freq, p
