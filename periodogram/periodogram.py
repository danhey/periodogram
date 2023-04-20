from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from .normalization import amplitude_spectrum, psd_muHz, power_spectrum


class Periodogram:
    """The big kahuna."""

    def __init__(self, frequency, power, meta={}):
        self.frequency = frequency
        self.power = power

        self.meta = meta

    @staticmethod
    def calculate(t: np.array, y: np.array, normalization: str = None, **kwargs):
        """Calculates the periodogram given the normalization. The normalization
        must be either one of `amplitude`, `power`, or `psd`. Note that if choosing
        `psd`, the periodogram will automatically convert the units such that the
        resulting spectrum is in microhertz and ppm^2/microhertz.

        Args:
            t (np.array): Input time values. Must be in units of days
            y (np.array): Input flux.
            normalization (str, optional): Normalization to use. Defaults to None.

        Returns:
            Periodogram: The normalized periodogram
        """
        normalization = normalization.lower()
        if normalization not in ["amplitude", "psd", "power"]:
            raise ValueError("Normalization must be one of `psd`, `amplitude`, `power")

        if normalization == "amplitude":
            return Periodogram(
                *amplitude_spectrum(t, y, **kwargs), meta={"normalization": "Amplitude"}
            )
        elif normalization == "psd":
            return Periodogram(*psd_muHz(t, y, **kwargs), meta={"normalization": "PSD"})
        elif normalization == "power":
            return Periodogram(
                *power_spectrum(t, y, **kwargs), meta={"normalization": "Power"}
            )

    def plot(self, ax=None):
        """Plots the periodogram

        Args:
            ax (_type_, optional): Matplotlib axis. Defaults to None.

        Returns:
            _type_: The plot
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.frequency, self.power, lw=0.7, c="k")
        ax.set(xlim=[self.frequency[0], self.frequency[-1]])
        ax.set_xlabel(r"Frequency [day$^{-1}$]")
        if self.meta["normalization"] == "PSD":
            ax.set(
                xlabel=r"Frequency [$\mu$Hz]",
                ylabel=r"PSD [ppm$^2$ / $\mu$Hz]",
                xscale="log",
                yscale="log",
            )
        elif self.meta["normalization"] == "Amplitude":
            ax.set_ylabel("Amplitude")
            ax.set_ylim(0, None)
        elif self.meta["normalization"] == "Power":
            ax.set_ylabel("Power")
            ax.set_ylim(0, None)
        return ax

    def estimate_background(self, log_width=0.01):
        """Estimates the background noise level in the periodogram using a moving log filter

        Args:
            log_width (float, optional): Width of the filter. Defaults to 0.01.

        Returns:
            np.array: Array of backgrounds
        """
        x, y = self.frequency, self.power
        count = np.zeros(len(x), dtype=int)
        bkg = np.zeros_like(x)
        x0 = np.log10(x[0])
        while x0 < np.log10(x[-1]):
            m = np.abs(np.log10(x) - x0) < log_width
            bkg[m] += np.median(y[m])
            count[m] += 1
            x0 += 0.5 * log_width
        return bkg / count

    # def flatten(self, **kwargs):
    #     bkg = self.estimate_background(**kwargs)
    #     return Periodogram(self.frequency, self.power / bkg, meta=self.meta)


# class AmplitudeSpectrum(Periodogram):
#     """A subclass of Periodogram for amplitude normalizations

#     Args:
#         Periodogram (_type_): _description_
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def plot(self, *args, **kwargs):
#         ax = super().plot(*args, **kwargs)
#         ax.set_xlabel(r"Frequency [day$^{-1}$]")
#         ax.set_ylabel("Amplitude")
#         return ax


# class PowerSpectrum(Periodogram):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def plot(self, *args, **kwargs):
#         ax = super().plot(*args, **kwargs)
#         ax.set_xlabel(r"Frequency [day$^{-1}$]")
#         ax.set_ylabel("Power")
#         return ax


# class PowerDensity(Periodogram):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def plot(self, *args, **kwargs):
#         ax = super().plot(*args, **kwargs)
#         ax.set_xlabel(r"Frequency [$\mu$Hz]")
#         ax.set_ylabel(r"PSD [ppm$^2$ / $\mu$Hz]")
#         ax.set_xscale("log")
#         ax.set_yscale("log")
#         return ax
