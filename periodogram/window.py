import tqdm
import numpy as np
import matplotlib.pyplot as plt

from .periodogram import amplitude_spectrum


def gauss(x, A, x0, sigma):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def sliding_window(
    time, flux, fmin, fmax, segment_size=5, gwidth=10, plot=True, cmap="Blues"
):
    t0s = np.arange(time.min(), time.max(), segment_size)
    amps = []
    for i in tqdm.tqdm(range(len(t0s[:-1]))):
        g = gauss(time, 1, t0s[i], gwidth)
        # g = boxcar(time, t0s[i], gwidth)
        f, a = amplitude_spectrum(time, flux * g, fmin=fmin, fmax=fmax)
        amps.append(a)

    if plot:
        fig, axes = plt.subplots(
            2, 2, gridspec_kw={"height_ratios": [0.5, 1], "width_ratios": [1, 0.5]}
        )
        axes = axes.flatten()

        ax = axes[0]
        ax.plot(time, flux, c="k", lw=0.8)
        ax.set(xlim=[time.min(), time.max()], xticks=[], ylabel="Flux")

        ax = axes[1]
        ax.set_axis_off()

        ax = axes[2]
        ax.imshow(
            np.sqrt(amps).T,
            aspect="auto",
            extent=[t0s.min(), t0s.max(), fmin, fmax],
            origin="lower",
            cmap=cmap,
        )
        ax.set(xlabel="Time [day]", ylabel=r"Frequency [$d^{-1}$]")

        ax = axes[3]
        freq, amplitude = amplitude_spectrum(time, flux, fmin, fmax)
        ax.plot(amplitude, freq, c="k", lw=0.7)
        ax.set(xlim=[0, None], ylim=[fmin, fmax], yticks=[], xlabel="Power")
        ax.set_xticks(ax.get_xticks()[1:])
        plt.subplots_adjust(hspace=0.02, wspace=0.02)
        plt.show()
    return t0s, amps


def sliding_window_box(time, flux, fmin, fmax, segment_size=5, plot=True, cmap="Blues"):
    t0s = np.arange(time.min(), time.max(), segment_size)

    df = 1.0 / (time.max() - time.min())
    freq = np.arange(fmin, fmax, df / 5)
    amps = np.empty((len(t0s) - 1, len(freq)))
    amps[:] = np.nan
    for i, t0 in tqdm.tqdm(enumerate(t0s[:-1]), total=len(t0s) - 1):
        m = (t0 <= time) & (time < t0s[i + 1])
        if m.sum() < 100:
            continue
        f, a = amplitude_spectrum(time[m], flux[m], freq=freq)
        amps[i] = a
    if plot:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=[10, 4],
            gridspec_kw={"height_ratios": [0.5, 1], "width_ratios": [1, 0.5]},
        )
        axes = axes.flatten()

        ax = axes[0]
        ax.plot(time, flux, c="k", lw=0.8)
        ax.set(xlim=[time.min(), time.max()], xticks=[], ylabel="Flux")

        ax = axes[1]
        ax.set_axis_off()

        ax = axes[2]
        ax.imshow(
            np.array(amps).T,
            aspect="auto",
            extent=[t0s.min(), t0s.max(), fmin, fmax],
            origin="lower",
            cmap=cmap,
            # interpolation='none'
        )
        ax.set(xlabel="Time [day]", ylabel=r"Frequency [$d^{-1}$]")

        ax = axes[3]
        freq, amplitude = amplitude_spectrum(time, flux, fmin, fmax)
        ax.plot(amplitude, freq, c="k", lw=0.7)
        ax.set(xlim=[0, None], ylim=[fmin, fmax], yticks=[], xlabel="Power")
        ax.set_xticks(ax.get_xticks()[1:])
        plt.subplots_adjust(hspace=0.02, wspace=0.02)
        plt.show()
    return t0s, amps
