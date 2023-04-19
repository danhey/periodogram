import numpy as np


def dft_phase(x: np.array, y: np.array, freq: float) -> float:
    """Calculates the phase from the discrete fourier transform. Useful for some things.

    Args:
        x (np.array): Input time
        y (np.array): Input flux
        freq (float): Single frequency at which to calculate phase

    Returns:
        float: Output phase
    """
    expo = 2.0 * np.pi * freq * x
    ft_real = np.sum(y * np.cos(expo))
    ft_imag = np.sum(y * np.sin(expo))
    return np.arctan2(ft_imag, ft_real)
