![](docs/img.png)

# Periodogram
Simple and tested periodogram normalizations for astronomical time-series

## Installation
```bash
pip install periodogram
```

## Usage

There are three types of normalizations available. The amplitude spectrum is normalized such that a sine wave of amplitude 1 will have a peak amplitude of 1. The power spectrum will have 1^2, and the power spectral density will have 1^2 * Tobs
```python
from periodogram import amplitude_spectrum, psd, power_spectrum

frequency, amplitude = amplitude_spectrum(time, flux)
```

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3629933.svg)](https://doi.org/10.5281/zenodo.3629933) -->