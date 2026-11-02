"""
Spectral De-Esser - Professional FFT-based de-essing
Works in frequency domain, preserving phase relationships.
"""

import numpy as np
from scipy.signal import get_window
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, Optional


class SpectralDeEsser:
    """
    Professional-grade spectral de-esser using STFT processing.

    Works by detecting spectral peaks in the sibilance range that exceed
    the surrounding spectral envelope, then applying gentle per-bin
    attenuation. Phase-coherent, no filter artifacts.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        fft_size: int = 2048,
        hop_size: int = 512,
        freq_low: float = 4000,
        freq_high: float = 9000,
        sensitivity: float = 0.5,
        strength: float = 0.7,
        smoothing: float = 0.3,
        envelope_width: float = 500,
    ):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.sensitivity = sensitivity
        self.strength = strength
        self.smoothing = smoothing
        self.envelope_width = envelope_width

        self.freq_bins = np.fft.rfftfreq(fft_size, 1/sample_rate)
        self.bin_low = np.searchsorted(self.freq_bins, freq_low)
        self.bin_high = np.searchsorted(self.freq_bins, freq_high)

        bin_resolution = sample_rate / fft_size
        self.envelope_bins = max(1, int(envelope_width / bin_resolution))

        self.window = np.sqrt(get_window('hann', fft_size, fftbins=True))
        self._prev_gain = None

    def _estimate_spectral_envelope(self, magnitude: np.ndarray) -> np.ndarray:
        return gaussian_filter1d(magnitude, sigma=self.envelope_bins)

    def _detect_sibilance(self, magnitude: np.ndarray) -> np.ndarray:
        envelope = self._estimate_spectral_envelope(magnitude)
        safe_envelope = np.maximum(envelope, 1e-10)
        ratio = magnitude / safe_envelope

        sib_mask = np.zeros_like(magnitude)
        sib_mask[self.bin_low:self.bin_high] = 1.0

        taper_bins = max(1, int(self.envelope_bins / 2))
        if self.bin_low > taper_bins:
            taper = np.linspace(0, 1, taper_bins)
            sib_mask[self.bin_low - taper_bins:self.bin_low] = taper
        if self.bin_high + taper_bins < len(sib_mask):
            taper = np.linspace(1, 0, taper_bins)
            sib_mask[self.bin_high:self.bin_high + taper_bins] = taper

        threshold = 2.0 - (0.9 * self.sensitivity)
        excess = np.maximum(0, (ratio - threshold) / (2.0 - threshold))
        excess = np.minimum(1.0, excess)

        return excess * sib_mask

    def _compute_gain(self, sibilance: np.ndarray) -> np.ndarray:
        max_reduction_db = self.strength * 12
        min_gain = 10 ** (-max_reduction_db / 20)
        gain = 1 - sibilance * (1 - min_gain) * (1 - np.cos(sibilance * np.pi / 2)) / 2
        return gain

    def _apply_temporal_smoothing(self, gain: np.ndarray) -> np.ndarray:
        if self._prev_gain is None:
            self._prev_gain = np.ones_like(gain)

        attack_coef = 0.8
        release_coef = 0.3 + 0.5 * self.smoothing

        smoothed = np.where(
            gain < self._prev_gain,
            attack_coef * gain + (1 - attack_coef) * self._prev_gain,
            release_coef * gain + (1 - release_coef) * self._prev_gain
        )

        self._prev_gain = smoothed.copy()
        return smoothed

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        windowed = frame * self.window
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        sibilance = self._detect_sibilance(magnitude)
        gain = self._compute_gain(sibilance)
        gain = self._apply_temporal_smoothing(gain)

        new_magnitude = magnitude * gain
        new_spectrum = new_magnitude * np.exp(1j * phase)
        processed = np.fft.irfft(new_spectrum, n=self.fft_size)
        processed = processed * self.window

        return processed, gain

    def process(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
            squeeze_output = True
        else:
            squeeze_output = False

        n_samples, n_channels = audio.shape
        pad_size = self.fft_size - self.hop_size
        audio_padded = np.pad(audio, ((pad_size, pad_size), (0, 0)), mode='reflect')

        output = np.zeros_like(audio_padded)
        norm = np.zeros(len(audio_padded))

        for ch in range(n_channels):
            self._prev_gain = None

            for i in range(0, len(audio_padded) - self.fft_size + 1, self.hop_size):
                frame = audio_padded[i:i + self.fft_size, ch]
                processed_frame, _ = self.process_frame(frame)
                output[i:i + self.fft_size, ch] += processed_frame

                if ch == 0:
                    norm[i:i + self.fft_size] += self.window ** 2

        norm = np.maximum(norm, 1e-10)
        output = output / norm[:, np.newaxis]
        output = output[pad_size:pad_size + n_samples]

        if squeeze_output:
            output = output.squeeze()

        return output.astype(np.float32)


def deess_vocals(audio: np.ndarray, sample_rate: int, preset: str = "moderate") -> np.ndarray:
    """
    De-ess vocals with preset configurations.

    Presets: gentle, moderate, aggressive, female, male
    """
    presets = {
        "gentle": {
            "freq_low": 4000, "freq_high": 8000,
            "sensitivity": 0.3, "strength": 0.4, "smoothing": 0.4
        },
        "moderate": {
            "freq_low": 4500, "freq_high": 9000,
            "sensitivity": 0.5, "strength": 0.6, "smoothing": 0.3
        },
        "aggressive": {
            "freq_low": 4000, "freq_high": 10000,
            "sensitivity": 0.7, "strength": 0.8, "smoothing": 0.2
        },
        "female": {
            "freq_low": 5000, "freq_high": 10000,
            "sensitivity": 0.5, "strength": 0.6, "smoothing": 0.3
        },
        "male": {
            "freq_low": 3500, "freq_high": 8000,
            "sensitivity": 0.5, "strength": 0.6, "smoothing": 0.3
        },
    }

    params = presets.get(preset, presets["moderate"])
    de_esser = SpectralDeEsser(sample_rate=sample_rate, **params)
    return de_esser.process(audio)
