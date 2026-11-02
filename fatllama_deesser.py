#!/usr/bin/env python3
"""
Fat Llama De-esser Module
=========================

This module provides de-essing that matches Fat Llama's smooth sibilance profile.

Key Findings from Analysis:
- Our pipeline's spectral centroid in sibilance band: 7229 Hz
- Fat Llama's spectral centroid: 5962 Hz
- Root cause: Our neural upscaling adds harmonics in 6-12kHz range
- Solution: Apply tilted high-shelf cut with peak at 9-10kHz

Usage in Pipeline:
------------------
    from fatllama_deesser import deess_like_fatllama

    # In your pipeline, after all processing:
    audio = deess_like_fatllama(audio, sr, strength=1.6)

    # Or as file-to-file:
    from fatllama_deesser import deess_file
    deess_file('input.wav', 'output.wav', strength=1.6)

Recommended Strength:
- 1.0 = Mild de-essing (440 Hz centroid error)
- 1.4 = Moderate de-essing (124 Hz error)
- 1.6 = Match Fat Llama (25 Hz error) <-- RECOMMENDED
- 2.0 = Aggressive (may sound dull)
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import gaussian_filter1d

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


class FatLlamaDeesser:
    """
    FFT-based de-esser calibrated to Fat Llama's spectral profile.

    The EQ curve applies:
    - Gentle rolloff starting at 2kHz
    - Steeper rolloff from 4-6kHz
    - Maximum cut at 9-11kHz (main sibilance zone)
    - Gradual return above 14kHz (preserve some air)
    """

    def __init__(self, sample_rate, fft_size=4096, strength=1.6):
        """
        Parameters:
        -----------
        sample_rate : int
            Audio sample rate (e.g., 44100, 48000, 96000)
        fft_size : int
            FFT window size. Larger = more frequency resolution, more latency.
        strength : float
            De-essing intensity.
            1.0 = baseline calibration
            1.6 = recommended to match Fat Llama
        """
        self.sr = sample_rate
        self.fft_size = fft_size
        self.hop_size = fft_size // 4  # 75% overlap
        self.strength = strength
        self.eq_curve = self._build_eq_curve()

    def _build_eq_curve(self):
        """Build the de-essing EQ curve."""
        freqs = fftfreq(self.fft_size, 1/self.sr)[:self.fft_size // 2]
        gains_db = np.zeros_like(freqs)

        for i, f in enumerate(freqs):
            if f < 2000:
                gains_db[i] = 0
            elif f < 4000:
                # Gentle rolloff
                gains_db[i] = -1.5 * (f - 2000) / 2000
            elif f < 6000:
                # Steeper rolloff
                gains_db[i] = -1.5 - 3.0 * (f - 4000) / 2000
            elif f < 9000:
                # Main de-essing zone
                gains_db[i] = -4.5 - 4.0 * (f - 6000) / 3000
            elif f < 11000:
                # Peak sibilance zone - maximum cut
                gains_db[i] = -8.5 - 1.5 * (f - 9000) / 2000
            elif f < 14000:
                # Gradual return
                gains_db[i] = -10.0 + 2.0 * (f - 11000) / 3000
            else:
                # Air region - moderate cut maintained
                gains_db[i] = -8.0 + 4.0 * (f - 14000) / (self.sr/2 - 14000)

        # Apply strength scaling
        gains_db *= self.strength

        # Smooth to avoid artifacts
        gains_db = gaussian_filter1d(gains_db, sigma=5)

        # Convert to linear and create symmetric curve
        gains_linear = 10**(gains_db / 20)
        return np.concatenate([gains_linear, gains_linear[::-1]])

    def process(self, audio):
        """
        Apply de-essing to audio.

        Parameters:
        -----------
        audio : numpy array
            Audio data. Shape can be:
            - (N,) for mono
            - (N, 2) for stereo

        Returns:
        --------
        numpy array : De-essed audio, same shape as input
        """
        # Handle stereo recursively
        if len(audio.shape) > 1 and audio.shape[1] == 2:
            left = self.process(audio[:, 0])
            right = self.process(audio[:, 1])
            return np.column_stack([left, right])

        # Pad to multiple of hop_size
        pad_len = (self.hop_size - len(audio) % self.hop_size) % self.hop_size
        audio_padded = np.pad(audio, (0, pad_len))

        output = np.zeros(len(audio_padded) + self.fft_size)
        window = np.hanning(self.fft_size)

        num_frames = (len(audio_padded) - self.fft_size) // self.hop_size + 1

        for i in range(num_frames):
            start = i * self.hop_size
            frame = audio_padded[start:start + self.fft_size] * window

            # FFT -> apply EQ -> IFFT
            spectrum = fft(frame)
            filtered = spectrum * self.eq_curve
            filtered_frame = np.real(ifft(filtered))

            # Overlap-add
            output[start:start + self.fft_size] += filtered_frame * window

        # Normalize for overlap-add (Hanning with 75% overlap) and trim
        return output[:len(audio)] * (2 / 3)


def deess_like_fatllama(audio, sr, strength=1.6):
    """
    Apply Fat Llama-style de-essing to audio array.

    This is the main function to use in your pipeline.

    Parameters:
    -----------
    audio : numpy array
        Audio data (mono or stereo)
    sr : int
        Sample rate
    strength : float
        De-essing strength. Default 1.6 matches Fat Llama.

    Returns:
    --------
    numpy array : De-essed audio

    Example:
    --------
        import soundfile as sf
        from fatllama_deesser import deess_like_fatllama

        audio, sr = sf.read('input.wav')
        audio = deess_like_fatllama(audio, sr, strength=1.6)
        sf.write('output.wav', audio, sr)
    """
    deesser = FatLlamaDeesser(sr, strength=strength)
    return deesser.process(audio)


def deess_file(input_path, output_path, strength=1.6):
    """
    Apply de-essing to an audio file.

    Parameters:
    -----------
    input_path : str
        Path to input audio file
    output_path : str
        Path to output audio file
    strength : float
        De-essing strength. Default 1.6 matches Fat Llama.
    """
    if not HAS_SOUNDFILE:
        raise ImportError("soundfile is required for file I/O. Install with: pip install soundfile")

    audio, sr = sf.read(input_path)
    print(f"De-essing: {input_path}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio)/sr:.1f}s")
    print(f"  Strength: {strength}")

    output = deess_like_fatllama(audio, sr, strength)

    sf.write(output_path, output, sr)
    print(f"  Saved: {output_path}")


# EQ curve data for documentation/visualization
EQ_CURVE_POINTS = """
# Fat Llama De-esser EQ Curve (at strength=1.0)
# Multiply gains by strength factor
Freq (Hz)    Gain (dB)
-----------------------
20           0
2000         0
4000        -1.5
6000        -4.5
9000        -8.5
11000      -10.0
14000       -8.0
20000       -4.0
"""


if __name__ == '__main__':
    # Demo usage
    print("Fat Llama De-esser Module")
    print("=" * 40)
    print(EQ_CURVE_POINTS)

    if HAS_SOUNDFILE:
        import sys
        if len(sys.argv) >= 3:
            input_file = sys.argv[1]
            output_file = sys.argv[2]
            strength = float(sys.argv[3]) if len(sys.argv) >= 4 else 1.6
            deess_file(input_file, output_file, strength)
        else:
            print("\nUsage: python fatllama_deesser.py input.wav output.wav [strength]")
            print("  strength: 1.0-2.0, default 1.6")
    else:
        print("\nInstall soundfile for file I/O: pip install soundfile")
