#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Upscaler Audio Enhancement Pipeline
=========================================
Source Separation → Psychoacoustic Enhancement → Spatial 3D → Mastering → 96kHz/24-bit

Transforms 160kbps AAC into something that makes Hi-Res sound average.
"""

import sys
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
from spectral_deesser import SpectralDeEsser
from fatllama_deesser import FatLlamaDeesser


class PsychoacousticEnhancer:
    """Harmonic exciter, transient shaper, sub-harmonic synthesis."""

    def __init__(self, sr=44100):
        self.sr = sr
        self.nyq = sr / 2

    def harmonic_exciter(self, audio, hp_freq=3000, drive=0.3, mix=0.15):
        """Aphex-style harmonic exciter - adds missing high frequencies."""
        b, a = butter(4, hp_freq / self.nyq, btype='high')
        highs = filtfilt(b, a, audio, axis=0)

        # Soft clipping for harmonic generation
        driven = np.tanh(highs * (1 + drive * 10))

        # Even harmonics (tube-like)
        even = np.abs(driven) - 0.5

        # Blend
        harmonics = 0.6 * even + 0.4 * driven
        b2, a2 = butter(2, hp_freq / self.nyq, btype='high')
        harmonics = filtfilt(b2, a2, harmonics, axis=0)

        return (audio + harmonics * mix).astype(np.float32)

    def subharmonic_synth(self, audio, bass_freq=80, mix=0.25):
        """MaxxBass-style sub-harmonic synthesis."""
        b, a = butter(4, bass_freq / self.nyq, btype='low')
        bass = filtfilt(b, a, audio, axis=0)

        # Get envelope
        if audio.ndim == 1:
            envelope = np.abs(hilbert(bass))
        else:
            envelope = np.abs(hilbert(bass, axis=0))

        b_env, a_env = butter(2, 30 / self.nyq, btype='low')
        envelope = filtfilt(b_env, a_env, envelope, axis=0)

        # Octave down
        rectified = np.maximum(bass, 0)
        b_sub, a_sub = butter(4, (bass_freq / 2) / self.nyq, btype='low')
        sub = filtfilt(b_sub, a_sub, rectified, axis=0) * 2

        # Apply envelope
        max_env = np.max(envelope) + 1e-10
        sub = sub * (envelope / max_env)

        return (audio + sub * mix).astype(np.float32)

    def transient_shaper(self, audio, attack_gain=1.3, sustain_gain=0.9):
        """Enhance drum transients."""
        def envelope_follower(sig, attack_time, release_time):
            attack_coef = np.exp(-1 / (self.sr * attack_time))
            release_coef = np.exp(-1 / (self.sr * release_time))
            env = np.zeros_like(sig)
            env[0] = np.abs(sig[0])
            for i in range(1, len(sig)):
                current = np.abs(sig[i])
                coef = attack_coef if current > env[i-1] else release_coef
                env[i] = coef * env[i-1] + (1 - coef) * current
            return env

        if audio.ndim == 1:
            fast_env = envelope_follower(audio, 0.001, 0.02)
            slow_env = envelope_follower(audio, 0.05, 0.2)
        else:
            mono = np.mean(audio, axis=1)
            fast_env = envelope_follower(mono, 0.001, 0.02)
            slow_env = envelope_follower(mono, 0.05, 0.2)
            fast_env = np.column_stack([fast_env, fast_env])
            slow_env = np.column_stack([slow_env, slow_env])

        transient = np.maximum(fast_env - slow_env, 0)
        sustain = np.minimum(fast_env, slow_env)

        transient /= (np.max(transient) + 1e-10)
        sustain /= (np.max(sustain) + 1e-10)

        gain = 1 + transient * (attack_gain - 1) + sustain * (sustain_gain - 1)
        b, a = butter(2, 500 / self.nyq, btype='low')
        gain = filtfilt(b, a, gain, axis=0)

        return (audio * gain).astype(np.float32)

    def tube_warmth(self, audio, drive=0.2, mix=0.2):
        """Tube-style warmth and saturation."""
        k = drive * 5
        x = audio * (1 + k)
        saturated = np.where(x >= 0, 1 - np.exp(-x), -1 + np.exp(x)) * (1 + 0.1 * k)
        return (audio * (1 - mix) + saturated * mix).astype(np.float32)

    def process_stem(self, audio, stem_type):
        """Apply stem-specific enhancement."""
        if stem_type == 'vocals':
            # Harmonic exciter (gentle)
            audio = self.harmonic_exciter(audio, hp_freq=5000, drive=0.12, mix=0.06)
            # Fat Llama-style de-esser (calibrated EQ curve)
            deesser = FatLlamaDeesser(sample_rate=self.sr, strength=1.6)
            audio = deesser.process(audio)
            audio = self.tube_warmth(audio, drive=0.1, mix=0.12)
        elif stem_type == 'drums':
            audio = self.transient_shaper(audio, attack_gain=1.4, sustain_gain=0.85)
            audio = self.harmonic_exciter(audio, hp_freq=5000, drive=0.15, mix=0.08)
        elif stem_type == 'bass':
            audio = self.subharmonic_synth(audio, bass_freq=80, mix=0.22)
            audio = self.tube_warmth(audio, drive=0.15, mix=0.18)
        elif stem_type == 'other':
            audio = self.harmonic_exciter(audio, hp_freq=3000, drive=0.25, mix=0.14)
            audio = self.tube_warmth(audio, drive=0.08, mix=0.1)
        return audio


class SpatialProcessor:
    """3D spatial audio processing with room simulation."""

    def __init__(self, sr=44100):
        self.sr = sr
        self.nyq = sr / 2

        # Stem spatial config: (width, room_size, reverb_mix, depth)
        self.spatial_config = {
            'vocals': (1.15, 0.28, 0.18, 0.3),   # Center-ish, intimate
            'drums':  (1.45, 0.38, 0.22, 0.5),   # Wide, punchy room
            'bass':   (0.6, 0.05, 0.02, 0.1),    # Mono, dry, forward
            'other':  (1.7, 0.5, 0.28, 0.7),     # Very wide, lush, back
        }

    def stereo_widen(self, audio, width):
        """Mid-side stereo widening."""
        if audio.ndim == 1:
            return np.column_stack([audio, audio])

        left, right = audio[:, 0], audio[:, 1]
        mid = (left + right) / 2
        side = (left - right) / 2
        side = side * width
        return np.column_stack([mid + side, mid - side]).astype(np.float32)

    def freq_dependent_width(self, audio, bass_width=0.3, mid_width=1.0, high_width=1.5):
        """Keep bass mono, widen highs progressively."""
        if audio.ndim == 1:
            return np.column_stack([audio, audio])

        b_low, a_low = butter(4, 200 / self.nyq, btype='low')
        b_mid, a_mid = butter(4, [200 / self.nyq, 4000 / self.nyq], btype='band')
        b_high, a_high = butter(4, 4000 / self.nyq, btype='high')

        low = filtfilt(b_low, a_low, audio, axis=0)
        mid = filtfilt(b_mid, a_mid, audio, axis=0)
        high = filtfilt(b_high, a_high, audio, axis=0)

        low = self.stereo_widen(low, bass_width)
        mid = self.stereo_widen(mid, mid_width)
        high = self.stereo_widen(high, high_width)

        return (low + mid + high).astype(np.float32)

    def simple_room(self, audio, room_size=0.3, mix=0.2):
        """Simple room simulation using pedalboard."""
        import pedalboard as pb

        board = pb.Pedalboard([
            pb.Reverb(
                room_size=room_size,
                damping=0.5,
                wet_level=mix,
                dry_level=1.0 - mix,
                width=1.0
            )
        ])

        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        return board(audio.T.astype(np.float32), self.sr).T

    def process_stem(self, audio, stem_type):
        """Apply spatial processing to a stem."""
        width, room_size, reverb_mix, depth = self.spatial_config[stem_type]

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        elif audio.shape[1] != 2:
            audio = np.column_stack([audio, audio])

        # Stereo widening
        audio = self.stereo_widen(audio, width)

        # Room reverb (skip bass to keep it tight)
        if stem_type != 'bass' and reverb_mix > 0:
            audio = self.simple_room(audio, room_size, reverb_mix)

        return audio

    def mix_stems(self, stems_dict):
        """Mix all processed stems."""
        gains = {'vocals': 0.88, 'drums': 0.78, 'bass': 0.85, 'other': 0.68}

        max_len = max(len(s) for s in stems_dict.values())
        output = np.zeros((max_len, 2), dtype=np.float32)

        for stem_type, audio in stems_dict.items():
            processed = self.process_stem(audio, stem_type)
            if len(processed) < max_len:
                processed = np.pad(processed, ((0, max_len - len(processed)), (0, 0)))
            output += processed * gains.get(stem_type, 0.7)

        # Frequency-dependent width on final mix
        output = self.freq_dependent_width(output, bass_width=0.4, mid_width=1.1, high_width=1.4)

        return output


class MasteringProcessor:
    """Final mastering: loudness normalization, limiting, dithering."""

    def __init__(self, sr=44100):
        self.sr = sr

    def normalize_loudness(self, audio, target_lufs=-14.0):
        """Normalize to target LUFS."""
        import pyloudnorm as pyln

        meter = pyln.Meter(self.sr)
        current = meter.integrated_loudness(audio)

        if current > -70:  # Valid measurement
            gain_db = target_lufs - current
            gain_lin = 10 ** (gain_db / 20)
            audio = audio * gain_lin
            print(f"  Normalized: {current:.1f} → {target_lufs:.1f} LUFS")
        else:
            print(f"  Loudness measurement failed, skipping normalization")

        return audio.astype(np.float32)

    def true_peak_limiter(self, audio, threshold_dbtp=-1.0):
        """Simple limiter using pedalboard."""
        import pedalboard as pb

        board = pb.Pedalboard([
            pb.Limiter(threshold_db=threshold_dbtp, release_ms=100)
        ])

        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        return board(audio.T.astype(np.float32), self.sr).T

    def tpdf_dither(self, audio, bits=24):
        """TPDF dithering for bit depth."""
        noise1 = np.random.uniform(-1, 1, audio.shape)
        noise2 = np.random.uniform(-1, 1, audio.shape)
        tpdf = (noise1 + noise2) / 2
        lsb = 1 / (2 ** (bits - 1))
        return (audio + tpdf * lsb).astype(np.float32)

    def process(self, audio, target_lufs=-14.0):
        """Complete mastering chain."""
        audio = self.normalize_loudness(audio, target_lufs)
        audio = self.true_peak_limiter(audio, threshold_dbtp=-1.0)
        audio = self.tpdf_dither(audio, bits=24)
        audio = np.clip(audio, -1.0, 1.0)
        return audio


def separate_stems(input_path):
    """Source separation using demucs-mlx."""
    from demucs_mlx import Separator

    separator = Separator(model="htdemucs")
    origin, stems = separator.separate_audio_file(input_path)

    result = {}
    for name in ['vocals', 'drums', 'bass', 'other']:
        audio = np.array(stems[name])
        # Ensure (samples, channels) format
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        elif audio.shape[0] == 2:
            audio = audio.T
        result[name] = audio

    return result, 44100


def resample_audio(audio, sr_in, sr_out):
    """High-quality resampling."""
    if sr_in == sr_out:
        return audio
    samples = int(len(audio) * sr_out / sr_in)
    return signal.resample(audio, samples, axis=0).astype(np.float32)


def process_track(input_path: str, output_path: str, target_sr: int = 96000):
    """Audio Upscaler enhancement pipeline."""

    print(f"\n{'='*60}")
    print(f"Audio Upscaler Enhancement: {Path(input_path).name}")
    print('='*60)

    print("\n[1/5] Source separation (demucs-mlx)...")
    stems, sr = separate_stems(input_path)

    print("[2/5] Psychoacoustic enhancement...")
    enhancer = PsychoacousticEnhancer(sr=sr)
    for stem_type in stems:
        stems[stem_type] = enhancer.process_stem(stems[stem_type], stem_type)

    print("[3/5] Spatial 3D processing...")
    spatial = SpatialProcessor(sr=sr)
    mix = spatial.mix_stems(stems)

    print("[4/5] Final mastering...")
    master = MasteringProcessor(sr=sr)
    mix = master.process(mix, target_lufs=-14.0)

    print(f"[5/5] Upsampling to {target_sr}Hz...")
    final = resample_audio(mix, sr, target_sr)

    sf.write(output_path, final, target_sr, subtype='PCM_24')
    print(f"\n✓ Saved: {output_path}")
    print(f"  Format: {target_sr}Hz / 24-bit / Stereo")


def batch_process(input_dir: str, output_dir: str, target_sr: int = 96000):
    """Process all tracks in directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {'.mp3', '.m4a', '.aac', '.wav', '.flac', '.ogg'}
    files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in extensions])

    print(f"\nFound {len(files)} tracks to process")
    print(f"Output: {output_path}")

    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]", end="")
        out_file = output_path / (f.stem + '_Audio Upscaler.wav')

        if out_file.exists():
            print(f" Skipping {f.name} (exists)")
            continue

        try:
            process_track(str(f), str(out_file), target_sr)
        except Exception as e:
            print(f" ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Complete! Processed {len(files)} tracks.")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("""
╔═══════════════════════════════════════════════════════════╗
║          Audio Upscaler Audio Enhancement Pipeline          ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  Transforms compressed audio into stratospheric quality   ║
║                                                           ║
║  Pipeline:                                                ║
║    1. Source separation (demucs-mlx) - isolate stems      ║
║    2. Psychoacoustic enhancement per stem:                ║
║       • Vocals: presence + air + tube warmth              ║
║       • Drums: transient shaping + crack                  ║
║       • Bass: sub-harmonic synthesis + warmth             ║
║       • Other: harmonic exciter + presence                ║
║    3. Spatial 3D processing:                              ║
║       • Per-stem stereo width                             ║
║       • Frequency-dependent width (mono bass, wide highs) ║
║       • Room simulation + depth                           ║
║    4. Final mastering:                                    ║
║       • LUFS normalization (-14 LUFS)                     ║
║       • True peak limiting (-1 dBTP)                      ║
║       • TPDF dithering                                    ║
║    5. Upsample to 96kHz/24-bit                            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

Usage:
  python stratospheric_enhance.py <input> <output>
  python stratospheric_enhance.py --batch <input_dir> <output_dir>

Estimated: ~10-15 sec per track on Apple Silicon
""")
        sys.exit(1)

    if sys.argv[1] == '--batch':
        target = int(sys.argv[4]) if len(sys.argv) > 4 else 96000
        batch_process(sys.argv[2], sys.argv[3], target)
    else:
        target = int(sys.argv[3]) if len(sys.argv) > 3 else 96000
        process_track(sys.argv[1], sys.argv[2], target)
