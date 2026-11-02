#!/usr/bin/env python3
"""
ultimate_enhance.py - Ultimate Audio Enhancement Pipeline
Combines: Source Separation + Spatial FX + Harmonic Enhancement + Mastering
"""

import sys
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal
from scipy.fft import rfft, irfft, rfftfreq


def process_track(input_path: str, output_path: str, target_sr: int = 96000):
    """Ultimate pipeline: Separation -> Enhancement -> Spatial -> Master"""

    print(f"[1/5] Source separation: {input_path}")
    stems = separate_stems(input_path)

    print("[2/5] Per-stem enhancement (harmonics, EQ)...")
    enhanced_stems = enhance_stems(stems)

    print("[3/5] Spatial processing (stereo width, depth)...")
    spatial_mix = apply_spatial_fx(enhanced_stems)

    print("[4/5] Mastering (compression, limiting, EQ)...")
    mastered = master_audio(spatial_mix, stems['sample_rate'])

    print(f"[5/5] Upsampling to {target_sr}Hz...")
    final = resample_audio(mastered, stems['sample_rate'], target_sr)

    sf.write(output_path, final.T, target_sr, subtype='PCM_24')
    print(f"\nDone: {output_path}")
    print(f"  Sample rate: {target_sr}Hz, Bit depth: 24-bit")


def separate_stems(input_path: str) -> dict:
    """Use demucs-mlx for source separation"""
    from demucs_mlx import Separator

    separator = Separator(model="htdemucs")
    origin, stems = separator.separate_audio_file(input_path)

    result = {
        'vocals': np.array(stems['vocals']),
        'drums': np.array(stems['drums']),
        'bass': np.array(stems['bass']),
        'other': np.array(stems['other']),
        'sample_rate': 44100
    }
    return result


def enhance_stems(stems: dict) -> dict:
    """Apply per-stem enhancement: harmonics, EQ, dynamics"""
    sr = stems['sample_rate']
    enhanced = {}

    for name, audio in stems.items():
        if name == 'sample_rate':
            continue

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        elif audio.shape[0] != 2 and audio.shape[1] == 2:
            audio = audio.T

        if name == 'vocals':
            # Vocal enhancement: presence boost (2-5kHz), air (10kHz+)
            audio = harmonic_exciter(audio, sr, amount=0.15)
            audio = eq_boost(audio, sr, freq=3500, gain_db=2, q=1.5)
            audio = eq_boost(audio, sr, freq=12000, gain_db=1.5, q=0.7)

        elif name == 'drums':
            # Drums: transient enhancement, punch
            audio = transient_shaper(audio, sr, attack_gain=1.3, sustain_gain=0.9)
            audio = eq_boost(audio, sr, freq=100, gain_db=2, q=1.0)  # Kick punch
            audio = eq_boost(audio, sr, freq=5000, gain_db=1.5, q=1.0)  # Snare crack

        elif name == 'bass':
            # Bass: sub-harmonic enhancement, tightness
            audio = sub_harmonic_synth(audio, sr, amount=0.2)
            audio = eq_boost(audio, sr, freq=80, gain_db=2, q=1.2)

        elif name == 'other':
            # Other (guitars, synths): harmonic richness, presence
            audio = harmonic_exciter(audio, sr, amount=0.2)
            audio = eq_boost(audio, sr, freq=2500, gain_db=1.5, q=1.0)

        enhanced[name] = audio

    enhanced['sample_rate'] = sr
    return enhanced


def harmonic_exciter(audio: np.ndarray, sr: int, amount: float = 0.15) -> np.ndarray:
    """Add harmonics to enhance perceived clarity and brightness"""
    # High-pass filter to get high frequencies
    nyq = sr / 2
    high_freq = 3000 / nyq
    b, a = signal.butter(2, high_freq, btype='high')

    excited = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        # Extract highs
        highs = signal.filtfilt(b, a, audio[ch])
        # Soft clip to generate harmonics
        harmonics = np.tanh(highs * 3) * amount
        excited[ch] = audio[ch] + harmonics

    return excited


def transient_shaper(audio: np.ndarray, sr: int,
                     attack_gain: float = 1.2, sustain_gain: float = 1.0) -> np.ndarray:
    """Shape transients (attack) vs sustain"""
    # Envelope follower
    window_ms = 10
    window_samples = int(sr * window_ms / 1000)

    shaped = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        # Get envelope
        envelope = np.abs(audio[ch])
        envelope = signal.medfilt(envelope, window_samples | 1)  # Ensure odd

        # Detect transients (fast rise in envelope)
        envelope_diff = np.diff(envelope, prepend=envelope[0])
        transient_mask = envelope_diff > 0

        # Apply different gains
        gain = np.where(transient_mask, attack_gain, sustain_gain)
        # Smooth the gain changes
        gain = signal.medfilt(gain.astype(np.float64), 101)

        shaped[ch] = audio[ch] * gain

    return shaped


def sub_harmonic_synth(audio: np.ndarray, sr: int, amount: float = 0.2) -> np.ndarray:
    """Generate sub-harmonics (one octave below) for bass enhancement"""
    # Low-pass to get bass content
    nyq = sr / 2
    low_freq = 150 / nyq
    b, a = signal.butter(4, low_freq, btype='low')

    enhanced = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        # Get bass
        bass = signal.filtfilt(b, a, audio[ch])
        # Generate sub-harmonic by frequency division (approximate)
        # Use envelope follower + low frequency oscillator
        envelope = np.abs(signal.hilbert(bass))
        sub = np.sin(np.cumsum(np.sign(bass) * 0.5)) * envelope * amount
        enhanced[ch] = audio[ch] + sub

    return enhanced


def eq_boost(audio: np.ndarray, sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    """Parametric EQ boost/cut at specific frequency"""
    # Design peaking EQ filter
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1, a1/a0, a2/a0])

    result = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        result[ch] = signal.filtfilt(b, a, audio[ch])

    return result


def apply_spatial_fx(stems: dict) -> np.ndarray:
    """Apply spatial effects to each stem, then remix"""
    import pedalboard as pb

    sr = stems['sample_rate']

    # Spatial config: (stereo_width, reverb_room_size, reverb_wet)
    spatial_config = {
        'vocals': (1.1, 0.25, 0.18),   # Slightly wide, medium reverb
        'drums':  (1.4, 0.35, 0.22),   # Wide, room sound
        'bass':   (0.7, 0.08, 0.03),   # Narrow, very dry
        'other':  (1.6, 0.45, 0.28),   # Very wide, lush reverb
    }

    processed_stems = {}

    for stem_name, audio in stems.items():
        if stem_name == 'sample_rate':
            continue

        width, room, wet = spatial_config[stem_name]

        # Stereo widening
        audio = stereo_widen(audio, width)

        # Reverb (skip bass)
        if stem_name != 'bass':
            board = pb.Pedalboard([
                pb.Reverb(room_size=room, wet_level=wet, dry_level=1.0 - wet)
            ])
            audio = board(audio.astype(np.float32), sr)

        processed_stems[stem_name] = audio

    # Mix stems
    mix = sum(processed_stems.values())

    # Normalize
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix * (0.95 / peak)

    return mix


def stereo_widen(audio: np.ndarray, width: float) -> np.ndarray:
    """Mid-side stereo widening"""
    left, right = audio[0], audio[1]
    mid = (left + right) / 2
    side = (left - right) / 2
    side = side * width
    return np.stack([mid + side, mid - side])


def master_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Final mastering: compression, EQ, limiting"""
    import pedalboard as pb

    # Mastering chain
    board = pb.Pedalboard([
        # Gentle compression
        pb.Compressor(
            threshold_db=-12,
            ratio=2.5,
            attack_ms=10,
            release_ms=100
        ),
        # High shelf for air
        pb.HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.0),
        # Low shelf for warmth
        pb.LowShelfFilter(cutoff_frequency_hz=100, gain_db=0.5),
        # Final limiter
        pb.Limiter(threshold_db=-1.0, release_ms=100)
    ])

    mastered = board(audio.astype(np.float32), sr)

    return mastered


def resample_audio(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High-quality resampling"""
    if sr_in == sr_out:
        return audio

    num_samples = int(audio.shape[1] * sr_out / sr_in)
    resampled = signal.resample(audio, num_samples, axis=1)
    return resampled


def process_batch(input_dir: str, output_dir: str, target_sr: int = 96000):
    """Process all audio files in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {'.mp3', '.m4a', '.aac', '.wav', '.flac', '.ogg'}
    files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in extensions])

    print(f"Found {len(files)} files to process")

    for i, f in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(files)}] {f.name}")
        out_file = output_path / (f.stem + '_ultimate.wav')

        if out_file.exists():
            print("  Skipping (exists)")
            continue

        try:
            process_track(str(f), str(out_file), target_sr)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Ultimate Audio Enhancement Pipeline")
        print("=" * 40)
        print("\nUsage:")
        print("  python ultimate_enhance.py <input> <output>")
        print("  python ultimate_enhance.py --batch <in_dir> <out_dir>")
        print("\nPipeline:")
        print("  1. Source separation (demucs-mlx)")
        print("  2. Per-stem enhancement:")
        print("     - Vocals: presence + air EQ, harmonic exciter")
        print("     - Drums: transient shaping, punch EQ")
        print("     - Bass: sub-harmonic synth, tightness")
        print("     - Other: harmonic exciter, presence")
        print("  3. Spatial processing (stereo width + reverb)")
        print("  4. Mastering (compression, EQ, limiting)")
        print("  5. Upsample to 96kHz/24-bit")
        sys.exit(1)

    if sys.argv[1] == '--batch':
        process_batch(sys.argv[2], sys.argv[3])
    else:
        process_track(sys.argv[1], sys.argv[2])
