#!/usr/bin/env python3
"""
spatial_upscale.py - SoundCloud to Hi-Res Spatial Audio
Pipeline: Source Separation (demucs-mlx) -> Spatial FX (pedalboard) -> Upsample
"""

import sys
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal


def process_track(input_path: str, output_path: str, target_sr: int = 96000):
    """Full pipeline: Separation -> Spatial FX -> Upsample"""

    print(f"[1/3] Source separation: {input_path}")
    stems = separate_stems(input_path)

    print("[2/3] Spatial enhancement...")
    spatial_mix = apply_spatial_fx(stems)

    print(f"[3/3] Resampling to {target_sr}Hz...")
    final = resample_audio(spatial_mix, stems['sample_rate'], target_sr)

    sf.write(output_path, final.T, target_sr, subtype='PCM_24')
    print(f"Done: {output_path}")
    print(f"  Sample rate: {target_sr}Hz, Bit depth: 24-bit")


def separate_stems(input_path: str) -> dict:
    """Use demucs-mlx for source separation (native Apple Silicon)"""
    from demucs_mlx import Separator

    separator = Separator(model="htdemucs")
    origin, stems = separator.separate_audio_file(input_path)

    # Convert MLX arrays to numpy
    result = {
        'vocals': np.array(stems['vocals']),
        'drums': np.array(stems['drums']),
        'bass': np.array(stems['bass']),
        'other': np.array(stems['other']),
        'sample_rate': 44100  # demucs outputs at 44.1kHz
    }
    return result


def apply_spatial_fx(stems: dict) -> np.ndarray:
    """Apply spatial effects to each stem, then remix"""
    import pedalboard as pb

    sr = stems['sample_rate']

    # Spatial config: (stereo_width, reverb_room_size, reverb_wet)
    # Width > 1 = wider stereo, < 1 = narrower
    spatial_config = {
        'vocals': (1.0, 0.2, 0.15),   # Center, subtle reverb
        'drums':  (1.3, 0.3, 0.20),   # Wide, some room
        'bass':   (0.8, 0.1, 0.05),   # Narrow, dry (keeps punch)
        'other':  (1.5, 0.4, 0.25),   # Widest, more reverb (guitars, synths)
    }

    processed_stems = {}

    for stem_name, audio in stems.items():
        if stem_name == 'sample_rate':
            continue

        width, room, wet = spatial_config[stem_name]

        # Ensure stereo (2, samples)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        elif audio.shape[0] != 2:
            # Transpose if needed (samples, channels) -> (channels, samples)
            if audio.shape[1] == 2:
                audio = audio.T

        # 1. Stereo widening via mid-side processing
        audio = stereo_widen(audio, width)

        # 2. Reverb for depth (skip bass to keep it tight)
        if stem_name != 'bass':
            board = pb.Pedalboard([
                pb.Reverb(room_size=room, wet_level=wet, dry_level=1.0 - wet)
            ])
            audio = board(audio.astype(np.float32), sr)

        processed_stems[stem_name] = audio

    # Mix all stems together
    mix = sum(processed_stems.values())

    # Normalize to prevent clipping
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix * (0.95 / peak)

    return mix


def stereo_widen(audio: np.ndarray, width: float) -> np.ndarray:
    """
    Mid-side stereo widening
    width=1.0 is original, >1 is wider, <1 is narrower
    """
    left, right = audio[0], audio[1]

    # Convert to mid-side
    mid = (left + right) / 2
    side = (left - right) / 2

    # Scale side channel (width)
    side = side * width

    # Convert back to L/R
    new_left = mid + side
    new_right = mid - side

    return np.stack([new_left, new_right])


def resample_audio(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample audio using scipy"""
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
        print(f"[{i}/{len(files)}] Processing: {f.name}")
        out_file = output_path / (f.stem + '_spatial.wav')

        if out_file.exists():
            print(f"  Skipping (already exists)")
            continue

        try:
            process_track(str(f), str(out_file), target_sr)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Spatial Audio Enhancement Pipeline")
        print("=" * 40)
        print("\nUsage:")
        print("  python spatial_upscale.py <input.mp3> <output.wav>")
        print("  python spatial_upscale.py --batch <input_dir/> <output_dir/>")
        print("\nWhat it does:")
        print("  1. Separates audio into stems (vocals, drums, bass, other)")
        print("  2. Applies different stereo widths per stem (DSEE Extreme style)")
        print("  3. Adds reverb for depth (except bass)")
        print("  4. Resamples to 96kHz/24-bit")
        print("\nEstimate: ~5-10 sec per 3-min track on Apple Silicon")
        sys.exit(1)

    if sys.argv[1] == '--batch':
        target = int(sys.argv[4]) if len(sys.argv) > 4 else 96000
        process_batch(sys.argv[2], sys.argv[3], target)
    else:
        target = int(sys.argv[3]) if len(sys.argv) > 3 else 96000
        process_track(sys.argv[1], sys.argv[2], target)
