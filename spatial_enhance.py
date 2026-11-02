#!/usr/bin/env python3
"""
Spatial Audio Enhancement Pipeline
1. Source separation (Demucs via torchaudio)
2. Stereo widening per stem
3. Mix with enhanced spatial characteristics
"""
import sys
import os
import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import numpy as np

def load_audio(path, target_sr=44100):
    """Load audio and resample to target sample rate."""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    # Ensure stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    return waveform, target_sr

def stereo_widen(audio, width=1.5):
    """
    Widen stereo image using Mid-Side processing.
    width > 1 = wider, width < 1 = narrower
    """
    if audio.shape[0] != 2:
        return audio

    left = audio[0]
    right = audio[1]

    # Mid-Side encoding
    mid = (left + right) / 2
    side = (left - right) / 2

    # Enhance side signal (spatial width)
    side = side * width

    # Mid-Side decoding
    new_left = mid + side
    new_right = mid - side

    # Normalize to prevent clipping
    max_val = max(new_left.abs().max(), new_right.abs().max())
    if max_val > 1.0:
        new_left = new_left / max_val
        new_right = new_right / max_val

    return torch.stack([new_left, new_right])

def add_subtle_reverb(audio, sr, decay=0.3, delay_ms=30):
    """Add subtle reverb for depth."""
    delay_samples = int(sr * delay_ms / 1000)

    # Simple delay-based reverb
    reverb = torch.zeros_like(audio)
    if delay_samples < audio.shape[1]:
        reverb[:, delay_samples:] = audio[:, :-delay_samples] * decay

    return audio + reverb

def separate_sources(waveform, sr, device='cpu'):
    """Separate audio into stems using Hybrid Demucs."""
    print("Loading Hybrid Demucs model...")
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    model.to(device)
    model.eval()

    # Model expects specific sample rate
    model_sr = bundle.sample_rate
    if sr != model_sr:
        resampler = torchaudio.transforms.Resample(sr, model_sr)
        waveform = resampler(waveform)

    # Add batch dimension
    waveform = waveform.unsqueeze(0).to(device)

    print("Separating sources...")
    with torch.no_grad():
        sources = model(waveform)

    # sources shape: [batch, sources, channels, time]
    # sources order: drums, bass, other, vocals
    sources = sources.squeeze(0)  # Remove batch dim

    return {
        'drums': sources[0],
        'bass': sources[1],
        'other': sources[2],
        'vocals': sources[3]
    }, model_sr

def process_spatial(input_path, output_path,
                    vocal_width=1.2,
                    drums_width=1.0,
                    bass_width=0.8,
                    other_width=1.8,
                    add_reverb=True):
    """
    Full spatial enhancement pipeline.
    - Vocals: slightly wider
    - Drums: centered (punch)
    - Bass: narrow (center)
    - Other (guitars, synths): very wide
    """
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load audio
    print(f"Loading {input_path}...")
    waveform, sr = load_audio(input_path)

    # Separate sources
    sources, model_sr = separate_sources(waveform, sr, device)

    # Apply different stereo width to each stem
    print("Applying spatial processing...")
    processed = {}

    widths = {
        'vocals': vocal_width,
        'drums': drums_width,
        'bass': bass_width,
        'other': other_width
    }

    for name, stem in sources.items():
        stem = stem.to('cpu')
        # Apply stereo widening
        processed[name] = stereo_widen(stem, widths[name])

        # Add subtle reverb to non-bass stems for depth
        if add_reverb and name not in ['bass', 'drums']:
            processed[name] = add_subtle_reverb(processed[name], model_sr)

    # Mix stems back together
    print("Mixing enhanced stems...")
    mixed = sum(processed.values())

    # Normalize
    max_val = mixed.abs().max()
    if max_val > 0.95:
        mixed = mixed * 0.95 / max_val

    # Save
    print(f"Saving to {output_path}...")
    torchaudio.save(output_path, mixed, model_sr)

    print(f"Done! Output: {output_path}")
    print(f"Sample rate: {model_sr}Hz, Channels: {mixed.shape[0]}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python spatial_enhance.py <input> <output>")
        print("\nThis script:")
        print("1. Separates audio into stems (vocals, drums, bass, other)")
        print("2. Applies different stereo widths to each stem")
        print("3. Adds subtle reverb for depth")
        print("4. Mixes back with enhanced spatial characteristics")
        sys.exit(1)

    process_spatial(sys.argv[1], sys.argv[2])
