#!/usr/bin/env python3
"""
A/B Comparison Tool - Visualize the difference between original and enhanced audio.
Generates spectrograms, frequency response, and stereo width plots.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import sys
import numpy as np
import soundfile as sf
from scipy.signal import stft, resample
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt


def load_and_match(path1, path2):
    """Load two audio files and resample to match."""
    audio1, sr1 = sf.read(path1)
    audio2, sr2 = sf.read(path2)

    # Resample to match
    if sr1 != sr2:
        if sr1 > sr2:
            audio2 = resample(audio2, int(len(audio2) * sr1 / sr2), axis=0)
            sr = sr1
        else:
            audio1 = resample(audio1, int(len(audio1) * sr2 / sr1), axis=0)
            sr = sr2
    else:
        sr = sr1

    return audio1, audio2, sr


def plot_spectrogram_comparison(original, enhanced, sr, output_path):
    """Side-by-side spectrogram comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for ax, audio, title in [(axes[0], original, 'Original (Compressed)'),
                              (axes[1], enhanced, 'Audio Upscaler Enhanced')]:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        f, t, Zxx = stft(audio, sr, nperseg=2048, noverlap=1536)
        magnitude = np.log10(np.abs(Zxx) + 1e-10)

        im = ax.pcolormesh(t, f / 1000, magnitude, shading='gouraud',
                           cmap='magma', vmin=-4, vmax=1)
        ax.set_ylabel('Frequency [kHz]')
        ax.set_xlabel('Time [s]')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, min(sr/2000, 48))

        # Mark where compression cuts off (~16-18kHz for 160kbps)
        if 'Original' in title:
            ax.axhline(y=16, color='red', linestyle='--', alpha=0.7, label='Compression cutoff (~16kHz)')
            ax.legend(loc='upper right')

    plt.colorbar(im, ax=axes, label='Magnitude [dB]', shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_frequency_response(original, enhanced, sr, output_path):
    """Frequency response comparison showing high-frequency extension."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for audio, label, color in [(original, 'Original', 'red'),
                                 (enhanced, 'Enhanced', 'blue')]:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/sr)

        # Smooth for visibility
        spectrum_smooth = uniform_filter1d(spectrum, size=2000)
        spectrum_db = 20 * np.log10(spectrum_smooth + 1e-10)

        ax.plot(freqs/1000, spectrum_db, label=label, color=color, alpha=0.8, linewidth=1.5)

    ax.axvline(x=16, color='gray', linestyle='--', alpha=0.5, label='160kbps cutoff')
    ax.axvline(x=20, color='green', linestyle=':', alpha=0.5, label='Human hearing limit')

    ax.set_xscale('log')
    ax.set_xlim(0.02, min(sr/2000, 48))
    ax.set_ylim(-80, 20)
    ax.set_xlabel('Frequency [kHz]', fontsize=12)
    ax.set_ylabel('Magnitude [dB]', fontsize=12)
    ax.set_title('Frequency Response - High Frequency Extension', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_stereo_width(original, enhanced, sr, output_path):
    """Stereo width comparison (Lissajous / Goniometer style)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, audio, title in [(axes[0], original, 'Original'),
                              (axes[1], enhanced, 'Enhanced (Wider)')]:
        if audio.ndim == 1:
            ax.text(0.5, 0.5, 'Mono audio', ha='center', va='center')
            ax.set_title(title)
            continue

        # Subsample for plotting
        step = max(1, len(audio) // 50000)
        left = audio[::step, 0] if audio.shape[1] >= 1 else audio[::step]
        right = audio[::step, 1] if audio.shape[1] >= 2 else left

        # Mid-Side representation
        mid = (left + right) / 2
        side = (left - right) / 2

        # Color by time
        colors = np.linspace(0, 1, len(mid))

        ax.scatter(mid, side, c=colors, cmap='viridis', alpha=0.1, s=1)
        ax.set_xlabel('Mid (L+R)')
        ax.set_ylabel('Side (L-R)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)

        # Draw width indicator
        width = np.std(side) / (np.std(mid) + 1e-10)
        ax.text(0.02, 0.98, f'Width ratio: {width:.2f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top')

    plt.suptitle('Stereo Width Comparison (Mid-Side)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def generate_report(original_path, enhanced_path, output_dir='.'):
    """Generate full A/B comparison report."""
    print(f"\nGenerating A/B comparison report...")
    print(f"  Original:  {original_path}")
    print(f"  Enhanced:  {enhanced_path}")

    original, enhanced, sr = load_and_match(original_path, enhanced_path)

    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # Generate all plots
    plot_spectrogram_comparison(original, enhanced, sr,
                                 str(out / 'AB_spectrogram.png'))
    plot_frequency_response(original, enhanced, sr,
                            str(out / 'AB_frequency.png'))
    plot_stereo_width(original, enhanced, sr,
                      str(out / 'AB_stereo_width.png'))

    print(f"\n✓ Report generated in {output_dir}/")
    print("  - AB_spectrogram.png   : High frequencies visible in enhanced")
    print("  - AB_frequency.png     : Frequency extension comparison")
    print("  - AB_stereo_width.png  : Stereo image widening")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python ab_compare.py <original.wav> <enhanced.wav> [output_dir]")
        sys.exit(1)

    original = sys.argv[1]
    enhanced = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else '.'

    generate_report(original, enhanced, output)
