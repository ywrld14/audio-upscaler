#!/usr/bin/env python3
"""
Audio Upscaler Audio Studio
==========================

Full pipeline GUI with:
- Import: m4a, mp3, wav, flac (auto-convert)
- Pipeline: demucs-mlx → psychoacoustic → spatial → de-esser → mastering
- A/B test with live comparison
- Real-time visualization (spectrogram + goniometer)
- Export with metadata preservation

Run: python audio_studio.py
"""

import os
import json
import tempfile
import subprocess
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import gradio as gr
import soundfile as sf


# =============================================================================
# LOGGING
# =============================================================================

def log(tag, msg, flush=True):
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}", flush=flush)


def log_error(tag, e):
    """Print error with full traceback."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] ERROR: {e}", flush=True)
    print(traceback.format_exc(), flush=True)
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert, stft
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import pipeline components
from fatllama_deesser import FatLlamaDeesser


# =============================================================================
# AUDIO FORMAT CONVERSION
# =============================================================================

NEEDS_CONVERSION = {'.m4a', '.mp3', '.aac', '.ogg', '.wma', '.opus'}


def convert_to_wav(input_path):
    """Convert audio file to WAV using ffmpeg if needed."""
    input_path = Path(input_path)
    ext = input_path.suffix.lower()

    if ext not in NEEDS_CONVERSION:
        log("CONVERT", f"No conversion needed ({ext})")
        return str(input_path)

    log("CONVERT", f"{ext} → wav via ffmpeg...")
    wav_path = os.path.join(tempfile.gettempdir(), f"stratospheric_{input_path.stem}.wav")
    cmd = ['ffmpeg', '-y', '-i', str(input_path), '-acodec', 'pcm_s16le', '-ar', '44100', wav_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log_error("CONVERT", f"ffmpeg failed: {result.stderr}")
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
    log("CONVERT", f"Converted → {wav_path}")
    return wav_path


# =============================================================================
# PIPELINE COMPONENTS (simplified for GUI)
# =============================================================================

class PsychoacousticEnhancer:
    """Harmonic exciter, transient shaper, sub-harmonic synthesis."""

    def __init__(self, sr=44100):
        self.sr = sr
        self.nyq = sr / 2

    def harmonic_exciter(self, audio, hp_freq=3000, drive=0.3, mix=0.15):
        b, a = butter(4, hp_freq / self.nyq, btype='high')
        highs = filtfilt(b, a, audio, axis=0)
        driven = np.tanh(highs * (1 + drive * 10))
        even = np.abs(driven) - 0.5
        harmonics = 0.6 * even + 0.4 * driven
        b2, a2 = butter(2, hp_freq / self.nyq, btype='high')
        harmonics = filtfilt(b2, a2, harmonics, axis=0)
        return (audio + harmonics * mix).astype(np.float32)

    def subharmonic_synth(self, audio, bass_freq=80, mix=0.25):
        b, a = butter(4, bass_freq / self.nyq, btype='low')
        bass = filtfilt(b, a, audio, axis=0)
        if audio.ndim == 1:
            envelope = np.abs(hilbert(bass))
        else:
            envelope = np.abs(hilbert(bass, axis=0))
        b_env, a_env = butter(2, 30 / self.nyq, btype='low')
        envelope = filtfilt(b_env, a_env, envelope, axis=0)
        rectified = np.maximum(bass, 0)
        b_sub, a_sub = butter(4, (bass_freq / 2) / self.nyq, btype='low')
        sub = filtfilt(b_sub, a_sub, rectified, axis=0) * 2
        max_env = np.max(envelope) + 1e-10
        sub = sub * (envelope / max_env)
        return (audio + sub * mix).astype(np.float32)

    def tube_warmth(self, audio, drive=0.2, mix=0.2):
        k = drive * 5
        x = audio * (1 + k)
        saturated = np.where(x >= 0, 1 - np.exp(-x), -1 + np.exp(x)) * (1 + 0.1 * k)
        return (audio * (1 - mix) + saturated * mix).astype(np.float32)


class SpatialProcessor:
    """Stereo widening and spatial processing."""

    def __init__(self, sr=44100):
        self.sr = sr
        self.nyq = sr / 2

    def stereo_widen(self, audio, width=1.3):
        if audio.ndim == 1:
            return np.column_stack([audio, audio])
        left, right = audio[:, 0], audio[:, 1]
        mid = (left + right) / 2
        side = (left - right) / 2
        side = side * width
        return np.column_stack([mid + side, mid - side]).astype(np.float32)

    def freq_dependent_width(self, audio, bass_width=0.3, mid_width=1.0, high_width=1.5):
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
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


class BinauralSpatialProcessor:
    """
    Binaural/3D spatial processing for immersive headphone listening.
    Simulates Dolby Atmos-style spatial audio using HRTF techniques.
    """

    def __init__(self, sr=44100):
        self.sr = sr
        self.nyq = sr / 2

    def hrtf_position(self, audio, angle_degrees, elevation=0):
        """
        Apply simple HRTF (Head-Related Transfer Function) positioning.

        Args:
            audio: Mono audio array
            angle_degrees: Azimuth angle (-90 to +90, 0 = center)
            elevation: Vertical angle (not fully implemented, for future use)

        Returns:
            Stereo audio with spatial positioning
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono first

        # ITD (Interaural Time Difference) - sound arrives later at far ear
        # Max ITD is ~0.6ms for human head
        itd_samples = int(0.0006 * self.sr * np.sin(np.radians(angle_degrees)))

        # ILD (Interaural Level Difference) - sound is quieter at far ear
        # More pronounced at higher frequencies, simplified here
        ild_db = 6 * np.sin(np.radians(angle_degrees))

        left = audio.copy()
        right = audio.copy()

        # Apply time delay
        if itd_samples > 0:
            # Sound on right side - right ear hears first
            left = np.pad(left, (itd_samples, 0))[:len(audio)]
        elif itd_samples < 0:
            # Sound on left side - left ear hears first
            right = np.pad(right, (-itd_samples, 0))[:len(audio)]

        # Apply level difference
        left_gain = 10**(-ild_db/20)
        right_gain = 10**(ild_db/20)
        left = left * left_gain
        right = right * right_gain

        # Add subtle high-frequency shadowing on far ear (head shadow effect)
        if abs(angle_degrees) > 15:
            shadow_ear = right if angle_degrees < 0 else left
            b, a = butter(2, 8000 / self.nyq, btype='low')
            shadowed = filtfilt(b, a, shadow_ear)
            blend = min(abs(angle_degrees) / 90, 0.5)
            if angle_degrees < 0:
                right = shadow_ear * (1 - blend) + shadowed * blend
            else:
                left = shadow_ear * (1 - blend) + shadowed * blend

        return np.column_stack([left, right]).astype(np.float32)

    def add_early_reflections(self, audio, depth=0.5):
        """Add subtle early reflections to simulate room acoustics."""
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        output = audio.copy()

        # Early reflections at different delays and gains
        reflections = [
            (0.015, 0.15, -20),   # 15ms, left, -20 degrees
            (0.022, 0.12, 25),    # 22ms, right, +25 degrees
            (0.031, 0.08, -45),   # 31ms, slight left
            (0.045, 0.05, 50),    # 45ms, right
        ]

        for delay_sec, gain, angle in reflections:
            delay_samples = int(delay_sec * self.sr)
            # Apply HRTF to mono sum for reflection
            mono = np.mean(audio, axis=1)
            reflection = self.hrtf_position(mono, angle)

            # Apply high-frequency rolloff (air absorption)
            b, a = butter(2, 6000 / self.nyq, btype='low')
            reflection = filtfilt(b, a, reflection, axis=0)

            # Delay and add
            padded = np.pad(reflection, ((delay_samples, 0), (0, 0)))[:len(audio)]
            output += padded * gain * depth

        return output.astype(np.float32)

    def process_stems_binaural(self, stems, depth=0.7):
        """
        Process separated stems with binaural spatial positioning.

        Positioning:
        - Vocals: Center (0°), slightly in front
        - Drums: Center with width, subtle rear reflections
        - Bass: Center mono (0°)
        - Other (synths, guitars): Spread to sides (±30° to ±60°)
        """
        log("BINAURAL", f"Processing stems with spatial depth={depth}")

        max_len = max(len(s) for s in stems.values())
        output = np.zeros((max_len, 2), dtype=np.float32)

        # Vocals: Dead center, no HRTF needed, just ensure stereo
        vocals = stems.get('vocals', np.zeros((max_len, 2)))
        if vocals.ndim == 1:
            vocals = np.column_stack([vocals, vocals])
        output += vocals * 0.85
        log("BINAURAL", "  → Vocals: 0° (center)")

        # Bass: Mono center - bass should not be spatialized
        bass = stems.get('bass', np.zeros((max_len, 2)))
        if bass.ndim > 1:
            bass = np.mean(bass, axis=1)
        bass_stereo = np.column_stack([bass, bass])
        if len(bass_stereo) < max_len:
            bass_stereo = np.pad(bass_stereo, ((0, max_len - len(bass_stereo)), (0, 0)))
        output += bass_stereo * 0.75
        log("BINAURAL", "  → Bass: 0° (mono center)")

        # Drums: Slightly widened with early reflections for "behind" effect
        drums = stems.get('drums', np.zeros((max_len, 2)))
        if drums.ndim == 1:
            drums = np.column_stack([drums, drums])
        # Add subtle early reflections to push drums slightly back
        drums_spatial = self.add_early_reflections(drums, depth * 0.4)
        if len(drums_spatial) < max_len:
            drums_spatial = np.pad(drums_spatial, ((0, max_len - len(drums_spatial)), (0, 0)))
        output += drums_spatial * 0.72
        log("BINAURAL", "  → Drums: 0° with rear reflections")

        # Other: Split and position at ±40°
        other = stems.get('other', np.zeros((max_len, 2)))
        if other.ndim == 1:
            other = np.column_stack([other, other])

        # Position left channel content at -40°, right at +40°
        other_left = self.hrtf_position(other[:, 0], -40 * depth)
        other_right = self.hrtf_position(other[:, 1], 40 * depth)

        if len(other_left) < max_len:
            other_left = np.pad(other_left, ((0, max_len - len(other_left)), (0, 0)))
        if len(other_right) < max_len:
            other_right = np.pad(other_right, ((0, max_len - len(other_right)), (0, 0)))

        output += (other_left + other_right) * 0.5 * 0.65
        log("BINAURAL", "  → Other: ±40° (sides)")

        # Add overall early reflections for room feel
        output = self.add_early_reflections(output, depth * 0.3)
        log("BINAURAL", f"  → Early reflections added (depth={depth*0.3:.2f})")

        return output


class MasteringProcessor:
    """Loudness normalization and limiting."""

    def __init__(self, sr=44100):
        self.sr = sr

    def normalize_rms(self, audio, target_db=-14.0):
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-10:
            target_rms = 10 ** (target_db / 20)
            gain = target_rms / rms
            audio = audio * min(gain, 10)  # Limit gain
        return audio.astype(np.float32)

    def soft_clip(self, audio, threshold=0.95):
        return np.tanh(audio / threshold) * threshold

    def process(self, audio, target_db=-14.0):
        audio = self.normalize_rms(audio, target_db)
        audio = self.soft_clip(audio)
        return np.clip(audio, -1.0, 1.0).astype(np.float32)


# =============================================================================
# FULL Audio Upscaler PIPELINE
# =============================================================================

def separate_stems_demucs(audio_path):
    """Source separation using demucs-mlx."""
    log("DEMUCS", "Initializing source separation (htdemucs model)...")
    try:
        from demucs_mlx import Separator
        t_start = time.time()
        separator = Separator(model="htdemucs")
        log("DEMUCS", "Running separation (this takes ~10-15s)...")
        origin, stems = separator.separate_audio_file(audio_path)

        result = {}
        for name in ['vocals', 'drums', 'bass', 'other']:
            arr = np.array(stems[name])
            if arr.ndim == 1:
                arr = np.column_stack([arr, arr])
            elif arr.shape[0] == 2:
                arr = arr.T
            result[name] = arr
            log("DEMUCS", f"  → {name}: {arr.shape[0]/44100:.1f}s")

        elapsed = time.time() - t_start
        log("DEMUCS", f"Separation complete in {elapsed:.1f}s")
        return result, 44100
    except Exception as e:
        log_error("DEMUCS", e)
        raise RuntimeError(f"Demucs separation failed: {e}")


def process_stratospheric(audio_path, deesser_strength=0.5, stereo_width=1.3, spatial_depth=0.0, progress_callback=None):
    """Full Audio Upscaler pipeline."""
    log("PIPELINE", "=" * 50)
    log("PIPELINE", "Starting Audio Upscaler enhancement")
    log("PIPELINE", f"  Input: {Path(audio_path).name}")
    log("PIPELINE", f"  De-esser: {deesser_strength}, Stereo: {stereo_width}x, Spatial: {spatial_depth}")
    log("PIPELINE", "=" * 50)

    pipeline_start = time.time()
    steps = ["Converting", "Separating stems", "Enhancing", "Spatial processing", "De-essing", "Mastering"]

    def update_progress(step_idx, msg=""):
        if progress_callback:
            pct = (step_idx + 1) / len(steps)
            progress_callback(pct, f"{steps[step_idx]}... {msg}")

    # Step 1: Convert if needed
    log("PIPELINE", "[1/6] Converting format...")
    update_progress(0)
    wav_path = convert_to_wav(audio_path)

    # Step 2: Source separation
    log("PIPELINE", "[2/6] Source separation (demucs-mlx)...")
    update_progress(1)
    stems, sr = separate_stems_demucs(wav_path)

    # Step 3: Psychoacoustic enhancement
    log("PIPELINE", "[3/6] Psychoacoustic enhancement...")
    update_progress(2)
    enhancer = PsychoacousticEnhancer(sr=sr)

    stem_configs = {
        'vocals': {'harmonic': (5000, 0.12, 0.06), 'tube': (0.1, 0.12)},
        'drums': {'harmonic': (5000, 0.15, 0.08), 'tube': None},
        'bass': {'sub': (80, 0.22), 'tube': (0.15, 0.18)},
        'other': {'harmonic': (3000, 0.25, 0.14), 'tube': (0.08, 0.1)},
    }

    for stem_name, config in stem_configs.items():
        audio = stems[stem_name]
        if 'harmonic' in config and config['harmonic']:
            audio = enhancer.harmonic_exciter(audio, *config['harmonic'])
        if 'sub' in config and config['sub']:
            audio = enhancer.subharmonic_synth(audio, *config['sub'])
        if 'tube' in config and config['tube']:
            audio = enhancer.tube_warmth(audio, *config['tube'])
        stems[stem_name] = audio

    log("PIPELINE", "  → Enhancement complete")

    # Step 4: Spatial processing + mixing
    log("PIPELINE", "[4/6] Spatial 3D processing...")
    update_progress(3)

    if spatial_depth > 0:
        # Use binaural/3D spatial processing (Dolby Atmos style)
        log("PIPELINE", f"  → Using BINAURAL 3D mode (depth={spatial_depth})")
        binaural = BinauralSpatialProcessor(sr=sr)
        output = binaural.process_stems_binaural(stems, depth=spatial_depth)

        # Apply additional stereo width adjustment if needed
        if stereo_width != 1.0:
            spatial = SpatialProcessor(sr=sr)
            output = spatial.stereo_widen(output, stereo_width)
    else:
        # Standard stereo processing
        spatial = SpatialProcessor(sr=sr)

        spatial_config = {
            'vocals': (1.15, 0.88),
            'drums': (1.45, 0.78),
            'bass': (0.6, 0.85),
            'other': (1.7, 0.68),
        }

        max_len = max(len(s) for s in stems.values())
        output = np.zeros((max_len, 2), dtype=np.float32)

        for stem_name, (width, gain) in spatial_config.items():
            audio = stems[stem_name]
            audio = spatial.stereo_widen(audio, width * stereo_width)
            if len(audio) < max_len:
                audio = np.pad(audio, ((0, max_len - len(audio)), (0, 0)))
            output += audio * gain

        output = spatial.freq_dependent_width(output, bass_width=0.4, mid_width=1.1, high_width=1.4)

    log("PIPELINE", "  → Spatial processing complete")

    # Step 5: De-essing
    log("PIPELINE", f"[5/6] De-essing (Fat Llama, strength={deesser_strength})...")
    update_progress(4)
    deesser = FatLlamaDeesser(sample_rate=sr, strength=deesser_strength)
    output = deesser.process(output)
    log("PIPELINE", "  → De-essing complete")

    # Step 6: Mastering
    log("PIPELINE", "[6/6] Mastering (normalization + soft clip)...")
    update_progress(5)
    master = MasteringProcessor(sr=sr)
    output = master.process(output, target_db=-14.0)

    elapsed = time.time() - pipeline_start
    log("PIPELINE", "=" * 50)
    log("DONE", f"Processing complete in {elapsed:.1f}s")
    log("PIPELINE", f"  Output: {len(output)/sr:.1f}s stereo @ {sr}Hz")
    log("PIPELINE", "=" * 50)

    return output, sr


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_spectrogram(audio, sr, title="Spectrogram"):
    """Create spectrogram visualization (optimized for speed)."""
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Limit to 60 seconds max for performance
    max_samples = min(len(audio), sr * 60)
    audio_viz = audio[:max_samples]
    duration = len(audio) / sr

    # Compute spectrogram
    f, t, Sxx = stft(audio_viz, sr, nperseg=2048, noverlap=1536)
    magnitude = 20 * np.log10(np.abs(Sxx) + 1e-10)

    # Use imshow instead of pcolormesh (27x faster!)
    # Limit frequency to 20kHz max
    max_freq_idx = min(len(f), int(20000 / (f[1] - f[0])) if len(f) > 1 else len(f))
    magnitude_viz = magnitude[:max_freq_idx, :]

    im = ax.imshow(magnitude_viz, aspect='auto', origin='lower', cmap='magma',
                   vmin=-80, vmax=0, extent=[0, len(audio_viz)/sr, 0, f[max_freq_idx-1]/1000])

    ax.set_ylabel('Frequency (kHz)', color='white', fontsize=10)
    ax.set_xlabel('Time (s)', color='white', fontsize=10)
    suffix = f" (first 60s of {duration:.0f}s)" if duration > 60 else ""
    ax.set_title(f"{title}{suffix}", color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#888')

    for spine in ax.spines.values():
        spine.set_color('#333')

    cbar = plt.colorbar(im, ax=ax, label='dB', shrink=0.8)
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='#888')
    plt.tight_layout()
    return fig


def create_goniometer(audio, title="Stereo Field"):
    """Create goniometer (Lissajous) visualization."""
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    if audio.ndim == 1:
        ax.text(0.5, 0.5, 'Mono', ha='center', va='center', color='white', fontsize=14)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return fig

    # Subsample for performance
    step = max(1, len(audio) // 20000)
    left = audio[::step, 0]
    right = audio[::step, 1]

    # Mid-Side
    mid = (left + right) / 2
    side = (left - right) / 2

    # Plot
    colors = np.linspace(0, 1, len(mid))
    ax.scatter(mid, side, c=colors, cmap='cool', alpha=0.3, s=1)

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel('Mid (L+R)', color='white', fontsize=10)
    ax.set_ylabel('Side (L-R)', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='#444', linewidth=0.5)
    ax.axvline(x=0, color='#444', linewidth=0.5)
    ax.set_aspect('equal')
    ax.tick_params(colors='#888')

    for spine in ax.spines.values():
        spine.set_color('#333')

    # Width indicator
    width_ratio = np.std(side) / (np.std(mid) + 1e-10)
    ax.text(0.02, 0.98, f'Width: {width_ratio:.2f}', transform=ax.transAxes,
            fontsize=10, color='#00ff88', verticalalignment='top')

    plt.tight_layout()
    return fig


def create_waveform_comparison(original, processed, sr):
    """Side-by-side waveform comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 4), facecolor='#1a1a2e')

    for ax, audio, title, color in [
        (axes[0], original, 'Original', '#ff6b6b'),
        (axes[1], processed, 'Audio Upscaler', '#4ecdc4')
    ]:
        ax.set_facecolor('#1a1a2e')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Downsample for display
        step = max(1, len(audio) // 5000)
        display = audio[::step]
        time = np.linspace(0, len(audio) / sr, len(display))

        ax.plot(time, display, color=color, linewidth=0.5, alpha=0.8)
        ax.fill_between(time, display, alpha=0.3, color=color)
        ax.set_ylabel('Amp', color='white', fontsize=9)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold')
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color='#444', linewidth=0.5)
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_color('#333')

    axes[1].set_xlabel('Time (s)', color='white', fontsize=10)
    plt.tight_layout()
    return fig


def create_spectrogram_comparison(original, processed, sr):
    """Side-by-side spectrogram comparison: Original vs Audio Upscaler."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor='#1a1a2e')

    for ax, audio, title, cmap in [
        (axes[0], original, 'Original', 'magma'),
        (axes[1], processed, 'Audio Upscaler', 'viridis')
    ]:
        ax.set_facecolor('#1a1a2e')
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio

        # Limit to 60 seconds max
        max_samples = min(len(audio_mono), sr * 60)
        audio_viz = audio_mono[:max_samples]

        # Compute spectrogram
        f, t, Sxx = stft(audio_viz, sr, nperseg=2048, noverlap=1536)
        magnitude = 20 * np.log10(np.abs(Sxx) + 1e-10)

        # Limit frequency to 20kHz
        max_freq_idx = min(len(f), int(20000 / (f[1] - f[0])) if len(f) > 1 else len(f))
        magnitude_viz = magnitude[:max_freq_idx, :]

        im = ax.imshow(magnitude_viz, aspect='auto', origin='lower', cmap=cmap,
                       vmin=-80, vmax=0, extent=[0, len(audio_viz)/sr, 0, f[max_freq_idx-1]/1000])

        ax.set_ylabel('Freq (kHz)', color='white', fontsize=9)
        ax.set_xlabel('Time (s)', color='white', fontsize=9)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_color('#333')

    plt.tight_layout()
    return fig


def create_goniometer_comparison(original, processed):
    """Side-by-side goniometer comparison: Original vs Audio Upscaler."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='#1a1a2e')

    for ax, audio, title, cmap in [
        (axes[0], original, 'Original', 'Reds'),
        (axes[1], processed, 'Audio Upscaler', 'Blues')
    ]:
        ax.set_facecolor('#1a1a2e')

        if audio.ndim == 1:
            ax.text(0.5, 0.5, 'Mono', ha='center', va='center', color='white', fontsize=14,
                    transform=ax.transAxes)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
        else:
            # Subsample for performance
            step = max(1, len(audio) // 15000)
            left = audio[::step, 0]
            right = audio[::step, 1]

            # Mid-Side
            mid = (left + right) / 2
            side = (left - right) / 2

            # Plot
            colors = np.linspace(0, 1, len(mid))
            ax.scatter(mid, side, c=colors, cmap=cmap, alpha=0.4, s=1)

            ax.set_xlim(-0.8, 0.8)
            ax.set_ylim(-0.8, 0.8)

        ax.set_xlabel('Mid (L+R)', color='white', fontsize=9)
        ax.set_ylabel('Side (L-R)', color='white', fontsize=9)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold')
        ax.axhline(y=0, color='#444', linewidth=0.5)
        ax.axvline(x=0, color='#444', linewidth=0.5)
        ax.set_aspect('equal')
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_color('#333')

    plt.tight_layout()
    return fig


# =============================================================================
# GUI STATE
# =============================================================================

class AppState:
    def __init__(self):
        self.original_audio = None
        self.processed_audio = None
        self.sample_rate = None
        self.original_path = None
        self.filename = None
        self.is_processing = False

    def reset(self):
        self.original_audio = None
        self.processed_audio = None
        self.sample_rate = None
        self.original_path = None
        self.filename = None
        self.is_processing = False


state = AppState()


# =============================================================================
# GUI FUNCTIONS
# =============================================================================

def load_audio(file):
    """Load audio file."""
    if file is None:
        return None, None, None, "No file selected"

    state.reset()

    try:
        # Get file info
        file_path = Path(file)
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        log("UPLOAD", f"{file_path.name} ({file_size:.1f} MB)")

        wav_path = convert_to_wav(file)
        audio, sr = sf.read(wav_path)
        state.original_audio = audio
        state.sample_rate = sr
        state.filename = Path(file).stem

        # Copy original to stable location for metadata export later
        # Gradio temp files can be deleted before export
        import shutil
        stable_original = os.path.join(tempfile.gettempdir(), f"stratospheric_original_{state.filename}{Path(file).suffix}")
        shutil.copy2(file, stable_original)
        state.original_path = stable_original
        log("UPLOAD", f"Original copied to: {stable_original}")

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
            state.original_audio = audio

        duration = len(audio) / sr
        channels = "Stereo" if audio.ndim > 1 else "Mono"

        log("UPLOAD", f"Loaded: {duration:.1f}s | {channels} | {sr}Hz")

        info = f"**{state.filename}**\n\n"
        info += f"Duration: {duration:.1f}s | {channels} | {sr} Hz\n\n"
        info += "Ready to process!"

        # Create visualizations
        log("UPLOAD", "Creating visualizations...")
        spec_fig = create_spectrogram(audio, sr, "Original Spectrogram")
        gonio_fig = create_goniometer(audio, "Original Stereo Field")
        log("UPLOAD", "Ready!")

        return (sr, audio), spec_fig, gonio_fig, info

    except Exception as e:
        log_error("UPLOAD", e)
        return None, None, None, f"Error: {str(e)}"


def process_audio(deesser_strength, stereo_width, spatial_depth, progress=gr.Progress()):
    """Run Audio Upscaler pipeline."""
    if state.original_audio is None:
        log("PROCESS", "No audio loaded!")
        return None, None, None, None, "Please load an audio file first!"

    if state.is_processing:
        log("PROCESS", "Already processing, ignoring request")
        return None, None, None, None, "Already processing..."

    state.is_processing = True
    log("PROCESS", f"Starting pipeline for: {state.filename}")
    log("PROCESS", f"  Settings: deesser={deesser_strength}, stereo={stereo_width}, spatial={spatial_depth}")

    try:
        def progress_callback(pct, msg):
            progress(pct, desc=msg)

        # Run pipeline
        processed, sr = process_stratospheric(
            state.original_path,
            deesser_strength=deesser_strength,
            stereo_width=stereo_width,
            spatial_depth=spatial_depth,
            progress_callback=progress_callback
        )

        state.processed_audio = processed
        state.sample_rate = sr

        # Create A/B comparison visualizations
        log("PROCESS", "Creating A/B comparison visualizations...")
        spec_fig = create_spectrogram_comparison(state.original_audio, processed, sr)
        gonio_fig = create_goniometer_comparison(state.original_audio, processed)
        compare_fig = create_waveform_comparison(state.original_audio, processed, sr)
        log("PROCESS", "Visualizations ready!")

        info = f"**Processing complete!**\n\n"
        info += f"De-esser strength: {deesser_strength}\n"
        info += f"Stereo width: {stereo_width}x\n"
        if spatial_depth > 0:
            info += f"Spatial 3D depth: {spatial_depth} 🎧\n"
        info += "\nUse A/B buttons to compare!"

        return (sr, processed), spec_fig, gonio_fig, compare_fig, info

    except Exception as e:
        log_error("PROCESS", e)
        return None, None, None, None, f"Error: {str(e)}"
    finally:
        state.is_processing = False


def get_original():
    """Get original audio for A/B test."""
    if state.original_audio is None:
        return None
    return (state.sample_rate, state.original_audio)


def get_processed():
    """Get processed audio for A/B test."""
    if state.processed_audio is None:
        return None
    return (state.sample_rate, state.processed_audio)


def export_audio():
    """Export processed audio with metadata."""
    if state.processed_audio is None:
        log("EXPORT", "No processed audio to export!")
        return None, "Please process audio first!"

    log("EXPORT", f"Exporting {state.filename}...")

    try:
        # Create temp WAV
        temp_wav = os.path.join(tempfile.gettempdir(), f"{state.filename}_temp.wav")
        sf.write(temp_wav, state.processed_audio, state.sample_rate)

        # Output as FLAC with metadata from original
        output_name = f"{state.filename}_Audio Upscaler.flac"
        output_path = os.path.join(tempfile.gettempdir(), output_name)

        if state.original_path and os.path.exists(state.original_path):
            log("EXPORT", f"Copying metadata from: {state.original_path}")
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_wav,
                '-i', state.original_path,
                '-map', '0:a',
                '-map', '1:v?',
                '-c:a', 'flac',
                '-c:v', 'copy',
                '-map_metadata', '1',
                '-metadata:s:a:0', f'title={state.filename}',
                output_path
            ]
            log("EXPORT", f"Running: ffmpeg with metadata copy")
            result = subprocess.run(cmd, capture_output=True, text=True)
            os.remove(temp_wav)

            if result.returncode != 0:
                log("EXPORT", f"ffmpeg failed (code {result.returncode}): {result.stderr[:200]}")
                # Fallback: just WAV
                sf.write(output_path.replace('.flac', '.wav'), state.processed_audio, state.sample_rate)
                output_path = output_path.replace('.flac', '.wav')
        else:
            log("EXPORT", "No original file for metadata, exporting as WAV")
            os.rename(temp_wav, output_path.replace('.flac', '.wav'))
            output_path = output_path.replace('.flac', '.wav')

        # Log final file size
        final_size = Path(output_path).stat().st_size / (1024 * 1024)
        log("EXPORT", f"Saved: {output_path}")
        log("EXPORT", f"Size: {final_size:.1f} MB")

        return output_path, f"Exported: {Path(output_path).name}"

    except Exception as e:
        log_error("EXPORT", e)
        return None, f"Export error: {str(e)}"


def export_dolby_atmos():
    """Export as Dolby Atmos 7.1.4 using spatial-audio-upmixer."""
    if state.processed_audio is None and state.original_audio is None:
        log("ATMOS", "No audio to export!")
        return None, "Please load audio first!"

    log("ATMOS", f"Creating Dolby Atmos 7.1.4 for {state.filename}...")

    try:
        from upmixer import UpmixPipeline, UpmixConfig

        # Use processed audio if available, otherwise original
        using_processed = state.processed_audio is not None
        audio_to_use = state.processed_audio if using_processed else state.original_audio
        sr = state.sample_rate

        # Debug: log which audio is being used and show difference
        log("ATMOS", f"Using {'PROCESSED (Audio Upscaler applied)' if using_processed else 'ORIGINAL (no processing)'} audio")
        log("ATMOS", f"Audio shape: {audio_to_use.shape}, dtype: {audio_to_use.dtype}")

        # Calculate and log audio stats
        rms = np.sqrt(np.mean(audio_to_use**2))
        spectrum = np.abs(np.fft.rfft(audio_to_use.mean(axis=1) if audio_to_use.ndim > 1 else audio_to_use))
        freqs = np.fft.rfftfreq(len(audio_to_use), 1/sr)
        spectral_centroid = np.sum(freqs * spectrum) / np.sum(spectrum)

        log("ATMOS", f"Audio stats - RMS: {rms:.4f}, Spectral Centroid: {spectral_centroid:.0f} Hz")

        if using_processed and state.original_audio is not None:
            # Show difference from original
            orig_rms = np.sqrt(np.mean(state.original_audio**2))
            log("ATMOS", f"Difference from original - RMS delta: {rms - orig_rms:+.4f}")

        # Create temp input file
        temp_input = os.path.join(tempfile.gettempdir(), f"{state.filename}_atmos_input.wav")
        sf.write(temp_input, audio_to_use, sr)

        # Output path
        output_name = f"{state.filename}_ATMOS_7.1.4.wav"
        output_path = os.path.join(tempfile.gettempdir(), output_name)

        # Configure for Dolby Atmos 7.1.4
        log("ATMOS", "Configuring 7.1.4 upmix pipeline...")
        config = UpmixConfig(
            output_format='7.1.4',
            output_subtype='PCM_24',
            loudness_normalize=True,
            loudness_target_lkfs=-18.0,
            spatial_intensity=1.0
        )

        # Process
        log("ATMOS", "Running upmix pipeline (this may take a moment)...")
        pipeline = UpmixPipeline(config)
        result = pipeline.process_file(temp_input, output_path)

        # Cleanup
        os.remove(temp_input)

        # Log result
        final_size = Path(output_path).stat().st_size / (1024 * 1024)
        log("ATMOS", f"Saved: {output_path}")
        log("ATMOS", f"Size: {final_size:.1f} MB | Format: 7.1.4 ADM-BWF")

        source_info = "with Audio Upscaler processing" if using_processed else "from ORIGINAL audio (no processing)"
        return output_path, f"Dolby Atmos 7.1.4 exported {source_info}: {output_name} ({final_size:.1f} MB)\n\nNote: File size is determined by format (12 channels × 24-bit), not audio content."

    except ImportError as e:
        log_error("ATMOS", e)
        return None, "upmixer package not installed. Run: pip install git+https://github.com/coderynx/spatial-audio-upmixer.git"
    except Exception as e:
        log_error("ATMOS", e)
        return None, f"Atmos export error: {str(e)}"


def export_binaural():
    """Export as binaural stereo (headphone optimized)."""
    if state.processed_audio is None:
        log("BINAURAL", "No processed audio!")
        return None, "Please process audio first!"

    log("BINAURAL", f"Exporting binaural for {state.filename}...")

    try:
        # The processed audio with spatial_depth already has binaural processing
        # Just export it as-is
        output_name = f"{state.filename}_BINAURAL.flac"
        output_path = os.path.join(tempfile.gettempdir(), output_name)

        # Create temp WAV
        temp_wav = os.path.join(tempfile.gettempdir(), f"{state.filename}_binaural_temp.wav")
        sf.write(temp_wav, state.processed_audio, state.sample_rate)

        # Export with metadata if available
        if state.original_path and os.path.exists(state.original_path):
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_wav,
                '-i', state.original_path,
                '-map', '0:a', '-map', '1:v?',
                '-c:a', 'flac', '-c:v', 'copy',
                '-map_metadata', '1',
                '-metadata:s:a:0', f'title={state.filename}',
                output_path
            ]
            subprocess.run(cmd, capture_output=True, text=True)
            os.remove(temp_wav)
        else:
            sf.write(output_path, state.processed_audio, state.sample_rate)

        final_size = Path(output_path).stat().st_size / (1024 * 1024)
        log("BINAURAL", f"Saved: {output_path} ({final_size:.1f} MB)")

        return output_path, f"Binaural exported: {Path(output_path).name}"

    except Exception as e:
        log_error("BINAURAL", e)
        return None, f"Binaural export error: {str(e)}"


def export_atmos_final():
    """Export as REAL Dolby Atmos E-AC-3 JOC (Apple Music compatible M4A)."""
    if state.processed_audio is None and state.original_audio is None:
        log("ATMOS-FINAL", "No audio to export!")
        return None, "Please load audio first!"

    log("ATMOS-FINAL", f"Creating REAL Dolby Atmos E-AC-3 JOC for {state.filename}...")

    try:
        import ac3forge as ac3
        from demucs_mlx import Separator

        # Use processed audio if available
        using_processed = state.processed_audio is not None
        audio_to_use = state.processed_audio if using_processed else state.original_audio
        sr = state.sample_rate

        log("ATMOS-FINAL", f"Using {'PROCESSED' if using_processed else 'ORIGINAL'} audio")

        # Ensure stereo and 48kHz for Atmos
        if audio_to_use.ndim == 1:
            audio_to_use = np.column_stack([audio_to_use, audio_to_use])

        if sr != 48000:
            from scipy.signal import resample
            new_len = int(len(audio_to_use) * 48000 / sr)
            audio_to_use = resample(audio_to_use, new_len, axis=0)
            sr = 48000
            log("ATMOS-FINAL", f"Resampled to 48kHz")

        # Step 1: Separate stems with demucs
        log("ATMOS-FINAL", "[1/4] Separating stems with demucs-mlx...")
        temp_input = os.path.join(tempfile.gettempdir(), f"{state.filename}_atmos_input.wav")
        sf.write(temp_input, audio_to_use, sr)

        separator = Separator(model="htdemucs")
        _, stems = separator.separate_audio_file(temp_input)  # Returns (audio, stems_dict)
        os.remove(temp_input)

        # stems: dict with 'vocals', 'drums', 'bass', 'other' - shape (2, N)
        def ensure_mono(stem):
            if stem is None:
                return np.zeros(1000, dtype=np.float32)
            if stem.ndim == 1:
                return stem.astype(np.float32)
            elif stem.shape[0] == 2:
                return stem.mean(axis=0).astype(np.float32)
            else:
                return stem.mean(axis=1).astype(np.float32)

        vocals_mono = ensure_mono(stems.get('vocals'))
        drums_mono = ensure_mono(stems.get('drums'))
        bass_mono = ensure_mono(stems.get('bass'))
        other_mono = ensure_mono(stems.get('other'))

        log("ATMOS-FINAL", "[2/4] Encoding E-AC-3 JOC with ac3forge...")

        # Create Atmos encoder with 4 objects
        config = ac3.AtmosConfig(
            sample_rate=ac3.SampleRate.k48000,
            bitrate_kbps=768,  # High quality
            emit_object_metadata=True
        )
        encoder = ac3.AtmosEncoder(config, 4)

        # Object positions in 3D space (x, y, z)
        # x: -1 (left) to 1 (right)
        # y: -1 (back) to 1 (front)
        # z: 0 (floor) to 1 (ceiling)
        placements = [
            ac3.ObjectPlacement(position=ac3.Position(x=0.0, y=1.0, z=0.5), gain=1.0, lfe_send=0.0),  # vocals: front center, elevated
            ac3.ObjectPlacement(position=ac3.Position(x=0.0, y=-0.2, z=0.3), gain=1.0, lfe_send=0.3), # drums: slightly back, with LFE
            ac3.ObjectPlacement(position=ac3.Position(x=0.0, y=0.0, z=0.0), gain=1.0, lfe_send=0.8),  # bass: center, heavy LFE
            ac3.ObjectPlacement(position=ac3.Position(x=-0.5, y=0.3, z=0.6), gain=0.9, lfe_send=0.0), # other: left front, elevated
        ]

        # Encode all frames
        frames = []
        n_samples = len(vocals_mono)
        frame_size = ac3.SAMPLES_PER_FRAME

        for i in range(0, n_samples - frame_size, frame_size):
            frame_vocals = vocals_mono[i:i+frame_size]
            frame_drums = drums_mono[i:i+frame_size]
            frame_bass = bass_mono[i:i+frame_size]
            frame_other = other_mono[i:i+frame_size]

            objects = [frame_vocals, frame_drums, frame_bass, frame_other]
            frame_data = encoder.encode_frame(objects, placements)
            frames.append(frame_data)

        # Combine all frames
        ec3_data = b''.join(frames)
        log("ATMOS-FINAL", f"Encoded {len(frames)} frames, {len(ec3_data)} bytes total")

        # Step 3: Write raw E-AC-3 file
        log("ATMOS-FINAL", "[3/4] Creating M4A container...")
        temp_ec3 = os.path.join(tempfile.gettempdir(), f"{state.filename}_atmos.ec3")
        with open(temp_ec3, 'wb') as f:
            f.write(ec3_data)

        # Step 4: Wrap in M4A container with ffmpeg
        output_name = f"{state.filename}_DOLBY_ATMOS.m4a"
        output_path = os.path.join(tempfile.gettempdir(), output_name)

        # Copy metadata from original if available
        if state.original_path and os.path.exists(state.original_path):
            cmd = [
                'ffmpeg', '-y',
                '-f', 'eac3', '-i', temp_ec3,
                '-i', state.original_path,
                '-map', '0:a',
                '-c:a', 'copy',
                '-map_metadata', '1',
                '-metadata:s:a:0', f'title={state.filename}',
                '-f', 'mp4',
                '-movflags', '+faststart',
                output_path
            ]
        else:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'eac3', '-i', temp_ec3,
                '-c:a', 'copy',
                '-metadata', f'title={state.filename}',
                '-f', 'mp4',
                '-movflags', '+faststart',
                output_path
            ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            log("ATMOS-FINAL", f"ffmpeg M4A failed, keeping EC3")
            os.rename(temp_ec3, output_path.replace('.m4a', '.ec3'))
            output_path = output_path.replace('.m4a', '.ec3')
        else:
            os.remove(temp_ec3)

        final_size = Path(output_path).stat().st_size / (1024 * 1024)
        log("ATMOS-FINAL", f"[4/4] Saved: {output_path}")
        log("ATMOS-FINAL", f"Size: {final_size:.1f} MB | Format: Dolby Atmos E-AC-3 JOC")

        source_info = "with Audio Upscaler" if using_processed else "from ORIGINAL"
        return output_path, f"🎬 REAL Dolby Atmos exported {source_info}!\n\nFile: {output_name}\nSize: {final_size:.1f} MB\nFormat: E-AC-3 JOC (Apple Music compatible)\n\nImport into Apple Music or play with VLC."

    except ImportError as e:
        log_error("ATMOS-FINAL", e)
        return None, f"Missing dependency: {str(e)}\n\nRun: pip install ac3forge demucs-mlx"
    except Exception as e:
        log_error("ATMOS-FINAL", e)
        return None, f"Atmos export error: {str(e)}"


def export_51_surround():
    """Export as standard 5.1 surround (E-AC-3) for VLC compatibility testing."""
    if state.processed_audio is None and state.original_audio is None:
        log("5.1-EXPORT", "No audio to export!")
        return None, "Please load audio first!"

    log("5.1-EXPORT", f"Creating 5.1 Surround for {state.filename}...")

    try:
        from demucs_mlx import Separator

        using_processed = state.processed_audio is not None
        audio_to_use = state.processed_audio if using_processed else state.original_audio
        sr = state.sample_rate

        log("5.1-EXPORT", f"Using {'PROCESSED' if using_processed else 'ORIGINAL'} audio")

        if audio_to_use.ndim == 1:
            audio_to_use = np.column_stack([audio_to_use, audio_to_use])

        if sr != 48000:
            from scipy.signal import resample
            new_len = int(len(audio_to_use) * 48000 / sr)
            audio_to_use = resample(audio_to_use, new_len, axis=0)
            sr = 48000
            log("5.1-EXPORT", "Resampled to 48kHz")

        log("5.1-EXPORT", "[1/3] Separating stems with demucs-mlx...")
        temp_input = os.path.join(tempfile.gettempdir(), f"{state.filename}_51_input.wav")
        sf.write(temp_input, audio_to_use, sr)

        separator = Separator(model="htdemucs")
        _, stems = separator.separate_audio_file(temp_input)
        os.remove(temp_input)

        def get_stereo(stem):
            if stem is None:
                return np.zeros((1000, 2), dtype=np.float32)
            if stem.ndim == 1:
                return np.column_stack([stem, stem]).astype(np.float32)
            elif stem.shape[0] == 2:
                return stem.T.astype(np.float32)
            else:
                return stem.astype(np.float32)

        vocals = get_stereo(stems.get('vocals'))
        drums = get_stereo(stems.get('drums'))
        bass = get_stereo(stems.get('bass'))
        other = get_stereo(stems.get('other'))

        n_samples = min(len(vocals), len(drums), len(bass), len(other))
        vocals = vocals[:n_samples]
        drums = drums[:n_samples]
        bass = bass[:n_samples]
        other = other[:n_samples]

        log("5.1-EXPORT", "[2/3] Creating 5.1 channel layout...")

        # 5.1 channel order: FL, FR, FC, LFE, SL, SR
        output_51 = np.zeros((n_samples, 6), dtype=np.float32)

        # FL (Front Left) - drums L + other L + some vocals L
        output_51[:, 0] = drums[:, 0] * 0.6 + other[:, 0] * 0.4 + vocals[:, 0] * 0.3

        # FR (Front Right) - drums R + other R + some vocals R
        output_51[:, 1] = drums[:, 1] * 0.6 + other[:, 1] * 0.4 + vocals[:, 1] * 0.3

        # FC (Front Center) - vocals mono + bass mono
        vocals_mono = (vocals[:, 0] + vocals[:, 1]) * 0.5
        bass_mono = (bass[:, 0] + bass[:, 1]) * 0.5
        output_51[:, 2] = vocals_mono * 0.9 + bass_mono * 0.3

        # LFE (Subwoofer) - bass only, low-passed
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 120 / (sr / 2), btype='low')
        output_51[:, 3] = filtfilt(b, a, bass_mono) * 1.2

        # SL (Surround Left) - other L + ambient drums
        output_51[:, 4] = other[:, 0] * 0.5 + drums[:, 0] * 0.2

        # SR (Surround Right) - other R + ambient drums
        output_51[:, 5] = other[:, 1] * 0.5 + drums[:, 1] * 0.2

        # Normalize to prevent clipping
        max_val = np.max(np.abs(output_51))
        if max_val > 0.95:
            output_51 = output_51 * (0.95 / max_val)
            log("5.1-EXPORT", f"Normalized (peak was {max_val:.2f})")

        log("5.1-EXPORT", "[3/3] Encoding to E-AC-3 5.1...")

        temp_wav = os.path.join(tempfile.gettempdir(), f"{state.filename}_51.wav")
        sf.write(temp_wav, output_51, sr)

        output_name = f"{state.filename}_5.1_SURROUND.m4a"
        output_path = os.path.join(tempfile.gettempdir(), output_name)

        # Copy metadata from original if available
        if state.original_path and os.path.exists(state.original_path):
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_wav,
                '-i', state.original_path,
                '-map', '0:a',
                '-c:a', 'eac3',
                '-b:a', '640k',
                '-channel_layout', '5.1',
                '-map_metadata', '1',
                '-metadata:s:a:0', f'title={state.filename}',
                '-f', 'mp4',
                '-movflags', '+faststart',
                output_path
            ]
        else:
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_wav,
                '-c:a', 'eac3',
                '-b:a', '640k',
                '-channel_layout', '5.1',
                '-metadata', f'title={state.filename}',
                '-f', 'mp4',
                '-movflags', '+faststart',
                output_path
            ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        os.remove(temp_wav)

        if result.returncode != 0:
            log("5.1-EXPORT", f"ffmpeg failed: {result.stderr}")
            return None, f"Encoding failed: {result.stderr}"

        final_size = Path(output_path).stat().st_size / (1024 * 1024)
        log("5.1-EXPORT", f"Saved: {output_path} ({final_size:.1f} MB)")

        source_info = "with Audio Upscaler" if using_processed else "from ORIGINAL"
        return output_path, f"🔊 5.1 Surround exported {source_info}!\n\nFile: {output_name}\nSize: {final_size:.1f} MB\nFormat: E-AC-3 5.1 (VLC compatible)\n\nChannels:\n- FL/FR: Drums + instruments\n- FC: Vocals + bass\n- LFE: Sub-bass\n- SL/SR: Ambient/surround"

    except ImportError as e:
        log_error("5.1-EXPORT", e)
        return None, f"Missing dependency: {str(e)}"
    except Exception as e:
        log_error("5.1-EXPORT", e)
        return None, f"5.1 export error: {str(e)}"


# =============================================================================
# PRESETS
# =============================================================================

PRESETS = {
    "Custom": {
        "deesser": None,  # Don't change
        "stereo": None,
        "spatial": None,
        "description": "*Adjust sliders manually*"
    },
    "Spatial 3D": {
        "deesser": 0.3,
        "stereo": 1.0,
        "spatial": 0.7,
        "description": "🎧 **Spatial 3D** — Dolby Atmos-style binaural audio. Immersive 3D soundstage for headphones. Vocals center, instruments positioned around you."
    },
    "Studio Hip-Hop": {
        "deesser": 0.3,
        "stereo": 0.6,
        "spatial": 0.0,
        "description": "🎤 **Studio Hip-Hop** — Centered mix like Juice WRLD/Travis Scott. Punchy 808s, clear vocals."
    },
    "Wide & Bright": {
        "deesser": 0.2,
        "stereo": 1.5,
        "spatial": 0.0,
        "description": "✨ **Wide & Bright** — Big stereo image for EDM/Pop. Airy highs, wide soundstage."
    },
    "Podcast/Voice": {
        "deesser": 0.8,
        "stereo": 0.8,
        "spatial": 0.0,
        "description": "🎙️ **Podcast/Voice** — Tamed sibilants, focused center. Clear speech."
    },
    "Default": {
        "deesser": 0.5,
        "stereo": 1.0,
        "spatial": 0.0,
        "description": "⚡ **Default** — Balanced starting point."
    }
}


def apply_preset(preset_name):
    """Apply a preset and return updated slider values + description."""
    preset = PRESETS.get(preset_name, PRESETS["Default"])

    deesser = preset["deesser"]
    stereo = preset["stereo"]
    spatial = preset.get("spatial", 0.0)
    desc = preset["description"]

    log("PRESET", f"Applied: {preset_name} (deesser={deesser}, stereo={stereo}, spatial={spatial})")

    # Return updates for sliders and description
    # If None, use gr.update() to keep current value
    return (
        gr.update(value=deesser) if deesser is not None else gr.update(),
        gr.update(value=stereo) if stereo is not None else gr.update(),
        gr.update(value=spatial) if spatial is not None else gr.update(),
        desc
    )


# =============================================================================
# GRADIO UI
# =============================================================================

def create_ui():
    """Create the Gradio interface."""

    css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .dark {
        background-color: #1a1a2e !important;
    }
    """

    with gr.Blocks() as app:

        gr.Markdown("""
        # 🚀 Audio Upscaler Audio Studio

        **Full AI Pipeline:** demucs-mlx (source separation) → Psychoacoustic Enhancement → Spatial 3D → De-essing → Mastering

        Upload any audio file and transform it into something that makes Hi-Res sound average.
        """)

        with gr.Row():
            # Left column: Controls
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Input")
                audio_input = gr.Audio(
                    label="Drop audio file here",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                load_btn = gr.Button("Load Audio", variant="primary")

                gr.Markdown("### ⚙️ Settings")
                preset_dropdown = gr.Dropdown(
                    choices=[
                        "Custom",
                        "Spatial 3D",
                        "Studio Hip-Hop",
                        "Wide & Bright",
                        "Podcast/Voice",
                        "Default"
                    ],
                    value="Custom",
                    label="Preset",
                    info="Quick presets based on genre/use case"
                )
                preset_info = gr.Markdown("*Adjust sliders manually or pick a preset above*", elem_classes=["preset-info"])

                deesser_strength = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.5, step=0.1,
                    label="De-esser Strength",
                    info="0.3-0.5 = clear/bright, 0.8-1.0 = balanced, >1.2 = aggressive"
                )
                stereo_width = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                    label="Stereo Width Multiplier",
                    info="<1.0 = centered (studio), >1.0 = wider"
                )
                spatial_depth = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.1,
                    label="🎧 Spatial 3D Depth",
                    info="0 = standard stereo, 0.5-0.8 = immersive 3D, 1.0 = maximum binaural"
                )

                process_btn = gr.Button("🚀 Process Audio Upscaler", variant="primary", size="lg")

                gr.Markdown("### 📤 Export")
                export_btn = gr.Button("📁 Export FLAC (Stereo)", size="sm")
                export_atmos_btn = gr.Button("🎬 Export Dolby Atmos 7.1.4 (ADM)", size="sm")
                export_atmos_final_btn = gr.Button("🍎 Export Dolby Atmos (Apple Music M4A)", size="sm", variant="primary")
                export_51_btn = gr.Button("🔊 Export 5.1 Surround (VLC compatible)", size="sm")
                export_binaural_btn = gr.Button("🎧 Export Binaural (Headphones)", size="sm")
                export_file = gr.File(label="Download")
                export_status = gr.Textbox(label="Status", interactive=False)

            # Middle column: Visualizations
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Visualization")

                with gr.Tabs():
                    with gr.TabItem("Spectrogram"):
                        spec_plot = gr.Plot(label="Spectrogram")

                    with gr.TabItem("Stereo Field"):
                        gonio_plot = gr.Plot(label="Goniometer")

                    with gr.TabItem("Comparison"):
                        compare_plot = gr.Plot(label="A/B Waveform")

                status_text = gr.Markdown("Ready to load audio...")

            # Right column: A/B Testing
            with gr.Column(scale=1):
                gr.Markdown("### 🎧 A/B Test")

                with gr.Row():
                    ab_original_btn = gr.Button("A: Original", size="sm")
                    ab_processed_btn = gr.Button("B: Processed", variant="primary", size="sm")

                ab_player = gr.Audio(
                    label="A/B Player",
                    interactive=False,
                    autoplay=False
                )

                gr.Markdown("### 📈 Live Preview")
                preview_audio = gr.Audio(
                    label="Processed Preview",
                    interactive=False,
                    autoplay=False
                )

        # Event handlers
        preset_dropdown.change(
            fn=apply_preset,
            inputs=[preset_dropdown],
            outputs=[deesser_strength, stereo_width, spatial_depth, preset_info]
        )

        load_btn.click(
            fn=load_audio,
            inputs=[audio_input],
            outputs=[ab_player, spec_plot, gonio_plot, status_text]
        )

        process_btn.click(
            fn=process_audio,
            inputs=[deesser_strength, stereo_width, spatial_depth],
            outputs=[preview_audio, spec_plot, gonio_plot, compare_plot, status_text]
        )

        ab_original_btn.click(fn=get_original, outputs=[ab_player])
        ab_processed_btn.click(fn=get_processed, outputs=[ab_player])

        export_btn.click(
            fn=export_audio,
            outputs=[export_file, export_status]
        )

        export_atmos_btn.click(
            fn=export_dolby_atmos,
            outputs=[export_file, export_status]
        )

        export_atmos_final_btn.click(
            fn=export_atmos_final,
            outputs=[export_file, export_status]
        )

        export_51_btn.click(
            fn=export_51_surround,
            outputs=[export_file, export_status]
        )

        export_binaural_btn.click(
            fn=export_binaural,
            outputs=[export_file, export_status]
        )

    return app


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Audio Upscaler Audio Studio")
    print("=" * 60)
    print("\nStarting server at http://localhost:7860")
    print("Press Ctrl+C to stop\n")

    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
