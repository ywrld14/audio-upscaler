# Audio Upscaler

AI-powered audio enhancement pipeline with stem separation, psychoacoustic processing, and spatial audio.

## Architecture

```mermaid
flowchart TB
    subgraph INPUT["📁 Input"]
        A[Audio File<br/>mp3/m4a/wav/flac]
    end

    subgraph CONVERT["🔄 Format Conversion"]
        B[FFmpeg<br/>→ WAV 44.1kHz]
    end

    subgraph SEPARATION["🎛️ AI Stem Separation"]
        C[demucs-mlx<br/>Apple Silicon GPU]
        C --> D[🎤 Vocals]
        C --> E[🥁 Drums]
        C --> F[🎸 Bass]
        C --> G[🎹 Other]
    end

    subgraph ENHANCE["✨ Psychoacoustic Enhancement"]
        H[Harmonic Exciter<br/>3-5kHz overtones]
        I[Sub-Harmonic Synth<br/>80Hz bass extension]
        J[Tube Warmth<br/>Even harmonics]
    end

    subgraph SPATIAL["🌐 Spatial Processing"]
        K{Mode?}
        K -->|Standard| L[Stereo Width<br/>M/S Processing]
        K -->|Spatial 3D| M[HRTF Binaural<br/>ITD + ILD]
        M --> N[Vocals: 0° center]
        M --> O[Drums: rear reflections]
        M --> P[Other: ±40° sides]
        M --> Q[Early Reflections]
    end

    subgraph DEESS["🔇 De-Esser"]
        R[FFT Analysis<br/>4-9kHz sibilance]
        R --> S[Dynamic Attenuation<br/>Threshold-based]
    end

    subgraph MASTER["🎚️ Mastering"]
        T[Loudness Normalization<br/>-14 LUFS target]
        T --> U[Soft Limiter<br/>-1 dBTP ceiling]
    end

    subgraph EXPORT["📤 Export Formats"]
        V[FLAC Stereo<br/>Lossless]
        W[Binaural FLAC<br/>Headphone optimized]
        X[5.1 Surround<br/>E-AC-3 640kbps]
        Y[Dolby Atmos<br/>E-AC-3 JOC]
    end

    A --> B --> C
    D --> H
    E --> H
    F --> I
    G --> J
    H --> K
    I --> K
    J --> K
    L --> R
    Q --> R
    S --> T
    U --> V
    U --> W
    U --> X
    U --> Y

    style INPUT fill:#e1f5fe
    style SEPARATION fill:#fff3e0
    style ENHANCE fill:#f3e5f5
    style SPATIAL fill:#e8f5e9
    style DEESS fill:#fce4ec
    style MASTER fill:#fff8e1
    style EXPORT fill:#e0f2f1
```

## Processing Pipeline

```mermaid
sequenceDiagram
    participant U as User
    participant G as Gradio UI
    participant P as Pipeline
    participant D as Demucs-MLX
    participant F as FFmpeg

    U->>G: Upload audio
    G->>P: Start processing
    P->>F: Convert to WAV
    F-->>P: WAV data
    P->>D: Separate stems
    Note over D: GPU accelerated<br/>~10-15 seconds
    D-->>P: vocals, drums, bass, other
    P->>P: Psychoacoustic enhancement
    P->>P: Spatial 3D / Stereo processing
    P->>P: De-essing
    P->>P: Mastering & normalization
    P-->>G: Processed audio
    G-->>U: Preview + A/B comparison
    U->>G: Export
    G->>F: Encode (FLAC/E-AC-3/Atmos)
    F-->>G: Output file
    G-->>U: Download
```

## 5.1 Surround Channel Mapping

```mermaid
flowchart LR
    subgraph STEMS["Separated Stems"]
        V[Vocals]
        D[Drums]
        B[Bass]
        O[Other]
    end

    subgraph CHANNELS["5.1 Channels"]
        FL[FL: Front Left<br/>Drums + Other + Vocals]
        FR[FR: Front Right<br/>Drums + Other + Vocals]
        FC[FC: Center<br/>Vocals mono + Bass]
        LFE[LFE: Subwoofer<br/>Bass low-passed 120Hz]
        SL[SL: Surround Left<br/>Other + Drums ambient]
        SR[SR: Surround Right<br/>Other + Drums ambient]
    end

    V --> FC
    V --> FL
    V --> FR
    D --> FL
    D --> FR
    D --> SL
    D --> SR
    B --> FC
    B --> LFE
    O --> FL
    O --> FR
    O --> SL
    O --> SR

    style FC fill:#ffeb3b
    style LFE fill:#ff5722
```

## Features

- **Stem Separation**: Isolate vocals, drums, bass, and instruments using demucs-mlx (Apple Silicon optimized)
- **Psychoacoustic Enhancement**: Harmonic exciter, sub-harmonic synthesis, tube warmth
- **Spatial 3D Processing**: HRTF binaural positioning for immersive headphone listening
- **De-esser**: FFT-based sibilance reduction calibrated to professional standards
- **Multi-format Export**: FLAC, Binaural, 5.1 Surround (E-AC-3), Dolby Atmos

## Presets

| Preset | Use Case |
|--------|----------|
| Studio Hip-Hop | Punch and clarity for hip-hop/rap (narrow stereo, -9 LUFS) |
| Spatial 3D | Immersive 3D soundstage for headphones |
| Wide & Bright | Enhanced stereo width and high frequencies |
| Podcast/Voice | Optimized for spoken word |

## Installation

```bash
# Clone
git clone https://github.com/ywrld14/audio-upscaler.git
cd audio-upscaler

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Launch GUI
python audio_studio.py

# Open http://127.0.0.1:7860
```

## Requirements

- Python 3.10+
- macOS with Apple Silicon (for demucs-mlx acceleration)
- ffmpeg

## Dependencies

- `demucs-mlx` - AI stem separation
- `gradio` - Web UI
- `soundfile` - Audio I/O
- `scipy` - Signal processing
- `numpy` - Numerical operations
- `pyloudnorm` - Loudness metering
- `ac3forge` - Dolby Atmos encoding

## License

MIT
