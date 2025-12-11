# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a PRT (Prosody Recognition Task) EEG experiment repository containing audio stimuli preprocessing tools and experimental materials. The repository handles emotional prosody-based audio stimuli (happy/sad) for EEG research.

## Repository Structure

```
PRT_EEG/
├── code/
│   ├── stimuli_preprocessing/    # Audio analysis and preprocessing tools
│   └── experiment/               # Empty directory for experiment code
├── stimuli/                      # Root-level audio stimuli files (.wav)
└── pilot_102025/                 # Pilot study data
    ├── code/                     # Empty directory for pilot-specific code
    └── stim/                     # Organized stimuli by ID, session, emotion
        └── {ID}_{session}_{emotion}/
            ├── story/            # Story audio files
            └── questions/        # Question audio files (q1-q5)
```

## Audio Stimuli Organization

### Naming Convention
Audio files follow the pattern: `{SubjectID}_{session}_{emotion}_{device}_q{number}.wav`
- SubjectID: e.g., 12008, 12015, 12016
- session: 1 or 2
- emotion: happy or sad
- device: nvidia or studio (recording device)
- q{number}: question number (1-5) for question files

### Directory Structure
- Story files: Located in `{ID}_{session}_{emotion}/story/`
- Question files: Located in `{ID}_{session}_{emotion}/questions/` with 5 questions per condition

## Commands

### Audio Analysis
Run audio analysis on WAV files from the repository root:
```bash
python3 code/stimuli_preprocessing/analyze_wav_files.py
```

This script:
- Analyzes all .wav files in the `stimuli/` directory
- Extracts format (sample rate, bit depth, channels), duration, RMS, peak amplitude
- Performs pitch analysis (F0 mean, range, variability using librosa piptrack)
- Calculates spectral features (centroid, zero-crossing rate)
- Outputs detailed console report and saves results to `wav_analysis_results.csv`

### Dependencies
Required Python packages (already installed):
- `librosa` - Audio analysis and pitch extraction
- `pandas` - Data processing and CSV export
- `numpy` - Numerical operations
- `scipy` - Scientific computing (wavfile reading)

Standard library: `wave`, `os`, `pathlib`

## Code Architecture

### Audio Analysis Pipeline
The `analyze_wav_file()` function in `code/stimuli_preprocessing/analyze_wav_files.py` performs multi-stage analysis:

1. **Basic Properties** (wave module): Sample rate, channels, bit depth, duration
2. **Audio Loading** (librosa): Loads audio with original sample rate, handles mono/stereo
3. **Amplitude Analysis**: RMS per channel, overall RMS (linear and dB), peak amplitude, dynamic range
4. **Temporal Features**: Zero-crossing rate (first channel)
5. **Spectral Features**: Spectral centroid
6. **Pitch Analysis** (librosa.piptrack):
   - F0 extraction with 50-500 Hz range (human voice)
   - Mean F0, standard deviation, min/max, range
   - Robust range (10th-90th percentile) for outlier resistance
   - Pitch variability metric

Returns dictionary with 18+ metrics per audio file.

### Key Implementation Details
- Mono audio converted to 2D array for consistent processing
- Pitch extraction uses magnitude threshold (0.1) to filter weak pitch estimates
- F0 statistics only calculated from valid (>0) pitch values
- Handles analysis errors gracefully with try/except and error reporting
