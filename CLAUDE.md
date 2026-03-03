# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a PRT (Prosody Recognition Task) EEG experiment repository containing audio stimuli preprocessing tools and experimental materials. The repository handles emotional prosody-based audio stimuli (happy/sad) for EEG research.

## Repository Structure

```
PRT_EEG/
├── code/
│   ├── stimuli_preprocessing/    # Audio analysis and preprocessing tools
│   ├── experiment/               # Experiment presentation scripts
│   │   └── prt_click_presentation.py  # Click train presentation & sound check
│   ├── click_QC/                 # Click ABR quality control
│   │   └── click_qc.py          # CLI tool for ABR signal QC
│   └── analysis/                 # Analysis scripts
│       └── check_click_quality.py  # Original ABR analysis (reference)
├── stimuli/                      # Root-level audio stimuli files (.wav)
├── stim_normalized/              # Normalized stimuli used by experiment
│   └── click/                    # Click WAV files (click000.wav, click001.wav, ...)
├── data/                         # Experiment data (gitignored)
│   └── protocol_dev_data/        # Pilot/protocol development data
├── docs/plans/                   # Design docs and implementation plans
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
- `scipy` - Scientific computing (wavfile reading, signal processing)
- `matplotlib` - Plotting (ABR QC figures)
- `mne` - EEG data loading and processing
- `expyfun` - Experiment control and WAV I/O (available on EEG computer)
- `pybv` - BrainVision file export (for testing/data extraction)
- `sounddevice` - Audio output for experiment presentation

Standard library: `wave`, `os`, `pathlib`, `glob`, `argparse`, `datetime`

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

## Click ABR Quality Control

### Purpose
CLI tool run immediately after click recording on the EEG computer to verify ABR signal quality before continuing the experiment.

### Usage
```bash
python click_qc.py <path_to_vhdr_file> [--stim_path <click_dir>]
```

### Processing Pipeline
1. Load EEG (.vhdr) via MNE
2. Scalp EEG channel QC (bandpass 1-100 Hz, then check std dev: flat < 0.5 uV, noisy > 75 uV)
3. Pick ABR channels (Plus_R/Minus_R, Plus_L/Minus_L), re-reference
4. High-pass filter (1 Hz) + notch filters (60 Hz harmonics)
5. Epoch to click events (with trigger deduplication - see below)
6. Load click .wav files, build pulse trains via rising-edge detection
7. Cross-correlate (FFT domain) to derive ABR
8. Compute SNR (Shan et al. 2023 method)
9. Identify ABR peaks (Waves I, III, V)

### Trigger Deduplication
The click presentation script (`prt_click_presentation.py`) sends TWO S1 triggers per click train:
- `ec.start_stimulus()` sends the FIRST S1 (actual audio onset)
- `ec.stamp_triggers([1])` sends the SECOND S1 ~0.1s later
The QC script keeps the FIRST trigger in each pair (minimum 1s gap between valid triggers).

### SNR Calculation (Shan et al. 2023, Scientific Reports)
- sigma^2_S+N = variance of ABR in [0, 15] ms
- sigma^2_N = mean variance of 15 ms segments in [-200, -20] ms baseline
- SNR = 10 * log10[(sigma^2_S+N - sigma^2_N) / sigma^2_N]
- Reports "insufficient signal" if sigma^2_S+N <= sigma^2_N

### ABR Peak Detection Windows
- Wave I: [0, 5] ms
- Wave III: [3.5, 5.5] ms
- Wave V: [5, 9] ms
- All three waves are always annotated on the plot

### Quality Thresholds
- SNR < 0 dB: POOR
- SNR 0-3 dB: MARGINAL
- SNR 3-6 dB: ACCEPTABLE
- SNR > 6 dB: GOOD

### Scalp EEG Channel QC
- Bandpass filter 1-100 Hz before computing std dev
- Flat threshold: < 0.5 uV
- Noisy threshold: > 75 uV
- Excludes ABR channels (Plus_R, Minus_R, Plus_L, Minus_L) and Audio channel

### Output
- Figure (`{basename}_abr_qc.png`): 2 panels - ABR waveform [-10, 15] ms with peak annotations + SNR text box, and per-channel std dev bar chart
- Text report (`{basename}_abr_qc.txt`): SNR, peak latencies/amplitudes, epoch count, quality summary, scalp channel table
- Both saved alongside the input .vhdr file

### EEG Recording Parameters
- EEG sampling rate: 25000 Hz
- Stimulus sampling rate: 48000 Hz
- Click rate: 40 Hz
- Click trial length: ~60s per file (5 files = 5 minutes total)
- Stimulus volume: 65 dB
- ABR channels: Plus_R, Minus_R, Plus_L, Minus_L (re-referenced to EP1, EP2)
- Data scaling: divide by 100 for microvolts

### Dependencies
- numpy, scipy, matplotlib, mne, expyfun (available on EEG computer)
- pybv (for BrainVision export, used in testing)

## Click Presentation Script

### Location
`code/experiment/prt_click_presentation.py`

### Usage
```bash
python prt_click_presentation.py <participant_id> <session>
```
- `participant_id`: Participant ID (e.g., 12544)
- `session`: Session number (e.g., 01)

Both arguments are passed to `ExperimentController(participant=pid, session=session)` for expyfun logging/output naming.

### Workflow
1. Sound check: plays 10s of a story segment for volume verification
2. Click trains: presents 5 click WAV files (click000.wav to click004.wav) sequentially
3. End prompt

### Stimulus Path
Uses `stim_normalized/` directory (relative to project root) for both click files and sound check audio.

## Story Presentation Script

### Location
`code/experiment/prt_story_presentation.py`

### Usage
```bash
python prt_story_presentation.py <participant_id> <session>
```
- `participant_id`: Participant ID (e.g., 12544)
- `session`: Session number (e.g., 01)

### Stimulus Pool
All participants receive the same 8 stories (~29.5 min total). No pre/post session split.

Story pool is defined in `code/stimuli_preprocessing/story_questions_mapping_pool.csv`:
- 12008_1_1_sad (5.6 min)
- 12008_1_2_happy (4.7 min)
- 12008_1_2_sad (2.4 min)
- 12014_1_2_happy (2.3 min)
- 12015_1_2_sad (2.4 min)
- 12016_1_1_happy (6.9 min)
- 12016_1_2_happy (3.5 min)
- 9227_3_1_spontaneous (1.9 min)

Each story has 5 comprehension questions (3 multiple-choice + 2 free response).

### Workflow
1. Instructions (3 screens)
2. For each story: play audio with fixation cross → 5 questions with audio + visual display
3. End prompt

### Key Design Decisions
- The old `story_questions_mapping_fin.csv` had `assigned_session` column (0=pre, 1=post). The new `story_questions_mapping_pool.csv` removes this — everyone gets the same pool.
- Stories were selected as: all `AssignedSession=0` from `partitioned_story.csv` + `12008_1_1_sad` + `12008_1_2_happy`.
