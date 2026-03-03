# PRT EEG Experiment

Prosody Recognition Task (PRT) EEG experiment repository. Contains audio stimuli preprocessing tools, experiment presentation scripts, and EEG quality control tools for research on emotional prosody (happy/sad).

## Repository Structure

```
PRT_EEG/
├── code/
│   ├── stimuli_preprocessing/         # Audio analysis and preprocessing tools
│   ├── experiment/                    # Experiment presentation scripts
│   │   ├── prt_click_presentation.py  # Click train presentation & sound check
│   │   └── prt_story_presentation.py  # Story presentation & comprehension questions
│   ├── click_QC/                      # Click ABR quality control
│   │   └── click_qc.py               # CLI tool for ABR signal QC (run after recording)
│   └── analysis/                      # Analysis scripts
│       └── check_click_quality.py     # Original ABR analysis (reference)
├── stimuli/                           # Root-level audio stimuli files (.wav)
├── stim_normalized/                   # Normalized stimuli used by the experiment
│   └── click/                         # Click WAV files (click000.wav - click004.wav)
├── data/                              # Experiment data (not tracked in git)
└── docs/plans/                        # Design docs and implementation plans
```

## Setup

### Python Dependencies

Install the required packages:

```bash
pip install numpy scipy matplotlib mne expyfun librosa pandas sounddevice pybv
```

**Note:** `expyfun` and `sounddevice` are only needed on the EEG computer. `librosa` and `pandas` are only needed for audio preprocessing.

## Experiment Workflow

The experiment session follows this order:

1. **Click presentation** — run `prt_click_presentation.py <pid> <session>` (sound check + 5 minutes of click trains)
2. **Click QC** — run `click_qc.py` on the recorded data to verify ABR signal quality
3. **Story experiment** — run `prt_story_presentation.py <pid> <session>` (~30 min of stories + questions)

---

## Click Presentation

**Script:** `code/experiment/prt_click_presentation.py`

Presents click trains for ABR recording. Run this BEFORE the main story experiment.

### What it does

1. **Sound check** — plays 10 seconds of a story segment so the participant can confirm they hear it clearly
2. **Click trains** — presents 5 click WAV files (~1 minute each) with a fixation cross on screen
3. **End prompt** — confirms completion

### How to run

Run from the EEG computer with the experiment environment activated:

```bash
python code/experiment/prt_click_presentation.py <participant_id> <session>
```

**Example:**

```bash
python code/experiment/prt_click_presentation.py 12544 01
```

| Argument | Description | Example |
|----------|-------------|---------|
| `participant_id` | Participant ID number | `12544` |
| `session` | Session number | `01` |

### Important notes

- Stimulus files must be in `stim_normalized/click/` (click000.wav through click004.wav)
- Sound check file: `stim_normalized/12008_1_1_happy/story/12008_1_1_happy_studio.wav`
- Stimulus volume is set to 65 dB
- Press **Space** to advance between screens
- Press **End** to force quit

---

## Click ABR Quality Control

**Script:** `code/click_QC/click_qc.py`

Run this immediately after the click recording session to check ABR signal quality before continuing with the story experiment.

### How to run

```bash
python code/click_QC/click_qc.py <path_to_vhdr_file>
```

**Example:**

```bash
python code/click_QC/click_qc.py data/subject01/subject01_clicks.vhdr
```

If the click WAV files are not in the default location (`stim_normalized/click/`), specify the path:

```bash
python code/click_QC/click_qc.py data/subject01/subject01_clicks.vhdr --stim_path /path/to/click/files
```

### What it checks

#### 1. ABR Signal Quality

Derives the Auditory Brainstem Response (ABR) via cross-correlation of EEG with click stimuli, then evaluates:

- **SNR** (Signal-to-Noise Ratio) using the method from [Shan et al. (2023), *Scientific Reports*](https://www.nature.com/articles/s41598-023-50438-0)
- **ABR peak detection** for Waves I, III, and V

**Quality ratings based on SNR:**

| SNR | Rating |
|-----|--------|
| > 6 dB | GOOD |
| 3–6 dB | ACCEPTABLE |
| 0–3 dB | MARGINAL |
| < 0 dB | POOR |

#### 2. Scalp EEG Channel Quality

Checks all scalp EEG channels (excluding ABR-specific and Audio channels) for noise levels:

- Bandpass filters to 1–100 Hz, then computes standard deviation per channel
- **Flat** (< 0.5 uV): channel may have a bad connection
- **Noisy** (> 75 uV): channel has excessive noise

### Output files

Both files are saved alongside the input `.vhdr` file:

1. **`{basename}_abr_qc.png`** — Two-panel figure:
   - Top: ABR waveform (−10 to 15 ms) with Wave I/III/V peak annotations and SNR text box
   - Bottom: Horizontal bar chart of per-channel standard deviation with flat/noisy thresholds

2. **`{basename}_abr_qc.txt`** — Full text report including:
   - SNR value and variance components
   - Peak latencies and amplitudes for Waves I, III, V
   - Overall quality rating
   - Scalp EEG channel table with status flags

### What to look for

- **SNR should be at least 3 dB** (ACCEPTABLE or better) to proceed with the story experiment
- **Wave V should be clearly visible** around 5–9 ms
- **All scalp channels should be OK** — if channels are flagged as FLAT or NOISY, check electrode connections before proceeding

---

## Story Presentation

**Script:** `code/experiment/prt_story_presentation.py`

Presents 8 emotional prosody stories (~30 min total) with comprehension questions. Run this AFTER the click QC passes.

### How to run

```bash
python code/experiment/prt_story_presentation.py <participant_id> <session>
```

**Example:**

```bash
python code/experiment/prt_story_presentation.py 12544 01
```

| Argument | Description | Example |
|----------|-------------|---------|
| `participant_id` | Participant ID number | `12544` |
| `session` | Session number | `01` |

### Stimulus pool

All participants receive the same 8 stories (~29.5 min total audio):

| Story | Emotion | Duration |
|-------|---------|----------|
| 12008_1_1_sad | sad | 5.6 min |
| 12008_1_2_happy | happy | 4.7 min |
| 12008_1_2_sad | sad | 2.4 min |
| 12014_1_2_happy | happy | 2.3 min |
| 12015_1_2_sad | sad | 2.4 min |
| 12016_1_1_happy | happy | 6.9 min |
| 12016_1_2_happy | happy | 3.5 min |
| 9227_3_1_spontaneous | spontaneous | 1.9 min |

Each story is followed by 5 comprehension questions (3 multiple-choice + 2 free response). The stimulus pool is defined in `code/stimuli_preprocessing/story_questions_mapping_pool.csv`.

### What it does

1. **Instructions** — 2 instruction screens explaining the task
2. **For each story:**
   - Displays "Story X of 8", press Space to begin
   - Plays story audio with fixation cross on screen
   - Presents 5 questions: audio plays while question text and answer options are displayed
   - Experimenter presses Space after participant responds verbally
3. **End prompt** — thanks the participant

### Important notes

- Stimulus files must be in `stim_normalized/` with the directory structure matching the CSV paths
- Experimenter controls pacing with **Space** between stories and after each question
- Press **End** to force quit
- The script allows starting from any story number (prompted at launch)

---

## Audio Stimuli

### Naming Convention

Audio files follow the pattern: `{SubjectID}_{session}_{emotion}_{device}_q{number}.wav`

| Field | Values | Example |
|-------|--------|---------|
| SubjectID | speaker ID number | 12008 |
| session | 1 or 2 | 1 |
| emotion | happy or sad | happy |
| device | nvidia or studio | studio |
| q{number} | question 1–5 | q3 |

### Audio Analysis

To analyze WAV files in the `stimuli/` directory:

```bash
python3 code/stimuli_preprocessing/analyze_wav_files.py
```

This extracts format info, duration, RMS, peak amplitude, pitch (F0), and spectral features, and saves results to `wav_analysis_results.csv`.

---

## EEG Recording Parameters

| Parameter | Value |
|-----------|-------|
| EEG sampling rate | 25,000 Hz |
| Stimulus sampling rate | 48,000 Hz |
| Click rate | 40 Hz |
| Click duration | ~60s per file (5 files total) |
| Stimulus volume | 65 dB |
| ABR channels | Plus_R, Minus_R, Plus_L, Minus_L |
| File format | BrainVision (.vhdr/.vmrk/.eeg) |

### ABR Channel Re-referencing

- Right ear: Plus_R − Minus_R → EP1
- Left ear: Plus_L − Minus_L → EP2
