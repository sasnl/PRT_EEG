# Click QC Tool Design

## Purpose
CLI tool run immediately after click recording on the EEG computer to verify ABR signal quality before continuing the experiment.

## Location
`code/click_QC/click_qc.py`

## CLI Interface
```
python click_qc.py <path_to_vhdr_file> [--stim_path <click_dir>]
```

## Processing Pipeline
1. Load EEG (.vhdr) via MNE
2. Pick ABR channels (Plus_R/Minus_R, Plus_L/Minus_L), re-reference
3. High-pass filter (1 Hz) + notch filters (60 Hz harmonics)
4. Epoch to click events
5. Load click .wav files, build pulse trains
6. Cross-correlate (FFT domain) to derive ABR
7. Compute SNR (paper method)
8. Identify ABR peaks (Waves I, III, V)

## SNR Calculation
From Shan et al. (2023), Scientific Reports:
- sigma^2_S+N = variance of ABR in [0, 15] ms
- sigma^2_N = mean variance of 15 ms segments in [-200, -20] ms baseline
- SNR = 10 * log10[(sigma^2_S+N - sigma^2_N) / sigma^2_N]
- Report "insufficient signal" if sigma^2_S+N <= sigma^2_N

## ABR Peak Detection
- Wave I (~1.5 ms), Wave III (~3.5 ms), Wave V (~5.5 ms)
- Search within expected latency ranges

## Output

### Figure (one panel, saved as `{basename}_abr_qc.png`)
- ABR waveform, x-axis [-10, 15] ms
- Wave I/III/V peaks annotated
- SNR value in text box

### Terminal + Text File (`{basename}_abr_qc.txt`)
- SNR (dB)
- Peak latencies and amplitudes (Waves I, III, V)
- Epoch count
- Quality summary

## Dependencies
- numpy, scipy, matplotlib, mne, expyfun
