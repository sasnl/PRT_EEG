# Click QC Tool Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a CLI tool at `code/click_QC/click_qc.py` that derives click ABR, computes SNR, detects peaks, and outputs a figure + text report for immediate quality control on the EEG computer.

**Architecture:** Single-file script based on `code/analysis/check_click_quality.py`. Adds three new capabilities on top of the existing ABR derivation pipeline: (1) SNR calculation per Shan et al. 2023, (2) ABR peak detection for Waves I/III/V, (3) structured text report saved to file. The existing filtering/epoching/cross-correlation logic is reused as-is.

**Tech Stack:** Python 3, numpy, scipy, matplotlib, mne, expyfun

---

### Task 1: Create directory and scaffold script with CLI + ABR pipeline

**Files:**
- Create: `code/click_QC/click_qc.py`

**Step 1: Create the directory**

Run: `mkdir -p code/click_QC`

**Step 2: Write the full script**

Copy the existing ABR derivation pipeline from `code/analysis/check_click_quality.py` into `code/click_QC/click_qc.py` with these modifications:

1. Keep all imports, filtering functions, CLI argument parsing, EEG loading, channel picking, re-referencing, preprocessing, epoching, click loading, and cross-correlation exactly as-is.
2. Change `t_start, t_stop` from `(-200e-3, 600e-3)` to `(-200e-3, 200e-3)` — we still need the full [-200, 200] ms window for SNR calculation, but the plot will only show [-10, 15] ms.
3. Remove `plt.show()` (headless-friendly for EEG computer).
4. Add placeholder functions for `compute_snr()`, `detect_abr_peaks()`, `generate_report()` that will be implemented in subsequent tasks.

The script should have this structure:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Click ABR Quality Control Script
...
Usage:
    python click_qc.py <path_to_vhdr_file> [--stim_path <path_to_clicks>]
"""

import argparse
import os
import sys
import numpy as np
import scipy.signal as signal
from numpy.fft import fft, ifft
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for EEG computer
import matplotlib.pyplot as plt
from expyfun.io import read_wav
import mne
import glob
from datetime import datetime

# --- Filtering functions (identical to check_click_quality.py lines 29-61) ---

def butter_highpass(cutoff, fs, order=1):
    ...

def butter_highpass_filter(data, cutoff, fs, order=1):
    ...

def butter_lowpass(cutoff, fs, order=1):
    ...

def butter_lowpass_filter(data, cutoff, fs, order=1):
    ...

def butter_bandpass(lowcut, highcut, fs, order=1):
    ...

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    ...

# --- New functions ---

def compute_snr(abr_response, lags):
    """Compute SNR using Shan et al. (2023) method.

    SNR = 10 * log10[(sigma^2_S+N - sigma^2_N) / sigma^2_N]

    - sigma^2_S+N: variance of ABR in [0, 15] ms
    - sigma^2_N: mean variance of 15 ms segments in [-200, -20] ms

    Returns:
        snr_db (float or None): SNR in dB, or None if insufficient signal
        sigma_sn (float): signal+noise variance
        sigma_n (float): noise variance
    """
    pass  # Task 2

def detect_abr_peaks(abr_response, lags):
    """Detect ABR Waves I, III, V in expected latency ranges.

    Search windows:
        Wave I:   [1.0, 2.5] ms
        Wave III: [3.0, 4.5] ms
        Wave V:   [4.5, 7.0] ms

    Returns:
        dict with keys 'I', 'III', 'V', each containing:
            'latency_ms': peak latency in ms
            'amplitude_uv': peak amplitude in uV
            'found': bool
    """
    pass  # Task 3

def generate_report(eeg_file, n_epochs, eeg_fs, snr_db, sigma_sn, sigma_n, peaks):
    """Generate text report string for terminal and file output.

    Returns:
        report (str): formatted report
    """
    pass  # Task 4

def plot_abr(abr_response, lags, peaks, snr_db, eeg_file, output_png):
    """Plot ABR waveform [-10, 15] ms with peak annotations and SNR text box.

    Saves to output_png.
    """
    pass  # Task 5

def main():
    # --- Argument parsing (same as check_click_quality.py) ---
    # --- EEG loading, channel picking, re-referencing, preprocessing ---
    # --- Epoching, click loading, cross-correlation ---
    # --- (all identical to check_click_quality.py lines 63-291) ---

    # --- NEW: SNR, peaks, report, plot ---
    snr_db, sigma_sn, sigma_n = compute_snr(abr_response, lags)
    peaks = detect_abr_peaks(abr_response, lags)
    report = generate_report(eeg_vhdr, n_epoch_click, eeg_fs, snr_db, sigma_sn, sigma_n, peaks)

    # Print to terminal
    print(report)

    # Save report to text file
    output_dir = os.path.dirname(os.path.abspath(eeg_vhdr))
    output_base = os.path.splitext(os.path.basename(eeg_vhdr))[0]
    output_txt = os.path.join(output_dir, f"{output_base}_abr_qc.txt")
    with open(output_txt, 'w') as f:
        f.write(report)
    print(f"Report saved to: {output_txt}")

    # Plot
    output_png = os.path.join(output_dir, f"{output_base}_abr_qc.png")
    plot_abr(abr_response, lags, peaks, snr_db, eeg_vhdr, output_png)
    print(f"Figure saved to: {output_png}")

if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add code/click_QC/click_qc.py
git commit -m "feat: scaffold click QC script with ABR pipeline from check_click_quality.py"
```

---

### Task 2: Implement `compute_snr()`

**Files:**
- Modify: `code/click_QC/click_qc.py` (the `compute_snr` function)

**Step 1: Implement the SNR function**

Replace the `pass` in `compute_snr()` with:

```python
def compute_snr(abr_response, lags):
    """Compute SNR using Shan et al. (2023) method."""
    # Signal + Noise window: [0, 15] ms
    sn_mask = (lags >= 0) & (lags <= 15)
    sigma_sn = np.var(abr_response[sn_mask])

    # Noise window: [-200, -20] ms, segmented into 15 ms chunks
    noise_mask = (lags >= -200) & (lags <= -20)
    noise_data = abr_response[noise_mask]

    # Segment into 15 ms chunks
    # Calculate samples per 15 ms segment
    dt = lags[1] - lags[0]  # ms per sample
    samples_per_segment = int(15.0 / dt)

    # Split noise into segments and compute variance of each
    n_full_segments = len(noise_data) // samples_per_segment
    noise_variances = []
    for i in range(n_full_segments):
        segment = noise_data[i * samples_per_segment : (i + 1) * samples_per_segment]
        noise_variances.append(np.var(segment))

    sigma_n = np.mean(noise_variances)

    # Compute SNR
    if sigma_sn <= sigma_n:
        snr_db = None  # Insufficient signal
    else:
        snr_db = 10 * np.log10((sigma_sn - sigma_n) / sigma_n)

    return snr_db, sigma_sn, sigma_n
```

**Step 2: Commit**

```bash
git add code/click_QC/click_qc.py
git commit -m "feat: implement SNR calculation per Shan et al. 2023"
```

---

### Task 3: Implement `detect_abr_peaks()`

**Files:**
- Modify: `code/click_QC/click_qc.py` (the `detect_abr_peaks` function)

**Step 1: Implement peak detection**

Replace the `pass` in `detect_abr_peaks()` with:

```python
def detect_abr_peaks(abr_response, lags):
    """Detect ABR Waves I, III, V in expected latency ranges."""
    # Expected latency windows (ms)
    wave_windows = {
        'I':   (1.0, 2.5),
        'III': (3.0, 4.5),
        'V':   (4.5, 7.0),
    }

    peaks = {}
    for wave_name, (t_min, t_max) in wave_windows.items():
        mask = (lags >= t_min) & (lags <= t_max)
        if not np.any(mask):
            peaks[wave_name] = {'latency_ms': None, 'amplitude_uv': None, 'found': False}
            continue

        segment = abr_response[mask]
        segment_lags = lags[mask]

        # Find the positive peak (ABR waves are typically vertex-positive)
        peak_idx = np.argmax(segment)
        peaks[wave_name] = {
            'latency_ms': segment_lags[peak_idx],
            'amplitude_uv': segment[peak_idx],
            'found': True,
        }

    return peaks
```

**Step 2: Commit**

```bash
git add code/click_QC/click_qc.py
git commit -m "feat: implement ABR peak detection for Waves I, III, V"
```

---

### Task 4: Implement `generate_report()`

**Files:**
- Modify: `code/click_QC/click_qc.py` (the `generate_report` function)

**Step 1: Implement the report generator**

Replace the `pass` in `generate_report()` with:

```python
def generate_report(eeg_file, n_epochs, eeg_fs, snr_db, sigma_sn, sigma_n, peaks):
    """Generate text report for terminal and file output."""
    lines = []
    lines.append("=" * 60)
    lines.append("         CLICK ABR QUALITY CONTROL REPORT")
    lines.append("=" * 60)
    lines.append(f"Date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"EEG File:       {os.path.basename(eeg_file)}")
    lines.append(f"Sampling Rate:  {eeg_fs} Hz")
    lines.append(f"Epochs Used:    {n_epochs}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  SNR Analysis")
    lines.append("-" * 60)
    lines.append(f"  sigma^2_S+N (signal+noise var):  {sigma_sn:.6e}")
    lines.append(f"  sigma^2_N   (noise var):         {sigma_n:.6e}")
    if snr_db is not None:
        lines.append(f"  SNR:                             {snr_db:.2f} dB")
    else:
        lines.append(f"  SNR:                             N/A (insufficient signal)")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  ABR Peak Detection")
    lines.append("-" * 60)
    for wave_name in ['I', 'III', 'V']:
        p = peaks[wave_name]
        if p['found']:
            lines.append(f"  Wave {wave_name:>3s}:  {p['latency_ms']:.2f} ms  |  {p['amplitude_uv']:.4f} uV")
        else:
            lines.append(f"  Wave {wave_name:>3s}:  not detected")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  Quality Summary")
    lines.append("-" * 60)

    # Quality assessment
    if snr_db is None:
        quality = "POOR - No detectable ABR signal"
    elif snr_db < 0:
        quality = "POOR - SNR below 0 dB"
    elif snr_db < 3:
        quality = "MARGINAL - SNR between 0-3 dB"
    elif snr_db < 6:
        quality = "ACCEPTABLE - SNR between 3-6 dB"
    else:
        quality = "GOOD - SNR above 6 dB"

    wave_v = peaks.get('V', {})
    if wave_v.get('found', False):
        quality += f"\n  Wave V detected at {wave_v['latency_ms']:.2f} ms"
    else:
        quality += "\n  WARNING: Wave V not clearly detected"

    lines.append(f"  {quality}")
    lines.append("=" * 60)

    return "\n".join(lines)
```

**Step 2: Commit**

```bash
git add code/click_QC/click_qc.py
git commit -m "feat: implement QC report generation with SNR and peak summary"
```

---

### Task 5: Implement `plot_abr()`

**Files:**
- Modify: `code/click_QC/click_qc.py` (the `plot_abr` function)

**Step 1: Implement the plotting function**

Replace the `pass` in `plot_abr()` with:

```python
def plot_abr(abr_response, lags, peaks, snr_db, eeg_file, output_png):
    """Plot ABR waveform [-10, 15] ms with peak annotations and SNR text box."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot full waveform in view range
    plot_mask = (lags >= -10) & (lags <= 15)
    ax.plot(lags[plot_mask], abr_response[plot_mask], 'k-', linewidth=1.2)

    # Annotate peaks
    colors = {'I': 'red', 'III': 'blue', 'V': 'green'}
    for wave_name in ['I', 'III', 'V']:
        p = peaks[wave_name]
        if p['found']:
            ax.plot(p['latency_ms'], p['amplitude_uv'], 'o',
                    color=colors[wave_name], markersize=8, zorder=5)
            ax.annotate(f"Wave {wave_name}",
                        xy=(p['latency_ms'], p['amplitude_uv']),
                        xytext=(5, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        color=colors[wave_name])

    # SNR text box
    if snr_db is not None:
        snr_text = f"SNR = {snr_db:.2f} dB"
    else:
        snr_text = "SNR = N/A\n(insufficient signal)"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.95, snr_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=props)

    ax.set_xlim([-10, 15])
    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_ylabel('Amplitude (uV)', fontsize=12)
    ax.set_title(f'Click ABR - {os.path.basename(eeg_file)}', fontsize=14)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Stimulus onset')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
```

**Step 2: Commit**

```bash
git add code/click_QC/click_qc.py
git commit -m "feat: implement ABR plot with peak annotations and SNR text box"
```

---

### Task 6: Final integration and cleanup

**Files:**
- Modify: `code/click_QC/click_qc.py` (the `main` function — wire everything together)

**Step 1: Verify `main()` wires all components together**

Ensure the `main()` function, after computing `abr_response` and `lags` from the cross-correlation pipeline, calls:

```python
    # Compute SNR
    snr_db, sigma_sn, sigma_n = compute_snr(abr_response, lags)

    # Detect peaks
    peaks = detect_abr_peaks(abr_response, lags)

    # Generate and print report
    report = generate_report(eeg_vhdr, n_epoch_click, eeg_fs,
                             snr_db, sigma_sn, sigma_n, peaks)
    print(report)

    # Save report
    output_dir = os.path.dirname(os.path.abspath(eeg_vhdr))
    output_base = os.path.splitext(os.path.basename(eeg_vhdr))[0]

    output_txt = os.path.join(output_dir, f"{output_base}_abr_qc.txt")
    with open(output_txt, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {output_txt}")

    # Plot
    output_png = os.path.join(output_dir, f"{output_base}_abr_qc.png")
    plot_abr(abr_response, lags, peaks, snr_db, eeg_vhdr, output_png)
    print(f"Figure saved to: {output_png}")
```

**Step 2: Test CLI help**

Run: `python code/click_QC/click_qc.py --help`

Expected output:
```
usage: click_qc.py [-h] [--stim_path STIM_PATH] eeg_file

Analyze Click ABR Quality

positional arguments:
  eeg_file              Path to the EEG .vhdr file

optional arguments:
  --stim_path STIM_PATH
                        Path to the click stimuli directory
```

**Step 3: Commit**

```bash
git add code/click_QC/click_qc.py
git commit -m "feat: complete click QC tool with SNR, peaks, report, and figure"
```
