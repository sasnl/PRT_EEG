#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Click ABR Quality Control Script

This script analyzes EEG data collected during the click presentation session
to derive the Auditory Brainstem Response (ABR). It computes SNR, detects
ABR peaks (Waves I, III, V), and generates a QC report with plots.

Usage:
    python click_qc.py <path_to_vhdr_file> [--stim_path <path_to_clicks>]

Example:
    python click_qc.py ../../data/subject01/click_data.vhdr
"""

import argparse
import os
import numpy as np
import scipy.signal as signal
from numpy.fft import fft, ifft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
import mne
import glob
from datetime import datetime

# %% Define Filtering Functions

def butter_highpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

# %% New QC Functions

def compute_snr(abr_response, lags):
    """Compute SNR using Shan et al. (2023) method."""
    # Signal + Noise window: [0, 15] ms
    sn_mask = (lags >= 0) & (lags <= 15)
    sigma_sn = np.var(abr_response[sn_mask])

    # Noise window: [-200, -20] ms, segmented into 15 ms chunks
    noise_mask = (lags >= -200) & (lags <= -20)
    noise_data = abr_response[noise_mask]

    dt = lags[1] - lags[0]  # ms per sample
    samples_per_segment = int(15.0 / dt)

    n_full_segments = len(noise_data) // samples_per_segment
    if n_full_segments == 0:
        return None, sigma_sn, 0.0

    noise_variances = []
    for i in range(n_full_segments):
        segment = noise_data[i * samples_per_segment : (i + 1) * samples_per_segment]
        noise_variances.append(np.var(segment))

    sigma_n = np.mean(noise_variances)

    if sigma_n == 0 or sigma_sn <= sigma_n:
        snr_db = None
    else:
        snr_db = 10 * np.log10((sigma_sn - sigma_n) / sigma_n)

    return snr_db, sigma_sn, sigma_n


def detect_abr_peaks(abr_response, lags):
    """Detect ABR Waves I, III, V in expected latency ranges."""
    wave_windows = {
        'I':   (0.0, 5.0),
        'III': (3.5, 5.5),
        'V':   (5.0, 9.0),
    }

    peaks = {}
    for wave_name, (t_min, t_max) in wave_windows.items():
        mask = (lags >= t_min) & (lags <= t_max)
        if not np.any(mask):
            peaks[wave_name] = {'latency_ms': None, 'amplitude_uv': None, 'found': False}
            continue

        segment = abr_response[mask]
        segment_lags = lags[mask]

        peak_idx = np.argmax(segment)
        peak_amp = segment[peak_idx]

        peaks[wave_name] = {
            'latency_ms': segment_lags[peak_idx],
            'amplitude_uv': peak_amp,
            'found': True,
        }

    return peaks


def check_scalp_eeg(eeg_raw, abr_channels):
    """Check noise level (std dev) for all scalp EEG channels.

    Bandpass filters to 1-100 Hz before computing std dev, so that
    thresholds are comparable across different sampling rates.
    Excludes ABR channels and non-EEG channels (e.g., Audio).

    Args:
        eeg_raw: MNE Raw object with all channels loaded.
        abr_channels: list of ABR channel names to exclude.

    Returns:
        channel_stats: list of dicts with 'name', 'std_uv', 'status'
    """
    all_channels = eeg_raw.ch_names
    exclude = set(abr_channels) | {'Audio'}
    scalp_channels = [ch for ch in all_channels if ch not in exclude]

    if not scalp_channels:
        return []

    # Copy and pick only scalp channels for filtering
    raw_scalp = eeg_raw.copy().pick_channels(scalp_channels)
    # Bandpass filter 1-100 Hz for standard EEG range
    raw_scalp.filter(l_freq=1, h_freq=100, verbose='warning')

    channel_stats = []
    for ch in scalp_channels:
        data = raw_scalp.get_data(picks=ch)[0]  # shape (n_samples,)
        std_uv = np.std(data) * 1e6  # Convert V to uV

        # Flag channels (thresholds for 1-100 Hz filtered EEG)
        if std_uv < 0.5:
            status = "FLAT"
        elif std_uv > 75:
            status = "NOISY"
        else:
            status = "OK"

        channel_stats.append({
            'name': ch,
            'std_uv': std_uv,
            'status': status,
        })

    return channel_stats


def generate_report(eeg_file, n_epochs, eeg_fs, snr_db, sigma_sn, sigma_n, peaks, channel_stats=None):
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
    if snr_db is not None and not np.isnan(snr_db):
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

    if snr_db is None or np.isnan(snr_db):
        quality = "POOR - No detectable ABR signal"
    elif snr_db < 0:
        quality = "POOR - SNR below 0 dB"
    elif snr_db < 3:
        quality = "MARGINAL - SNR between 0-3 dB"
    elif snr_db < 6:
        quality = "ACCEPTABLE - SNR between 3-6 dB"
    else:
        quality = "GOOD - SNR above 6 dB"

    lines.append(f"  {quality}")

    wave_v = peaks.get('V', {})
    if wave_v.get('found', False):
        lines.append(f"  Wave V detected at {wave_v['latency_ms']:.2f} ms")
    else:
        lines.append("  WARNING: Wave V not clearly detected")

    # Scalp EEG channel QC section
    if channel_stats:
        lines.append("")
        lines.append("-" * 60)
        lines.append("  Scalp EEG Channel QC")
        lines.append("-" * 60)

        n_ok = sum(1 for ch in channel_stats if ch['status'] == 'OK')
        n_flat = sum(1 for ch in channel_stats if ch['status'] == 'FLAT')
        n_noisy = sum(1 for ch in channel_stats if ch['status'] == 'NOISY')
        n_total = len(channel_stats)

        lines.append(f"  Total channels:  {n_total}")
        lines.append(f"  OK:              {n_ok}")
        lines.append(f"  Flat:            {n_flat}")
        lines.append(f"  Noisy:           {n_noisy}")
        lines.append("")

        # List all channels with their RMS and status
        lines.append(f"  {'Channel':<12s}  {'Std Dev (uV)':>10s}  {'Status':<6s}")
        lines.append(f"  {'-'*12}  {'-'*10}  {'-'*6}")
        for ch in channel_stats:
            flag = ""
            if ch['status'] != 'OK':
                flag = f"  <-- {ch['status']}"
            lines.append(f"  {ch['name']:<12s}  {ch['std_uv']:>10.2f}  {ch['status']:<6s}{flag}")

    lines.append("=" * 60)

    return "\n".join(lines)


def plot_abr(abr_response, lags, peaks, snr_db, eeg_file, output_png, channel_stats=None):
    """Plot ABR waveform and scalp EEG channel RMS."""
    n_panels = 2 if channel_stats else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    # --- Top panel: ABR waveform [-10, 15] ms ---
    ax = axes[0]
    plot_mask = (lags >= -10) & (lags <= 15)
    ax.plot(lags[plot_mask], abr_response[plot_mask], 'k-', linewidth=1.2)

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

    # --- Bottom panel: Scalp EEG channel RMS ---
    if channel_stats:
        ax2 = axes[1]
        ch_names = [ch['name'] for ch in channel_stats]
        rms_values = [ch['std_uv'] for ch in channel_stats]
        statuses = [ch['status'] for ch in channel_stats]

        bar_colors = []
        for s in statuses:
            if s == 'FLAT':
                bar_colors.append('royalblue')
            elif s == 'NOISY':
                bar_colors.append('tomato')
            else:
                bar_colors.append('forestgreen')

        y_pos = np.arange(len(ch_names))
        ax2.barh(y_pos, rms_values, color=bar_colors, edgecolor='gray', linewidth=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(ch_names, fontsize=8)
        ax2.set_xlabel('Std Dev (uV)', fontsize=12)
        ax2.set_title('Scalp EEG Channel Std Dev', fontsize=14)
        ax2.invert_yaxis()

        # Threshold lines
        ax2.axvline(x=0.5, color='royalblue', linestyle='--', alpha=0.7, label='Flat threshold (0.5 uV)')
        ax2.axvline(x=75, color='tomato', linestyle='--', alpha=0.7, label='Noisy threshold (75 uV)')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


# %% Main Pipeline

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze Click ABR Quality')
    parser.add_argument('eeg_file', type=str, help='Path to the EEG .vhdr file') # change this to PID for our naming convention in later version
    parser.add_argument('--stim_path', type=str, default=None,
                        help='Path to the click stimuli directory (default: ../../stim_normalized/click relative to script)')

    args = parser.parse_args()

    eeg_vhdr = args.eeg_file

    # Resolve stim path
    if args.stim_path:
        stim_path = args.stim_path
    else:
        # Default to ../../stim_normalized/click relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        stim_path = os.path.join(script_dir, '../../stim_normalized/click')

    if not os.path.exists(eeg_vhdr):
        raise FileNotFoundError(f"EEG file not found: {eeg_vhdr}")

    if not os.path.exists(stim_path):
        print(f"Warning: Stimuli path not found: {stim_path}")
        # We might need stimuli to generate x_in. If not found, we can't proceed with cross-correlation exactly as designed
        # unless we generate synthetic clicks. But let's assume they exist.
        return

    print(f"Analyzing: {eeg_vhdr}")
    print(f"Stimuli path: {stim_path}")

    # %% Parameters
    # Analysis
    # is_click = True # if derive click ABR
    # is_ABR = True # if derive only ABR channels

    # Stim param
    stim_fs = 48000 # stimulus sampling frequency
    t_click = 60 # click trial length (approx, depends on file)
    click_rate = 40

    # EEG param
    # eeg_n_channel = 2 # total channel of ABR
    eeg_fs = 25000 # eeg sampling frequency (will check actual fs)
    eeg_f_hp = 1 # high pass cutoff

    # %% Load EEG data
    print("Loading EEG data...")
    # Suppress MNE info output to keep console clean
    eeg_raw = mne.io.read_raw_brainvision(eeg_vhdr, preload=True, verbose='warning')

    # Get actual sampling rate
    eeg_fs = eeg_raw.info['sfreq']
    print(f"EEG Sampling Rate: {eeg_fs} Hz")

    # Get events
    events, event_dict = mne.events_from_annotations(eeg_raw, verbose='warning')
    print(f"Found events: {event_dict}")

    # Check if we have the 'Stimulus/S  1' event (or similar) which usually corresponds to '1' in event_dict
    # The notebook used event_id=1. We need to find the key for 'Stimulus/S  1' or just '1'
    # MNE events_from_annotations maps descriptions to integers.
    # Usually 'Stimulus/S  1' -> 1 if it's the first one, but let's be robust.

    # In the notebook: Used Annotations descriptions: ['New Segment/', 'Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  4', 'Stimulus/S  8']
    # And events_from_annotations returns a dict.
    # We look for the event id corresponding to the click start.
    # Assuming 'Stimulus/S  1' is the start trigger for clicks.

    click_event_id = None
    for key, val in event_dict.items():
        if 'S  1' in key or 'S 1' in key:
            click_event_id = val
            break

    if click_event_id is None:
        # Fallback: try to find just '1' if the dict is simple
        if '1' in event_dict:
            click_event_id = event_dict['1']
        else:
            print("Warning: Could not find 'Stimulus/S 1' event. Using event ID 1 as default.")
            click_event_id = 1

    # Filter duplicate triggers: start_stimulus and stamp_triggers both send S1,
    # producing pairs ~0.1s apart. Keep the FIRST event in each pair (start_stimulus
    # marks actual audio onset).
    click_events = events[events[:, 2] == click_event_id]
    if len(click_events) > 1:
        min_gap_samples = int(1.0 * eeg_fs)  # 1 second minimum gap
        keep = [click_events[0]]
        for i in range(1, len(click_events)):
            if (click_events[i, 0] - keep[-1][0]) >= min_gap_samples:
                keep.append(click_events[i])
        click_events = np.array(keep)
        # Rebuild full events array with only deduplicated click events + other events
        other_events = events[events[:, 2] != click_event_id]
        events = np.vstack([other_events, click_events])
        events = events[events[:, 0].argsort()]  # re-sort by sample
    # The sound check also sends S1 triggers via start_stimulus().
    # The last 5 S1 triggers are always the click train trials.
    if len(click_events) > 5:
        print(f"Found {len(click_events)} S1 triggers after dedup. "
              f"Taking the last 5 as click train trials.")
        click_events = click_events[-5:]
        # Rebuild events array so mne.Epochs only sees these 5 click triggers
        other_events = events[events[:, 2] != click_event_id]
        events = np.vstack([other_events, click_events])
        events = events[events[:, 0].argsort()]
    print(f"Click triggers used: {len(click_events)}")

    # %% Scalp EEG QC (before picking ABR-only channels)
    print("Checking scalp EEG channels...")
    target_channels = ['Plus_R', 'Minus_R', 'Plus_L', 'Minus_L']
    channel_stats = check_scalp_eeg(eeg_raw, target_channels)
    n_flagged = sum(1 for ch in channel_stats if ch['status'] != 'OK')
    print(f"Scalp EEG: {len(channel_stats)} channels checked, {n_flagged} flagged.")

    # %% Pick ABR channels
    print("Picking ABR channels...")
    # Check if channels exist
    if not all(ch in eeg_raw.ch_names for ch in target_channels):
        print(f"Error: Missing required channels. Available: {eeg_raw.ch_names}")
        return

    eeg_raw.pick_channels(target_channels)

    # %% Get ABR channels with reference
    # Right ear: Plus_R - Minus_R
    # Left ear: Plus_L - Minus_L
    print("Re-referencing channels...")
    data_R = eeg_raw.get_data(picks='Plus_R') - eeg_raw.get_data(picks='Minus_R') # Right ear
    data_L = eeg_raw.get_data(picks='Plus_L') - eeg_raw.get_data(picks='Minus_L') # Left ear
    data = np.vstack((data_R, data_L)) # Combine channels
    data /= 100 # Scale data to microvolts (Notebook said /= 100, assuming gain factor?)

    # make info for RawArray
    info = mne.create_info(ch_names=["EP1","EP2"], sfreq=eeg_fs, ch_types='eeg')
    eeg_raw_ref = mne.io.RawArray(data, info, verbose='warning')

    # %% EEG Preprocessing
    print("Preprocessing...")
    # high pass filter
    eeg_raw_ref._data = butter_highpass_filter(eeg_raw_ref._data, eeg_f_hp, eeg_fs)

    # Notch filter
    notch_freq = np.arange(60, 540, 180)
    notch_width = 5
    for nf in notch_freq:
        bn, an = signal.iirnotch(nf / (eeg_fs / 2.), float(nf) / notch_width)
        eeg_raw_ref._data = signal.lfilter(bn, an, eeg_raw_ref._data)

    # %% Epoching
    print('Epoching EEG click data...')
    # t_click is 60s in notebook.
    # We need to ensure we don't go out of bounds if the recording is shorter.

    epochs_click = mne.Epochs(eeg_raw_ref, events, tmin=0,
                                tmax=(t_click - 1/stim_fs + 1),
                                event_id=click_event_id, baseline=None,
                                preload=True, proj=False, verbose='warning')

    if len(epochs_click) == 0:
        print("Error: No epochs found.")
        return

    epoch_click = epochs_click.get_data()
    n_epoch_click = len(epochs_click)
    print(f"Found {n_epoch_click} epochs.")

    # Keep only the last 5 epochs (click trains); earlier ones may be sound check
    if n_epoch_click > 5:
        print(f"Trimming to last 5 epochs (dropping first {n_epoch_click - 5} non-click epochs).")
        epoch_click = epoch_click[-5:]
        n_epoch_click = 5

    # %% Load click wave files and convert to pulse trains
    print("Loading click stimuli...")
    x_in = np.zeros((n_epoch_click, int(t_click * eeg_fs)), dtype=float)

    # Find click files
    click_files = sorted(glob.glob(os.path.join(stim_path, "click*.wav")))
    if len(click_files) < n_epoch_click:
        print(f"Warning: Found {len(click_files)} click files but have {n_epoch_click} epochs.")
        # We will loop over available files or repeat?
        # Notebook assumed click{0:03d}.wav matches epoch index.
        pass

    for ei in range(n_epoch_click):
        # Construct filename as in notebook
        fname = os.path.join(stim_path, 'click{0:03d}.wav'.format(ei))
        if not os.path.exists(fname):
            # Try to use the i-th file from glob if specific name fails
            if ei < len(click_files):
                fname = click_files[ei]
            else:
                print(f"Error: Could not find click file for epoch {ei}")
                continue

        fs_stim, stim_raw = wavfile.read(fname)
        # Convert to float and normalize to [-1, 1]
        if stim_raw.dtype == np.int16:
            stim = stim_raw.astype(float) / 32768.0
        elif stim_raw.dtype == np.int32:
            stim = stim_raw.astype(float) / 2147483648.0
        elif stim_raw.dtype == np.float32 or stim_raw.dtype == np.float64:
            stim = stim_raw.astype(float)
        else:
            stim = stim_raw.astype(float)

        # scipy.io.wavfile returns (n_samples,) for mono or (n_samples, n_channels) for stereo
        # Take first channel if stereo
        if stim.ndim > 1:
            stim = stim[:, 0]

        stim_abs = np.abs(stim)

        click_times = [(np.where(np.diff(s) > 0)[0] + 1) / float(fs_stim) for s in [stim_abs]]
        # Flatten list of lists
        click_times = click_times[0]

        click_inds = (click_times * eeg_fs).astype(int)

        # Ensure indices are within bounds
        click_inds = click_inds[click_inds < x_in.shape[1]]

        x_in[ei, click_inds] = 1 # generate click train as x_in

    # %% FFT of click trains (x_in) and EEG (x_out)
    print("Calculating Cross-Correlation...")
    len_eeg = int(eeg_fs * t_click)

    # Ensure dimensions match
    if epoch_click.shape[2] < len_eeg:
        print(f"Warning: Epoch length {epoch_click.shape[2]} is shorter than expected {len_eeg}. Truncating analysis window.")
        len_eeg = epoch_click.shape[2]
        x_in = x_in[:, :len_eeg]

    x_out = np.zeros((n_epoch_click, 2, len_eeg))
    for i in range(n_epoch_click):
        x_out_i = epoch_click[i, :, 0:len_eeg]
        # Resample if needed? Notebook did: mne.filter.resample(x_out_i, eeg_fs, eeg_fs)
        # which does nothing if up/down are same. But maybe it handles some edge cases?
        # We'll skip explicit resample if fs matches, or just copy.
        x_out[i, :, :] = x_out_i

    x_out = np.mean(x_out, axis=1) # average the two channels (EP1 and EP2)

    # %% Cross Correlation in frequency domain
    # Derive ABR
    t_start, t_stop = -200e-3, 200e-3 # ABR showed range

    # FFT
    x_in_fft = fft(x_in, axis=-1)
    x_out_fft = fft(x_out, axis=-1)

    # Cross Correlation
    cc = np.real(ifft(x_out_fft * np.conj(x_in_fft)))
    abr = np.mean(cc, axis=0) # average across trials
    abr /= (click_rate * t_click) # real unit value

    # %% Concatenate the derived response
    # Since the impulse response is circular, we can concatenate the last 200 ms to be as -200 ms before onset

    start_idx = int(t_start * eeg_fs) # This is negative
    stop_idx = int(t_stop * eeg_fs)

    # abr is length N. abr[start_idx:] takes the last part (negative indexing).
    abr_response = np.concatenate((abr[start_idx:], abr[0:stop_idx]))

    # generate time vector
    lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)

    # Ensure lengths match (sometimes np.arange can be off by 1 due to float precision)
    if len(lags) != len(abr_response):
        min_len = min(len(lags), len(abr_response))
        lags = lags[:min_len]
        abr_response = abr_response[:min_len]

    # %% QC Analysis and Reporting

    # Compute SNR
    snr_db, sigma_sn, sigma_n = compute_snr(abr_response, lags)

    # Detect peaks
    peaks = detect_abr_peaks(abr_response, lags)

    # Generate and print report
    report = generate_report(eeg_vhdr, n_epoch_click, eeg_fs,
                             snr_db, sigma_sn, sigma_n, peaks, channel_stats)
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
    plot_abr(abr_response, lags, peaks, snr_db, eeg_vhdr, output_png, channel_stats)
    print(f"Figure saved to: {output_png}")


if __name__ == "__main__":
    main()
