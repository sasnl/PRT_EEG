#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Click ABR Quality Check Script

This script analyzes EEG data collected during the click presentation session
to derive the Auditory Brainstem Response (ABR). It is intended to be run
immediately after data collection to verify signal quality.

Usage:
    python check_click_quality.py <path_to_vhdr_file> [--stim_path <path_to_clicks>]

Example:
    python check_click_quality.py ../../data/subject01/click_data.vhdr
"""

import argparse
import os
import numpy as np
import scipy.signal as signal
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from expyfun.io import read_wav
import mne
import glob

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

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze Click ABR Quality')
    parser.add_argument('eeg_file', type=str, help='Path to the EEG .vhdr file') # change this to PID for our naming convention in later version
    parser.add_argument('--stim_path', type=str, default=None, 
                        help='Path to the click stimuli directory (default: ../../stimuli/click relative to script)')
    
    args = parser.parse_args()
    
    eeg_vhdr = args.eeg_file
    
    # Resolve stim path
    if args.stim_path:
        stim_path = args.stim_path
    else:
        # Default to ../../stimuli/click relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        stim_path = os.path.join(script_dir, '../../stimuli/click')
    
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

    # %% Pick ABR channels
    print("Picking ABR channels...")
    target_channels = ['Plus_R', 'Minus_R', 'Plus_L', 'Minus_L']
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
                
        stim, fs_stim = read_wav(fname)
        stim_abs = np.abs(stim)
        
        # Read click event
        # Note: stim_abs might be stereo (2, N) or mono (1, N) or (N,). 
        # read_wav returns (data, fs). data is (n_channels, n_samples).
        if stim_abs.ndim > 1:
            stim_abs = stim_abs[0] # Take first channel if stereo
            
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
    t_start, t_stop = -200e-3, 600e-3 # ABR showed range
    
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

    # %% Plot the response
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    plt.plot(lags, abr_response)
    plt.xlim([-20, 60])
    plt.xlabel('Lag (ms)')
    plt.ylabel('Amplitude (uV)')
    plt.title(f'Click ABR Response\n{os.path.basename(eeg_vhdr)}')
    plt.grid(True)
    
    # Save plot
    output_dir = os.path.dirname(eeg_vhdr)
    output_base = os.path.splitext(os.path.basename(eeg_vhdr))[0]
    output_png = os.path.join(output_dir, f"{output_base}_abr_check.png")
    
    plt.savefig(output_png)
    print(f"Plot saved to: {output_png}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
