#!/usr/bin/env python3
"""
Audio Analysis Script for WAV Files
Analyzes all .wav files in the stimuli folder and extracts format, duration, channels, RMS, and other properties.
"""

import os
import wave
import numpy as np
import librosa
from scipy.io import wavfile
import pandas as pd
from pathlib import Path

def analyze_wav_file(file_path):
    """Analyze a single WAV file and return its properties."""
    try:
        # Basic file info
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        # Using wave module for basic properties
        with wave.open(file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            duration = n_frames / sample_rate
            
        # Using librosa for advanced analysis
        audio_data, sr = librosa.load(file_path, sr=None, mono=False)
        
        # Ensure audio_data is 2D for consistent processing
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)
            
        # Calculate RMS for each channel
        rms_values = []
        for i in range(audio_data.shape[0]):
            rms = np.sqrt(np.mean(audio_data[i] ** 2))
            rms_values.append(rms)
        
        # Calculate overall RMS
        overall_rms = np.sqrt(np.mean(audio_data ** 2))
        
        # Peak amplitude
        peak_amplitude = np.max(np.abs(audio_data))
        
        # Dynamic range
        dynamic_range = 20 * np.log10(peak_amplitude / overall_rms) if overall_rms > 0 else 0
        
        # Zero crossing rate (for first channel)
        zcr = librosa.feature.zero_crossing_rate(audio_data[0] if audio_data.shape[0] > 0 else audio_data)
        avg_zcr = np.mean(zcr)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data[0] if audio_data.shape[0] > 0 else audio_data, sr=sr)
        avg_spectral_centroid = np.mean(spectral_centroid)
        
        # Pitch (F0) analysis using librosa's piptrack
        # Use the first channel for mono or the first channel of stereo
        audio_mono = audio_data[0] if audio_data.shape[0] > 0 else audio_data
        
        # Extract fundamental frequency using piptrack
        pitches, magnitudes = librosa.piptrack(y=audio_mono, sr=sr, threshold=0.1, fmin=50, fmax=500)
        
        # Extract the most prominent pitch at each time frame
        f0_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Only consider valid pitch values
                f0_values.append(pitch)
        
        # Calculate pitch statistics
        if f0_values:
            f0_values = np.array(f0_values)
            mean_f0 = np.mean(f0_values)
            f0_std = np.std(f0_values)
            f0_min = np.min(f0_values)
            f0_max = np.max(f0_values)
            f0_range = f0_max - f0_min
            # Robust range (10th to 90th percentile)
            f0_robust_range = np.percentile(f0_values, 90) - np.percentile(f0_values, 10)
        else:
            mean_f0 = f0_std = f0_min = f0_max = f0_range = f0_robust_range = 0
        
        return {
            'filename': os.path.basename(file_path),
            'file_size_mb': round(file_size, 2),
            'duration_seconds': round(duration, 3),
            'duration_minutes': round(duration / 60, 2),
            'sample_rate': sample_rate,
            'channels': n_channels,
            'channel_type': 'mono' if n_channels == 1 else 'stereo' if n_channels == 2 else f'{n_channels}-channel',
            'bit_depth': sample_width * 8,
            'total_samples': n_frames,
            'overall_rms': round(overall_rms, 6),
            'overall_rms_db': round(20 * np.log10(overall_rms), 2) if overall_rms > 0 else -np.inf,
            'peak_amplitude': round(peak_amplitude, 6),
            'dynamic_range_db': round(dynamic_range, 2),
            'avg_zero_crossing_rate': round(avg_zcr, 6),
            'avg_spectral_centroid_hz': round(avg_spectral_centroid, 2),
            'mean_f0_hz': round(mean_f0, 2),
            'f0_std_hz': round(f0_std, 2),
            'f0_min_hz': round(f0_min, 2),
            'f0_max_hz': round(f0_max, 2),
            'f0_range_hz': round(f0_range, 2),
            'f0_robust_range_hz': round(f0_robust_range, 2),
            'pitch_variability': round(f0_std, 2),
            'rms_per_channel': [round(rms, 6) for rms in rms_values]
        }
        
    except Exception as e:
        return {
            'filename': os.path.basename(file_path),
            'error': str(e)
        }

def main():
    """Main function to analyze all WAV files in the stimuli folder."""
    stimuli_folder = Path("stimuli")
    
    if not stimuli_folder.exists():
        print(f"Error: {stimuli_folder} folder not found!")
        return
    
    # Find all WAV files
    wav_files = list(stimuli_folder.glob("*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in {stimuli_folder}")
        return
    
    print(f"Found {len(wav_files)} WAV files to analyze...\n")
    
    # Analyze each file
    results = []
    for wav_file in wav_files:
        print(f"Analyzing: {wav_file.name}")
        result = analyze_wav_file(str(wav_file))
        results.append(result)
    
    # Create DataFrame for better formatting
    df = pd.DataFrame(results)
    
    # Display results
    print("\n" + "="*80)
    print("AUDIO ANALYSIS RESULTS")
    print("="*80)
    
    if 'error' in df.columns:
        error_files = df[df['error'].notna()]
        if not error_files.empty:
            print("\nERRORS:")
            for _, row in error_files.iterrows():
                print(f"  {row['filename']}: {row['error']}")
            print()
    
    # Remove error rows for summary statistics
    df_clean = df[~df['error'].notna()] if 'error' in df.columns else df
    
    if not df_clean.empty:
        # Summary statistics
        print("\nSUMMARY STATISTICS:")
        print(f"Total files analyzed: {len(df_clean)}")
        print(f"Total duration: {df_clean['duration_minutes'].sum():.2f} minutes")
        print(f"Average duration: {df_clean['duration_seconds'].mean():.2f} seconds")
        print(f"Total file size: {df_clean['file_size_mb'].sum():.2f} MB")
        
        # Channel distribution
        channel_counts = df_clean['channel_type'].value_counts()
        print(f"\nChannel distribution:")
        for channel_type, count in channel_counts.items():
            print(f"  {channel_type}: {count} files")
        
        # Sample rate distribution
        sr_counts = df_clean['sample_rate'].value_counts()
        print(f"\nSample rate distribution:")
        for sr, count in sr_counts.items():
            print(f"  {sr} Hz: {count} files")
        
        # Bit depth distribution
        bit_depth_counts = df_clean['bit_depth'].value_counts()
        print(f"\nBit depth distribution:")
        for depth, count in bit_depth_counts.items():
            print(f"  {depth}-bit: {count} files")
        
        print(f"\nRMS range: {df_clean['overall_rms'].min():.6f} to {df_clean['overall_rms'].max():.6f}")
        print(f"RMS (dB) range: {df_clean['overall_rms_db'].min():.2f} to {df_clean['overall_rms_db'].max():.2f} dB")
        
        # Pitch analysis summary
        print(f"\nPITCH ANALYSIS SUMMARY:")
        print(f"Mean F0 range: {df_clean['mean_f0_hz'].min():.2f} to {df_clean['mean_f0_hz'].max():.2f} Hz")
        print(f"F0 variability range: {df_clean['pitch_variability'].min():.2f} to {df_clean['pitch_variability'].max():.2f} Hz")
        print(f"Average F0 across all files: {df_clean['mean_f0_hz'].mean():.2f} Hz")
        print(f"Average pitch variability: {df_clean['pitch_variability'].mean():.2f} Hz")
        
        # Detailed file information
        print("\n" + "="*80)
        print("DETAILED FILE INFORMATION")
        print("="*80)
        
        for _, row in df_clean.iterrows():
            print(f"\nFile: {row['filename']}")
            print(f"  Duration: {row['duration_seconds']}s ({row['duration_minutes']} min)")
            print(f"  Format: {row['sample_rate']} Hz, {row['bit_depth']}-bit, {row['channel_type']}")
            print(f"  Size: {row['file_size_mb']} MB ({row['total_samples']:,} samples)")
            print(f"  RMS: {row['overall_rms']:.6f} ({row['overall_rms_db']:.2f} dB)")
            print(f"  Peak: {row['peak_amplitude']:.6f}")
            print(f"  Dynamic Range: {row['dynamic_range_db']:.2f} dB")
            print(f"  Zero Crossing Rate: {row['avg_zero_crossing_rate']:.6f}")
            print(f"  Spectral Centroid: {row['avg_spectral_centroid_hz']:.2f} Hz")
            print(f"  Mean F0 (Pitch): {row['mean_f0_hz']:.2f} Hz")
            print(f"  F0 Range: {row['f0_range_hz']:.2f} Hz ({row['f0_min_hz']:.2f} - {row['f0_max_hz']:.2f} Hz)")
            print(f"  F0 Robust Range: {row['f0_robust_range_hz']:.2f} Hz")
            print(f"  Pitch Variability (F0 Std): {row['pitch_variability']:.2f} Hz")
            if row['channels'] > 1:
                print(f"  RMS per channel: {row['rms_per_channel']}")
    
    # Save to CSV
    output_file = "wav_analysis_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()