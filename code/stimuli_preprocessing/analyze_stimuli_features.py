#!/usr/bin/env python3
"""
Stimuli Feature Analysis Script
Analyzes WAV files and extracts acoustic features: pitch range, mean pitch,
pitch variability, and speech rate using Parselmouth.
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import parselmouth
from parselmouth.praat import call


def extract_emotion_from_filename(filename):
    """
    Extract and normalize emotion label from filename.
    Handles variations like 'happy', 'Happy1', 'sad', 'Sad2', 'spontaneous'.
    Returns normalized lowercase emotion: 'happy', 'sad', or 'spontaneous'.
    """
    filename_lower = filename.lower()

    # Check for emotion keywords (with optional numbers)
    if 'happy' in filename_lower:
        return 'happy'
    elif 'sad' in filename_lower:
        return 'sad'
    elif 'spontaneous' in filename_lower:
        return 'spontaneous'
    else:
        return 'unknown'


def analyze_stimuli_file(file_path):
    """
    Analyze a single WAV file and extract acoustic features.

    Features extracted:
    - Pitch: min, max, range, mean, variability (std dev)
    - Speech rate: phonation time ratio, syllable rate
    - Emotion: from filename
    """
    try:
        filename = os.path.basename(file_path)

        # Extract emotion from filename
        emotion = extract_emotion_from_filename(filename)

        # Load sound using parselmouth
        sound = parselmouth.Sound(file_path)

        # Extract pitch (F0) - optimized for children's speech
        pitch = call(sound, "To Pitch", 0.0, 75, 400)  # time_step=0 (auto), fmin=75, fmax=400

        # Get pitch values
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]  # Filter out unvoiced frames (0 Hz)

        # Calculate pitch statistics
        if len(pitch_values) > 0:
            f0_min = np.min(pitch_values)
            f0_max = np.max(pitch_values)
            pitch_range = f0_max - f0_min
            mean_pitch = np.mean(pitch_values)
            pitch_variability = np.std(pitch_values)
        else:
            f0_min = f0_max = pitch_range = mean_pitch = pitch_variability = 0

        # Calculate phonation time ratio (voiced frames / total frames)
        total_frames = len(pitch.selected_array['frequency'])
        voiced_frames = np.sum(pitch.selected_array['frequency'] > 0)
        phonation_time_ratio = voiced_frames / total_frames if total_frames > 0 else 0

        # Calculate syllable rate using intensity peaks as proxy for syllables
        # This is a common method: detect peaks in intensity contour
        intensity = call(sound, "To Intensity", 75, 0.0, "yes")  # min_pitch=75, time_step=0 (auto)

        # Get intensity values as array
        intensity_array = intensity.values[0]  # Get the intensity values

        # Find peaks in intensity (syllable nuclei)
        # Use a simple peak detection: local maxima above mean intensity
        if len(intensity_array) > 0:
            mean_intensity = np.mean(intensity_array)

            # Find local maxima
            peaks = []
            for i in range(1, len(intensity_array) - 1):
                if (intensity_array[i] > intensity_array[i-1] and
                    intensity_array[i] > intensity_array[i+1] and
                    intensity_array[i] > mean_intensity):
                    peaks.append(i)

            # Calculate syllable rate
            duration = sound.get_total_duration()
            num_syllables = len(peaks)
            syllable_rate = num_syllables / duration if duration > 0 else 0
        else:
            syllable_rate = 0

        return {
            'filename': filename,
            'emotion': emotion,
            'duration_seconds': round(sound.get_total_duration(), 3),
            'f0_min_hz': round(f0_min, 2),
            'f0_max_hz': round(f0_max, 2),
            'pitch_range_hz': round(pitch_range, 2),
            'mean_pitch_hz': round(mean_pitch, 2),
            'pitch_variability_hz': round(pitch_variability, 2),
            'phonation_time_ratio': round(phonation_time_ratio, 4),
            'syllable_rate': round(syllable_rate, 2)
        }

    except Exception as e:
        return {
            'filename': os.path.basename(file_path),
            'emotion': extract_emotion_from_filename(os.path.basename(file_path)),
            'error': str(e)
        }


def main():
    """Main function to analyze all WAV files in the stimuli folder."""
    stimuli_folder = Path("/Users/tongshan/Documents/PRT_EEG/stimuli")

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
    for i, wav_file in enumerate(wav_files, 1):
        print(f"[{i}/{len(wav_files)}] Analyzing: {wav_file.name}")
        result = analyze_stimuli_file(str(wav_file))
        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Display results
    print("\n" + "="*80)
    print("STIMULI ACOUSTIC FEATURES ANALYSIS")
    print("="*80)

    # Check for errors
    if 'error' in df.columns:
        error_files = df[df['error'].notna()]
        if not error_files.empty:
            print("\nERRORS:")
            for _, row in error_files.iterrows():
                print(f"  {row['filename']}: {row['error']}")
            print()

    # Remove error rows for analysis
    df_clean = df[~df['error'].notna()] if 'error' in df.columns else df

    if not df_clean.empty:
        # Overall summary statistics
        print("\nOVERALL SUMMARY:")
        print(f"Total files analyzed: {len(df_clean)}")
        print(f"Total duration: {df_clean['duration_seconds'].sum():.2f} seconds")
        print(f"Average duration: {df_clean['duration_seconds'].mean():.2f} seconds")

        # Emotion distribution
        emotion_counts = df_clean['emotion'].value_counts()
        print(f"\nEmotion distribution:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} files")

        # Summary by emotion
        print("\n" + "="*80)
        print("SUMMARY BY EMOTION")
        print("="*80)

        for emotion in df_clean['emotion'].unique():
            emotion_data = df_clean[df_clean['emotion'] == emotion]
            print(f"\n{emotion.upper()} (n={len(emotion_data)}):")
            print(f"  Mean pitch: {emotion_data['mean_pitch_hz'].mean():.2f} Hz (SD: {emotion_data['mean_pitch_hz'].std():.2f})")
            print(f"  Pitch range: {emotion_data['pitch_range_hz'].mean():.2f} Hz (SD: {emotion_data['pitch_range_hz'].std():.2f})")
            print(f"  Pitch variability: {emotion_data['pitch_variability_hz'].mean():.2f} Hz (SD: {emotion_data['pitch_variability_hz'].std():.2f})")
            print(f"  F0 min: {emotion_data['f0_min_hz'].mean():.2f} Hz (SD: {emotion_data['f0_min_hz'].std():.2f})")
            print(f"  F0 max: {emotion_data['f0_max_hz'].mean():.2f} Hz (SD: {emotion_data['f0_max_hz'].std():.2f})")
            print(f"  Phonation time ratio: {emotion_data['phonation_time_ratio'].mean():.4f} (SD: {emotion_data['phonation_time_ratio'].std():.4f})")
            print(f"  Syllable rate: {emotion_data['syllable_rate'].mean():.2f} syllables/sec (SD: {emotion_data['syllable_rate'].std():.2f})")

        # Detailed file information
        print("\n" + "="*80)
        print("DETAILED FILE INFORMATION")
        print("="*80)

        # Sort by emotion then filename
        df_sorted = df_clean.sort_values(['emotion', 'filename'])

        for _, row in df_sorted.iterrows():
            print(f"\nFile: {row['filename']}")
            print(f"  Emotion: {row['emotion']}")
            print(f"  Duration: {row['duration_seconds']}s")
            print(f"  F0 Min: {row['f0_min_hz']} Hz")
            print(f"  F0 Max: {row['f0_max_hz']} Hz")
            print(f"  Pitch Range: {row['pitch_range_hz']} Hz")
            print(f"  Mean Pitch: {row['mean_pitch_hz']} Hz")
            print(f"  Pitch Variability: {row['pitch_variability_hz']} Hz")
            print(f"  Phonation Time Ratio: {row['phonation_time_ratio']}")
            print(f"  Syllable Rate: {row['syllable_rate']} syllables/sec")

    # Save to CSV
    output_file = "stimuli_features_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
