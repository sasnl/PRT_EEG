#!/usr/bin/env python3
"""
Convert .aifc files to .wav format with specified sampling frequency.
Supports both single file and batch processing of folders.
"""

import os
import argparse
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np


def convert_aifc_to_wav(input_path, output_path=None, target_sr=44100, verbose=True):
    """
    Convert a single .aifc file to .wav format.

    Parameters:
    -----------
    input_path : str or Path
        Path to input .aifc file
    output_path : str or Path, optional
        Path to output .wav file. If None, replaces .aifc extension with .wav
    target_sr : int, default=44100
        Target sampling rate in Hz
    verbose : bool, default=True
        Print conversion details

    Returns:
    --------
    str : Path to the output .wav file
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.with_suffix('.wav')
    else:
        output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load audio file (librosa handles .aifc format)
        audio, original_sr = librosa.load(str(input_path), sr=None, mono=False)

        if verbose:
            print(f"Processing: {input_path.name}")
            print(f"  Original SR: {original_sr} Hz")
            print(f"  Target SR: {target_sr} Hz")

        # Resample if necessary
        if original_sr != target_sr:
            if audio.ndim == 1:
                # Mono audio
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
            else:
                # Stereo/multi-channel audio - resample each channel
                resampled_channels = []
                for channel in audio:
                    resampled = librosa.resample(channel, orig_sr=original_sr, target_sr=target_sr)
                    resampled_channels.append(resampled)
                audio = np.array(resampled_channels)

            if verbose:
                print(f"  Resampled to {target_sr} Hz")

        # Transpose if multi-channel (soundfile expects channels as last dimension)
        if audio.ndim > 1:
            audio = audio.T

        # Save as .wav file
        sf.write(str(output_path), audio, target_sr, subtype='PCM_16')

        if verbose:
            print(f"  Saved to: {output_path}")
            print()

        return str(output_path)

    except Exception as e:
        print(f"Error converting {input_path.name}: {str(e)}")
        raise


def batch_convert_aifc_to_wav(input_folder, output_folder=None, target_sr=44100, recursive=False):
    """
    Convert all .aifc files in a folder to .wav format.

    Parameters:
    -----------
    input_folder : str or Path
        Path to folder containing .aifc files
    output_folder : str or Path, optional
        Path to output folder. If None, files are saved in the same location as input
    target_sr : int, default=44100
        Target sampling rate in Hz
    recursive : bool, default=False
        If True, search for .aifc files recursively in subfolders

    Returns:
    --------
    list : Paths to converted .wav files
    """
    input_folder = Path(input_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # Find all .aifc files
    if recursive:
        aifc_files = list(input_folder.rglob('*.aifc')) + list(input_folder.rglob('*.AIFC'))
    else:
        aifc_files = list(input_folder.glob('*.aifc')) + list(input_folder.glob('*.AIFC'))

    if not aifc_files:
        print(f"No .aifc files found in {input_folder}")
        return []

    print(f"Found {len(aifc_files)} .aifc file(s)")
    print(f"Target sampling rate: {target_sr} Hz")
    print("-" * 60)

    converted_files = []
    successful = 0
    failed = 0

    for aifc_file in aifc_files:
        try:
            # Determine output path
            if output_folder is None:
                output_path = aifc_file.with_suffix('.wav')
            else:
                output_folder = Path(output_folder)
                # Preserve relative directory structure if recursive
                if recursive:
                    relative_path = aifc_file.relative_to(input_folder)
                    output_path = output_folder / relative_path.with_suffix('.wav')
                else:
                    output_path = output_folder / aifc_file.with_suffix('.wav').name

            # Convert file
            result = convert_aifc_to_wav(aifc_file, output_path, target_sr)
            converted_files.append(result)
            successful += 1

        except Exception as e:
            failed += 1
            print(f"Failed to convert {aifc_file.name}: {str(e)}\n")

    print("-" * 60)
    print(f"Conversion complete: {successful} successful, {failed} failed")

    return converted_files


def main():
    parser = argparse.ArgumentParser(
        description='Convert .aifc files to .wav format with specified sampling frequency'
    )

    parser.add_argument(
        'input',
        help='Input .aifc file or folder containing .aifc files'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output .wav file or folder (optional, defaults to same location as input)'
    )

    parser.add_argument(
        '-sr', '--sample-rate',
        type=int,
        default=44100,
        help='Target sampling rate in Hz (default: 44100)'
    )

    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Process subfolders recursively (only for folder input)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    # Check if input is a file or folder
    if input_path.is_file():
        # Single file conversion
        convert_aifc_to_wav(input_path, args.output, args.sample_rate)

    elif input_path.is_dir():
        # Batch conversion
        batch_convert_aifc_to_wav(
            input_path,
            args.output,
            args.sample_rate,
            args.recursive
        )

    else:
        print(f"Error: Input path not found: {input_path}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
