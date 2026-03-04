#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sound Calibration Script

Plays a speech recording on loop through expyfun at stim_db=65
for volume calibration. Press Space to stop.

Usage:
    python sound_calibration.py

@author: Tong
"""

import os
os.environ['SD_ENABLE_ASIO'] = '1'
import sounddevice as sd

import numpy as np
from expyfun import ExperimentController
from expyfun.io import read_wav

FS = 48000
N_CHANNELS = 2
STIM_DB = 65


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    wav_path = os.path.join(
        project_root, 'stim_normalized',
        '12008_1_1_happy', 'story', '12008_1_1_happy_studio.wav')

    if not os.path.exists(wav_path):
        print(f"ERROR: File not found: {wav_path}")
        return

    print("=" * 40)
    print("  Sound Calibration")
    print("=" * 40)
    print(f"\nFile: {os.path.basename(wav_path)}")
    print(f"stim_db: {STIM_DB}")

    audio, _ = read_wav(wav_path)
    duration = audio.shape[1] / FS
    print(f"Duration: {duration:.1f}s (will loop)")

    ec_args = dict(
        exp_name='SoundCalibration',
        participant='calibration',
        session='0',
        window_size=[800, 600],
        full_screen=False,
        n_channels=N_CHANNELS,
        version='dev',
        stim_fs=FS,
        stim_db=STIM_DB,
        force_quit=['end'],
    )

    with ExperimentController(**ec_args) as ec:
        ec.screen_text(
            "Sound Calibration\n\n"
            "Speech is playing on loop.\n\n"
            "Adjust volume to desired level.\n\n"
            "Press Space to stop.",
            pos=[0, 0], units='norm', color='w', font_size=24, wrap=True)
        ec.flip()

        playing = True
        while playing:
            ec.load_buffer(audio)
            ec.start_stimulus()

            t0 = ec.current_time
            while ec.current_time < t0 + duration:
                ec.check_force_quit()
                pressed = ec.get_presses(live_keys=['space'],
                                         timestamp=False)
                if pressed:
                    playing = False
                    break
                ec.wait_secs(0.05)

            ec.stop()

        ec.screen_text(
            "Calibration complete.",
            pos=[0, 0], units='norm', color='w', font_size=24)
        ec.flip()
        ec.wait_secs(1.0)

    print("\nCalibration done.")


if __name__ == "__main__":
    main()
