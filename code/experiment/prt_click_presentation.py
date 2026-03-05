#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRT Click Presentation & Sound Check Script

This script presents 5 click trains for:
1) Sound check (using a story segment)
2) EEG signal quality check
3) Baseline EEG collection

It is designed to be run BEFORE the main story experiment.

Usage:
    python prt_click_presentation.py

The script will interactively prompt for participant ID and session number.

@author: Tong
"""

# %% Set up sound device
import os
os.environ['SD_ENABLE_ASIO'] = '1'
import sounddevice as sd

# %% Import libraries
import numpy as np
from expyfun import ExperimentController
from expyfun.io import read_wav
import glob

# %% Parameters
FS = 48000  # Sample rate
N_CHANNELS = 2  # Stereo output
STIM_DB = 65  # Stimulus volume in dB

# %% Experiment instructions
INSTRUCTION_SOUND_CHECK = """
We are going to do a sound check\n 

You will hear a short part of a story\n

You should hear the story clearly and\n

the sound should be coming from both earphones.
"""

INSTRUCTION_CLICKS = """
Great! Now we are going to listen to some clicking sounds!
"""

INSTRUCTION_END = """
Great job!

Now we are ready to listen to some stories.
"""


def main():
    print("=" * 40)
    print("  PRT Click Presentation & Sound Check")
    print("=" * 40)

    pid = input("\nEnter participant ID (5 digits, e.g., 12544): ").strip()
    while not (pid.isdigit() and len(pid) == 5):
        pid = input("Invalid. Enter a 5-digit participant ID: ").strip()

    visit = input("Enter visit number (1 digit, e.g., 1): ").strip()
    while not (visit.isdigit() and len(visit) == 1):
        visit = input("Invalid. Enter a 1-digit visit number: ").strip()

    session = input("Enter session number (1 digit, e.g., 1): ").strip()
    while not (session.isdigit() and len(session) == 1):
        session = input("Invalid. Enter a 1-digit session number: ").strip()

    # %% Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    stim_path = os.path.join(project_root, 'stim_normalized')

    click_path = os.path.join(stim_path, "click")
    sound_check_file = os.path.join(
        stim_path, "12008_1_1_happy", "story", "12008_1_1_happy_studio.wav")

    # Validate paths
    click_files = sorted(glob.glob(os.path.join(click_path, "click*.wav")))
    if not click_files:
        print(f"Error: No click files found at {click_path}")
        return

    if not os.path.exists(sound_check_file):
        print(f"Error: Sound check file not found: {sound_check_file}")
        return

    print(f"Participant: {pid}, Visit: {visit}, Session: {session}")
    print(f"Click files: {len(click_files)} found")

    # %% Experiment setup
    ec_args = dict(
        exp_name='PRT_Clicks',
        participant=f"click_{pid}_{visit}_{session}",
        session=f"{visit}{session}",
        window_size=[2560, 1440],
        full_screen=True,
        n_channels=N_CHANNELS,
        version='dev',
        stim_fs=FS,
        stim_db=STIM_DB,
        force_quit=['end']
    )

    # %% Run experiment
    with ExperimentController(**ec_args) as ec:

        # --- Sound Check ---
        ec.screen_prompt(INSTRUCTION_SOUND_CHECK, live_keys=['space'])

        print(f"Loading sound check file: {sound_check_file}")
        story_audio, _ = read_wav(sound_check_file)

        ec.screen_text("Sound Check", pos=[0, 0], units='norm',
                        color='w', font_size=32)
        ec.flip()

        ec.load_buffer(story_audio)

        ec.identify_trial(ec_id="sound_check", ttl_id=[])

        # Show fixation cross for 1 second before sound check starts
        ec.screen_text("+", pos=(0.75, 0), units='norm',
                        color='white', font_size=64)
        ec.flip()
        ec.wait_secs(1.0)

        t_start = ec.start_stimulus()

        # Redraw cross after starting stimulus so it stays on screen
        ec.screen_text("+", pos=(0.75, 0), units='norm',
                        color='white', font_size=64)
        ec.flip()
        ec.wait_secs(0.1)

        while ec.current_time < t_start + 10.0:
            ec.check_force_quit()
            ec.wait_secs(0.1)

        ec.stop()
        ec.trial_ok()

        ec.screen_prompt(
            "Did you hear the story clearly?\n\nWas the sound playing from both earphones?",
            live_keys=['space'])

        # --- Click Trains ---
        ec.screen_prompt(INSTRUCTION_CLICKS, live_keys=['space'])

        loaded_clicks = []
        for fname in click_files:
            data, _ = read_wav(fname)
            loaded_clicks.append(data)

        print(f"Loaded {len(loaded_clicks)} click trains.")

        # Show fixation cross for the entire click session
        ec.screen_text("+", pos=(0.75, 0), units='norm',
                        color='white', font_size=64)
        ec.flip()
        ec.wait_secs(1.0)

        for i, click_data in enumerate(loaded_clicks):
            ec.load_buffer(click_data)

            ec.identify_trial(ec_id=f"click_{i}", ttl_id=[])

            t_start = ec.start_stimulus()

            # Redraw cross after starting stimulus so it stays on screen
            ec.screen_text("+", pos=(0.75, 0), units='norm',
                            color='white', font_size=64)
            ec.flip()
            ec.wait_secs(0.1)

            ec.stamp_triggers([1])

            duration = click_data.shape[1] / FS

            while ec.current_time < t_start + duration:
                ec.check_force_quit()
                ec.wait_secs(0.1)

            ec.stop()
            ec.trial_ok()

            # 1s pause between click trains (skip after last one)
            if i < len(loaded_clicks) - 1:
                ec.wait_secs(1.0)

        # --- End ---
        ec.screen_prompt(INSTRUCTION_END, live_keys=['space'])

    print("Click sounds completed!")


if __name__ == "__main__":
    main()
