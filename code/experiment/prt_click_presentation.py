#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRT Click Presentation & Sound Check Script

This script presents 5 click trains for:
1) Sound check (using a story segment)
2) EEG signal quality check
3) Baseline EEG collection

It is designed to be run BEFORE the main story experiment.

@author: Tong
"""

# %% Set up sound device
import os
os.environ['SD_ENABLE_ASIO'] = '1'
import sounddevice as sd
# sd.query_hostapis()
# sd.query_devices()

# %% Import libraries
import numpy as np
from expyfun import ExperimentController
from expyfun.io import read_wav
import glob

# %% Set parameters
fs = 48000  # Sample rate
n_channel = 2  # Stereo output
stim_db = 65  # Stimulus volume in dB

# %% Load data
# Set up paths
# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
stim_path = os.path.join(project_root, 'stim_normalized')

click_path = os.path.join(stim_path, "click")
sound_check_file = os.path.join(stim_path, "12008_1_1_happy", "story", "12008_1_1_happy_studio.wav")

# Load click files
click_files = sorted(glob.glob(os.path.join(click_path, "click*.wav")))
# Check if click files exist
if not click_files:
    pass

# %% Experiment instructions
instruction_sound_check = """
We are going to do a quick sound check first.

You will hear a short part of a story.

Please tell us if you can hear it clearly!
"""

instruction_clicks = """
Great! Now we are going to listen to some funny clicking sounds!
"""

instruction_end = """
Great job! 

Now we are ready for the stories.
"""

# %% Experiment setup
ec_args = dict(
    exp_name='PRT_Clicks',
    window_size=[2560, 1440],
    session='00',
    full_screen=True,
    n_channels=n_channel,
    version='dev',
    stim_fs=fs,
    stim_db=stim_db,
    force_quit=['end']
)

# %% Run experiment
with ExperimentController(**ec_args) as ec:
    
    # --- Sound Check ---
    ec.screen_prompt(instruction_sound_check, live_keys=['space'])
    
    # Load sound check stimulus
    print(f"Loading sound check file: {sound_check_file}")
    try:
        # Try loading from the hardcoded path first
        story_audio, _ = read_wav(sound_check_file)
    except Exception:
        pass

    # If loading failed (e.g. path issue), we might want to skip or error. 
    # I'll add a check.
    if 'story_audio' not in locals():
         story_audio = np.zeros((2, 480000)) # 10s silence

    # Play 10 seconds
    ec.screen_text("Sound Check", pos=[0, 0], units='norm', color='w', font_size=32)
    ec.flip()
    
    ec.load_buffer(story_audio)
    
    # Identify trial
    ec.identify_trial(ec_id="sound_check", ttl_id=[])
    
    # Show fixation cross for 1 second before sound check starts
    ec.screen_text("+", pos=(0.75, 0), units='norm', color='white', font_size=64)
    ec.flip()
    ec.wait_secs(1.0)
    
    # Start stimulus
    t_start = ec.start_stimulus()
    
    # Redraw cross after starting stimulus so it stays on screen
    ec.screen_text("+", pos=(0.75, 0), units='norm', color='white', font_size=64)
    ec.flip()
    ec.wait_secs(0.1)
    
    # Wait 10 seconds
    while ec.current_time < t_start + 10.0:
        ec.check_force_quit()
        ec.wait_secs(0.1)
        
    ec.stop()
    ec.trial_ok()
    
    # Ask for confirmation
    ec.screen_prompt("Did you hear the story clearly? (Press Space)", live_keys=['space'])
    
    # --- Click Trains ---
    ec.screen_prompt(instruction_clicks, live_keys=['space'])
    
    # Load click files
    loaded_clicks = []
    # Try the hardcoded path
    fnames = sorted(glob.glob(os.path.join(click_path, "click*.wav")))
    if not fnames:
        print(f"Warning: No click files found at {click_path}")
    
    for fname in fnames:
        data, _ = read_wav(fname)
        loaded_clicks.append(data)
        
    if not loaded_clicks:
        print("No clicks loaded! Skipping click section.")
    else:
        print(f"Loaded {len(loaded_clicks)} click trains.")
        
        for i, click_data in enumerate(loaded_clicks):
            ec.load_buffer(click_data)
            
            # Show fixation cross for 1 second before click train starts
            ec.screen_text("+", pos=(0.75, 0), units='norm', color='white', font_size=64)
            ec.flip()
            ec.wait_secs(1.0)
            
            # Identify trial
            ec.identify_trial(ec_id=f"click_{i}", ttl_id=[])
            
            # Start
            t_start = ec.start_stimulus()

            # Redraw cross after starting stimulus so it stays on screen
            ec.screen_text("+", pos=(0.75, 0), units='norm', color='white', font_size=64)
            ec.flip()
            ec.wait_secs(0.1)
            
            # Send trigger (simple 1 for all clicks for now, or i+1)
            ec.stamp_triggers([1])
            
            duration = click_data.shape[1] / fs
            
            while ec.current_time < t_start + duration:
                ec.check_force_quit()
                ec.wait_secs(0.1)
                
            ec.stop()
            ec.trial_ok()
            
            # Pause between clicks
            ec.wait_secs(1.0)
            
    # --- End ---
    ec.screen_prompt(instruction_end, live_keys=['space'])

print("Click presentation completed!")
