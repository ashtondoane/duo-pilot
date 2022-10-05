"""
Converts an audio file to a binary NumPy array, where 1 values indicate time
samples that were suprathreshold. The threshold is computed automatically from
the audio file by comparing the first second of the file (presumed free of
pulses) to the max absolute value. Results are saved as .npy files.
"""
import os
from pathlib import Path

import numpy as np
import yaml
from scipy.io import wavfile

# config
config_file = Path(__file__).parent.parent / 'config.yml'

with open(config_file, 'r') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)

audio_dir = Path(cfg['audio_folder'])
pulse_dir = audio_dir.parent / 'pulses'
os.makedirs(pulse_dir, exist_ok=True)

for audio_file in audio_dir.glob('*.wav'):
    # use the first 1 second to guess at a good pulse detection threshold
    sfreq, data = wavfile.read(audio_file)
    first_sec = data[:sfreq]
    noise_max = np.abs(first_sec).max()
    # this is super coarse / not "real" SNR, but should be good enough
    snr = np.abs(data).max() / noise_max
    thresh = noise_max * int(snr // 2)
    # now find the pulses
    binarized = (data > thresh).astype(np.uint8)
    outfile = pulse_dir / f'{audio_file.stem}.npy'
    np.save(outfile, binarized)
