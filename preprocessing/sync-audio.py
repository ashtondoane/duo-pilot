"""
Find the necessary shift and stretch factors to align the pulses in the video
with the pulses recorded by the MEG.
"""
from pathlib import Path

import mne
import numpy as np
import yaml

# config
config_file = Path(__file__).parent.parent / 'config.yml'

with open(config_file, 'r') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)

stim_ch = cfg['meg_ttl_channel']
raw_dir = Path(cfg['raw_folder'])
pulse_dir = Path(cfg['audio_folder']).parent / 'pulses'

mne.set_log_level('error')

for _dict in cfg['files']:
    # load pulses
    cam_file = Path(_dict['cam'])
    pulse_path = pulse_dir / f'{cam_file.stem}.npy'
    cam = np.load(pulse_path)
    cam_sfreq = np.load(pulse_dir / f'{cam_file.stem}_sfreq.npy')
    # load TTL from raw
    raw_path = raw_dir / _dict['raw']
    raw = mne.io.read_raw_fif(raw_path, preload=False, allow_maxshield=True)
    raw.pick(stim_ch).load_data()
    meg = raw.get_data().squeeze()
    meg_sfreq = raw.info['sfreq']
    # upsample
    # meg = mne.filter.resample(meg, up=cam_sfreq / meg_sfreq, npad='auto')
    # binarize
    meg = np.rint(meg / meg.max()).astype(int)
    # find the onsets
    meg_onset_times = (np.nonzero(np.diff(meg) == 1)[0] + 1) / meg_sfreq
    cam_onset_times = (np.nonzero(np.diff(cam) == 1)[0] + 1) / cam_sfreq
    # truncate
    last_idx = min(meg_onset_times.size, cam_onset_times.size) + 1
    meg_onset_times = meg_onset_times[:last_idx]
    cam_onset_times = cam_onset_times[:last_idx]
    # find regression line
    raise RuntimeError
    m, b = np.polyfit(x, y, 1)
    # TODO resume here ↑↑↑↑↑↑↑
