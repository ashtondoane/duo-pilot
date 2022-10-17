"""
Find the necessary shift and stretch factors to align the pulses in the video
with the pulses recorded by the MEG.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import yaml
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import correlate

# config
config_file = Path(__file__).resolve().parent.parent / 'config.yml'
threshold = 0.1  # pulse detection in cam audio channel (global wav max = 1)
show_plots = False

with open(config_file, 'r') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)

stim_ch = cfg['meg_ttl_channel']
raw_dir = Path(cfg['raw_folder'])
audio_dir = Path(cfg['audio_folder'])
sync_dir = audio_dir.parent / 'sync'
os.makedirs(sync_dir, exist_ok=True)

mne.set_log_level('error')

if show_plots:
    plt.ion()

for _dict in cfg['files']:
    if _dict['cam'] is None:
        continue
    print(f'processing {Path(_dict["raw"]).parts[0]}')

    # load TTL from raw & binarize
    raw_path = raw_dir / _dict['raw']
    raw = mne.io.read_raw_fif(raw_path, preload=False, allow_maxshield=True)
    raw.pick(stim_ch).load_data()
    meg, meg_times = raw.get_data(return_times=True)
    meg = np.rint(meg / meg.max()).astype(int).squeeze()
    meg_sfreq = raw.info['sfreq']

    # load original wav
    fname_stem = Path(_dict['cam']).stem
    wav_path = audio_dir / f'{fname_stem}.wav'
    wav_sfreq, wav = wavfile.read(wav_path)
    wav = wav / np.abs(wav).max()  # convert to float
    wav_times = np.arange(wav.size) / wav_sfreq

    if show_plots:
        fig, axs = plt.subplots(5, 1, sharex=True, sharey=True,
                                figsize=(20, 10))
        axs[0].plot(wav_times, wav, label='cam')
        axs[0].plot(meg_times, meg + 1.1, label='meg')
        axs[0].set(title='raw signals')

    # compute WAV pulse onset times
    wav_pulses = (wav / np.abs(wav).max() > threshold).astype(int)
    wav_onset_samps = np.nonzero(np.diff(wav_pulses) == 1)[0] + 1
    wav_onset_times = wav_onset_samps / wav_sfreq

    # compute MEG pulse onset times
    meg_onset_samps = mne.find_events(raw)[:, 0] - raw.first_samp
    meg_onset_times = meg_onset_samps / meg_sfreq

    # compute the offset to align the *first* pulse of each sequence
    start_offset_secs = np.diff(
        [meg_onset_times[0], wav_onset_times[0]])[0]

    if show_plots:
        axs[1].plot(wav_times - start_offset_secs, wav, label='cam')
        axs[1].plot(meg_times, meg + 1.1, label='meg')
        axs[1].set(title='first pulse aligned')

    # correlate the *pause durations* between pulse onsets
    corr = correlate(np.diff(meg_onset_times), np.diff(wav_onset_times),
                     mode='valid')
    # because the reported sample rate is inaccurate, max correlation does not
    # *necessarily* happen when all pulses are aligned, but hopefully we're
    # lucky and it still gives the correct index.
    shift_idx = wav_onset_times.size - meg_onset_times.size - corr.argmax()
    # now we can find the additional offset needed to align *matching* pulses
    match_offset_secs = np.diff(wav_onset_times[[0, shift_idx]])[0]

    if show_plots:
        axs[2].plot(wav_times - start_offset_secs - match_offset_secs, wav,
                    label='cam')
        axs[2].plot(meg_times, meg + 1.1, label='meg')
        axs[2].set(title='correct pulses aligned (at start)')

    # trim or pad *BEGINNING* of WAV
    total_offset_secs = start_offset_secs + match_offset_secs
    total_offset_samps = np.rint(total_offset_secs * wav_sfreq).astype(int)
    if total_offset_samps < 0:
        # zero-pad
        zeros = np.zeros(abs(total_offset_samps), dtype=wav.dtype)
        wav = np.hstack([zeros, wav])
        wav_pulses = np.hstack([zeros.astype(wav_pulses.dtype), wav_pulses])
        wav_times = np.arange(total_offset_samps, wav.size) / wav_sfreq
    elif total_offset_samps > 0:
        # trim. it's possible/likely that there are pulses in the "extra" cam
        # time, just notify the user how many as a sanity check.
        n_pulses = (np.diff(wav_pulses[:total_offset_samps]) == 1).sum()
        if n_pulses != 0:
            print(f'INFO: there were {n_pulses} pulses in the trimmed-off '
                  'beginning part of the camera signal.')
        wav = wav[total_offset_samps:]
        wav_pulses = wav_pulses[total_offset_samps:]
        wav_times = wav_times[:-total_offset_samps]
    # recompute these after padding/trimming
    wav_onset_samps = np.nonzero(np.diff(wav_pulses) == 1)[0] + 1
    wav_onset_times = wav_onset_samps / wav_sfreq
    err = 'CAM pulses end before MEG pulses! CAM shifted too far?'
    assert meg_onset_samps.size <= wav_onset_samps.size, err

    if show_plots:
        axs[3].plot(wav_times, wav, label='cam')
        axs[3].plot(meg_times, meg + 1.1, label='meg')
        axs[3].set(title='after wav beginning padded/trimmed')
        # axs[3].axvline(meg_onset_times[-1], color='C3')
        # axs[3].axvline(wav_onset_times[meg_onset_times.size], color='C4')

    # now figure out how much to "stretch" wav by
    time_stretch = (meg_onset_times[-1] /
                    wav_onset_times[meg_onset_times.size - 1])

    # resample to raw sfreq. This also trims the end. kind='nearest' is suited
    # to square-wave or step-function types of signals.
    interp_func = interp1d(
        x=wav_times, y=wav, kind='nearest', fill_value='extrapolate',
        assume_sorted=True)
    wav_resamp = interp_func(meg_times / time_stretch)

    # re-binarize after resampling
    wav_pulses_resamp = (wav_resamp / np.abs(wav_resamp).max() > threshold
                         ).astype(int)

    if show_plots:
        axs[4].plot(meg_times, wav_resamp, label='cam')
        axs[4].plot(meg_times, meg + 1.1, label='meg')
        axs[4].set(title='after wav resampled & end-trimmed')

    # assert max correlation is within 2 samples of how we've aligned things
    corr1 = correlate(meg, wav_pulses_resamp, mode='full')
    corr_offset = meg.size - corr1.argmax() - 1
    print('INFO: max correlation of MEG & CAM signals differs from final '
          f'alignment by {corr_offset} samples')
    assert corr_offset in (-2, -1, 0, 1, 2)

    # convert wav to Raw and combine with meg pulses
    if show_plots:
        wav_info = mne.create_info(
            ['CAM_BINARY', 'CAM_ORIG'], meg_sfreq, 'stim')
        wav_raw = mne.io.RawArray(
            np.vstack([wav_pulses_resamp, wav_resamp]), wav_info)
        # check alignment
        raw.add_channels([wav_raw], force_update_info=True)
        raw.plot()

    # save the data that matters: the camera shift and stretch values. Casts
    # to float are for cleaner YAML writes (it doesn't handle np.float64 well).
    data = dict(shift=float(total_offset_secs), stretch=float(time_stretch))
    msg = '\n'.join([
        '# "shift" should be subtracted from camera times; resulting times ',
        '# should then be multiplied by "stretch" to yield MEG times.',
        ''])
    with open(sync_dir / f'{fname_stem}.yml', 'w') as f:
        f.write(msg)
        yaml.dump(data, f)
    print()
