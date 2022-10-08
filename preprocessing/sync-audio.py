"""
Find the necessary shift and stretch factors to align the pulses in the video
with the pulses recorded by the MEG.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import yaml
from numpy.polynomial import Polynomial
from scipy.io import wavfile
from scipy.signal import correlate

plt.ion()

# config
config_file = Path(__file__).parent.parent / 'config.yml'

with open(config_file, 'r') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)

stim_ch = cfg['meg_ttl_channel']
raw_dir = Path(cfg['raw_folder'])
audio_dir = Path(cfg['audio_folder'])

mne.set_log_level('error')

for _dict in cfg['files']:
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

    fig, axs = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(20, 4))
    axs[0].plot(wav_times, wav, label='cam')
    axs[0].plot(meg_times, meg + 1.1, label='meg')
    axs[0].set(title='raw signals')

    # compute WAV pulse onset times
    wav_pulses = np.rint(wav / wav.max()).astype(int)
    wav_pulses[wav_pulses < 0] = 0  # ignore falling-edge undershoot
    wav_onset_samps = np.nonzero(np.diff(wav_pulses) == 1)[0] + 1
    wav_onset_times = wav_onset_samps / wav_sfreq

    # compute MEG pulse onset times
    meg_onset_samps = mne.find_events(raw)[:, 0] - raw.first_samp
    meg_onset_times = meg_onset_samps / meg_sfreq

    # compute the offset to align the *first* pulse of each sequence
    start_offset_secs = np.diff(
        [meg_onset_times[0], wav_onset_times[0]])[0]

    axs[1].plot(wav_times - start_offset_secs, wav, label='cam')
    axs[1].plot(meg_times, meg + 1.1, label='meg')
    axs[1].set(title='first pulse aligned')

    # correlate the *pause durations* between pulse onsets
    corr = correlate(np.diff(meg_onset_times), np.diff(wav_onset_times),
                     mode='valid')
    # because the reported sample rate is inaccurate, max correlation happens
    # not when all pulses are aligned, but when the *middle* pulse is aligned
    # (and thus the early/late pulses are off by as little as possible).
    # Therefore the shift to get the *first* pulse aligned will be *half* the
    # reported argmax.
    shift_idx = corr.argmax() // 2
    # now we can find the additional offset needed to align *matching* pulses
    match_offset_secs = np.diff(wav_onset_times[[0, shift_idx]])[0]

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
        # trim
        n_pulses = (np.diff(wav_pulses[:total_offset_samps]) == 1).sum()
        print(f'WARNING: there were {n_pulses} pulses in the trimmed-off '
              'beginning part of the camera signal.')
        wav = wav[total_offset_samps:]
        wav_pulses = wav_pulses[total_offset_samps:]
        wav_times = wav_times[:-total_offset_samps]
    # recompute these after padding/trimming
    wav_onset_samps = np.nonzero(np.diff(wav_pulses) == 1)[0] + 1
    wav_onset_times = wav_onset_samps / wav_sfreq
    err = 'CAM pulses end before MEG pulses! CAM shifted too far?'
    assert meg_onset_samps.size <= wav_onset_samps.size, err

    axs[3].plot(wav_times, wav, label='cam')
    axs[3].plot(meg_times, meg + 1.1, label='meg')
    axs[3].set(title='after wav beginning padded/trimmed')

    # now figure out how much to "stretch" wav by
    time_stretch = meg_onset_times[-1] / wav_onset_times[meg_onset_times.size]

    # resample to raw sfreq
    sfreq_ratio = meg_sfreq / wav_sfreq * time_stretch
    wav_resamp = mne.filter.resample(wav, up=sfreq_ratio, npad='auto')
    wav_resamp_times = np.arange(wav_resamp.size) / meg_sfreq

    axs[4].plot(wav_resamp_times, wav_resamp, label='cam')
    axs[4].plot(meg_times, meg + 1.1, label='meg')
    axs[4].set(title='after wav resampled')

    # TODO above subplot still looks wrong

    raise RuntimeError

    # re-binarize after resampling
    wav_pulses_resamp = np.rint(wav_resamp).astype(int)
    wav_pulses_resamp[wav_pulses_resamp < 0] = 0

    # trim or pad *END* of WAV
    diff = meg.size - wav_resamp.size
    if diff < 0:
        # here it's possible that there are pulses in the "extra" cam time
        # so we merely notify, not assert/error
        if wav_pulses_resamp[diff:].any():
            n_pulses = (np.diff(wav_pulses_resamp[diff:]) == 1).sum()
            print(f'WARNING: there were {n_pulses} pulses in the trimmed-off '
                  'end part of the camera signal.')
        wav_pulses_resamp = wav_pulses_resamp[:diff]
        wav_resamp = wav_resamp[:diff]
    elif diff > 0:
        zeros = np.zeros(diff, dtype=wav_pulses_resamp.dtype)
        wav_pulses_resamp = np.hstack([wav_pulses_resamp, zeros])
        wav_resamp = np.hstack([wav_resamp, zeros.astype(wav_resamp.dtype)])

    # convert wav to Raw and combine with meg pulses
    wav_info = mne.create_info(['CAM_PULSES', 'CAM_SYNC'], meg_sfreq, 'stim')
    wav_raw = mne.io.RawArray(np.vstack([wav_pulses_resamp, wav_resamp]), wav_info)
    # check alignment
    raw.add_channels([wav_raw], force_update_info=True)
    raw.plot()
