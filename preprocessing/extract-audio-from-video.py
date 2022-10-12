"""
Extracts an audio channel from a video file. Which channel gets kept (left or
right) is determined by the config file value "audio_sync_channel".
"""
from pathlib import Path
from subprocess import run

import yaml
from scipy.io import wavfile

# config
config_file = Path(__file__).resolve().parent.parent / 'config.yml'

with open(config_file, 'r') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)

chan_idx = cfg['audio_sync_channel']

for _dict in cfg['files']:
    if _dict['cam'] is None:
        continue
    video_path = Path(cfg['video_folder']) / _dict['cam']
    # determine audio file exension
    result = run(['ffprobe',
                  '-loglevel', 'error',
                  '-select_streams', 'a:0',  # audio only
                  '-show_entries', 'stream=codec_name',
                  '-print_format', 'default=noprint_wrappers=1:nokey=1',
                  video_path],
                 capture_output=True)
    codec = result.stdout.decode().lower()
    if codec.startswith('pcm'):
        audio_ext = 'wav'
    else:
        raise NotImplementedError(f'unsupported audio codec {codec}')
    # extract audio channel from each video
    audio_path = Path(cfg['audio_folder']) / f'{video_path.stem}.{audio_ext}'
    result = run(['ffmpeg',
                  '-y',  # overwrite destination files
                  '-i', video_path,  # input file
                  '-codec', 'copy',  # "stream copy" mode, AKA don't transcode
                  # select ↓↓↓ all audio channels
                  '-map', '0:a', audio_path
                  ])

    # extract only the channel with triggers in it
    sfreq, data = wavfile.read(audio_path)
    if data.ndim == 2:
        data = data[:, chan_idx]
    wavfile.write(audio_path, sfreq, data)
