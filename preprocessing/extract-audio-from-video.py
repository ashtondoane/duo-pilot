from pathlib import Path
from subprocess import run

import yaml

from scipy.io import wavfile

# config
sync_channel = 0  # left=0, right=1
config_file = Path(__file__).parent.parent / 'config.yml'

with open(config_file, 'r') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)


for _dict in cfg['files']:
    video_path = Path(cfg['video_folder']) / _dict['cam']
    # determine audio file exension
    result = run(['ffprobe',
                  '-loglevel', 'error',
                  '-select_streams', 'a:0',  # audio only
                  '-show_entries', 'stream=codec_name,channels',
                  '-print_format', 'default=noprint_wrappers=1:nokey=1',
                  video_path],
                 capture_output=True)
    codec, n_chan = [x.lower() for x in result.stdout.decode().split()]
    n_chan = int(n_chan)
    if codec.startswith('pcm'):
        audio_ext = 'wav'
    elif codec.startswith('aac'):
        audio_ext = 'aac'  # or m4a?
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

    # extract only the channel with triggers in it. Assumes that the other
    # channel is silent(ish) or at any rate much quieter.
    sfreq, data = wavfile.read(audio_path)
    if data.ndim == 2:
        idx = data.max(axis=0).argmax()
        data = data[:, idx]
    wavfile.write(audio_path, sfreq, data)
