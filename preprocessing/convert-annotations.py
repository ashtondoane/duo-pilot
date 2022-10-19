"""
Converts camera annotation files into MNE-Python annotation objects, correcting
for mismatched time bases of the camera and the MEG system.
"""
import os
import re
import warnings
from pathlib import Path, PureWindowsPath

import mne
import numpy as np
import pandas as pd
import yaml


def simplify_descriptions(description):
    for descr in ('attend', 'ignore', 'cry'):
        if description.startswith(descr):
            return descr
    return description


# config
config_file = Path(__file__).resolve().parent.parent / 'config.yml'

with open(config_file, 'r') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)

raw_dir = Path(cfg['raw_folder'])
annot_dir = Path(cfg['annot_folder'])
sync_dir = Path(cfg['audio_folder']).parent / 'sync'
raw_annot_dir = annot_dir.parent / 'raw-annot'
os.makedirs(raw_annot_dir, exist_ok=True)

for _dict in cfg['files']:
    if _dict['ann'] is None or _dict['cam'] is None or _dict['raw'] is None:
        continue
    print(f'INFO: processing {Path(_dict["raw"]).parts[0]}')
    # make sure the camera file used for sync and the camera file used for
    # annotation were the same camera
    pattern = re.compile(r'cam[\-_\w]?\d', re.IGNORECASE)
    matches = pattern.findall(_dict['ann'])
    assert len(matches) == 1, 'cannot determine which camera was annotated'
    pattern = re.compile(matches[0], re.IGNORECASE)  # now has specific digit
    matches = pattern.findall(_dict['cam'])
    assert len(matches) == 1, 'sync camera and annotation camera mismatch'
    # load annotations. first make sure the header rows give what we expect.
    ann_path = annot_dir / _dict['ann']
    with open(ann_path, 'r') as f:
        line0 = f.readline().strip().strip('"#')
        line1 = f.readline().strip().split('\t')
    # the camera annotated doesn't necessarily match the camera that had the
    # audio sync data on it, so just check for the right subject number
    fname_from_header = PureWindowsPath(line0.split()[0].strip('file:/'))
    # cast to int to make sure we grabbed the right characters
    if _dict['cam'] is not None:
        subj_num = int(_dict['cam'][3:6])
        assert str(subj_num) in fname_from_header.name
    # now make sure the column names are consistent across files
    assert line1 == ['Begin Time - msec', 'End Time - msec', 'Duration - msec',
                     'default', 'attend', 'ignore']
    # clean up header. there aren't really 3 separate columns for
    # default/attend/ignore, so replace with a generic heading
    header = ['onset', 'offset', 'duration', 'description']
    # load just the data rows; add header; convert milliseconds to seconds
    ann = pd.read_table(ann_path, skiprows=2, delim_whitespace=True)
    ann.columns = header
    ann.iloc[:, :3] /= 1000
    np.testing.assert_array_almost_equal(ann['offset'] - ann['onset'],
                                         ann['duration'])
    # apply the shift and stretch from the pulse data.
    # "shift" should be subtracted from camera times; resulting times
    # should then be multiplied by "stretch" to yield MEG times.
    with open(sync_dir / f'{Path(_dict["cam"]).stem}.yml', 'r') as f:
        time_correction = yaml.load(f, Loader=yaml.SafeLoader)
    ann.iloc[:, :2] -= time_correction['shift']
    ann.iloc[:, :2] *= time_correction['stretch']
    ann['duration'] = ann['offset'] - ann['onset']
    # unify descriptions: attend1, attend01, attend2, etc → "attend"
    ann['description'] = ann['description'].map(simplify_descriptions)
    annotations = mne.Annotations(onset=ann['onset'], duration=ann['duration'],
                                  description=ann['description'])
    # load raw
    raw_path = raw_dir / _dict['raw']
    raw = mne.io.read_raw_fif(raw_path, preload=False, allow_maxshield=True,
                              verbose='ERROR')
    # set annotations and save. Here we set annotations just to make sure it
    # works (annotation times don't fall outside the range of raw). We save the
    # annotations separately for later use.
    with warnings.catch_warnings(record=True) as w:
        # ↓↓↓ this is likely too strict, given that our time warping is based
        # ↓↓↓ on a different camera than the one annotated
        # warnings.simplefilter(action='error')
        raw.set_annotations(annotations)
    for warning in w:
        print(f'WARN: {w[0].message}')
    # save annotations
    annotations.save(raw_annot_dir / f'{raw_path.stem}-annotations.csv',
                     overwrite=True, verbose=False)
