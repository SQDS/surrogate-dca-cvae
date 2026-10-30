"""Relative signal arrays, train-only scaling and source-preserving windows."""
from pathlib import Path
import numpy as np
from ._io import config
from .fir import fir_step


def normalize(abp, cbv, scaler=None):
    signals = (scaler or config()['scaler'])['signals']
    return tuple(((np.asarray(a) - signals[k]['mean']) / signals[k]['std']).astype(np.float32)
                 for a, k in [(abp, 'abp'), (cbv, 'cbv')])


def load_dataset(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    if not {'abp', 'cbv', 'fs_hz'} <= data.keys():
        raise ValueError('Required fields: abp, cbv, fs_hz; see data/schema.json')
    x, y = data['abp'], data['cbv']
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] != 100 or not len(x):
        raise ValueError('ABP and CBFV must be paired, nonempty N x 100 arrays')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or float(data['fs_hz']) != 1.:
        raise ValueError('Signals must be finite relative perturbations sampled at 1 Hz')
    n = len(x)
    if 'source_record_id' in data:
        required = ['source_record_id', 'segment', 'ari', 'split']
        if any(k not in data or data[k].shape != (n,) for k in required):
            raise ValueError('Missing or invalid surrogate identities/splits')
        if set(data['split'].tolist()) != {'train', 'val', 'test'}:
            raise ValueError('Surrogate data must contain train, val and test splits')
        ids = {s: set(data['source_record_id'][data['split'] == s]) for s in ['train', 'val', 'test']}
        if ids['train'] & ids['val'] or ids['train'] & ids['test'] or ids['val'] & ids['test']:
            raise ValueError('A source record appears in more than one split')
        keys = np.column_stack([data[k] for k in ['source_record_id', 'segment', 'ari']])
        if len(np.unique(keys, axis=0)) != n:
            raise ValueError('Duplicate record/segment/ARI pairs')
        order = np.lexsort((data['ari'], data['segment'], data['source_record_id']))
        if n % 10 or not np.all(data['ari'][order].reshape(-1, 10) == np.arange(10)):
            raise ValueError('Every pressure window must retain ARI 0–9')
        grouped = keys[order].reshape(-1, 10, 3)
        if not np.all(grouped[:, :, :2] == grouped[:, :1, :2]):
            raise ValueError('Incomplete ARI group for a pressure window')
    elif 'person_ids' in data:
        for k in ['person_ids', 'record_ids', 'labels', 'sides', 'window_start_sec']:
            if k not in data or data[k].shape != (n,):
                raise ValueError('Missing or invalid clinical field: ' + k)
        if not set(data['labels'].tolist()) <= {0, 1}:
            raise ValueError('Clinical labels must be 0 (control) or 1 (stroke)')
        if not set(data['sides'].tolist()) <= {'L', 'R'}:
            raise ValueError('Clinical sides must be L or R')
        for person in np.unique(data['person_ids']):
            if len(np.unique(data['labels'][data['person_ids'] == person])) != 1:
                raise ValueError('Inconsistent labels for the same participant')
        for record in np.unique(data['record_ids']):
            mask = data['record_ids'] == record
            if len(np.unique(data['person_ids'][mask])) != 1:
                raise ValueError('A record ID cannot belong to multiple participants')
            sides = set(data['sides'][mask].tolist())
            label = int(data['labels'][mask][0])
            if label == 0 and sides != {'L', 'R'}:
                raise ValueError('The manuscript protocol requires bilateral control recordings')
            if label == 1 and len(sides) != 1:
                raise ValueError('Select only the symptom-ipsilateral side of each stroke recording before inference')
    else:
        raise ValueError('Expected surrogate source_record_id or clinical person_ids')
    xz, yz = normalize(x, y)
    if 'step' not in data:
        data['step'] = fir_step(xz, yz)[1]
    if data['step'].shape != (n, 12) or not np.isfinite(data['step']).all():
        raise ValueError('Step responses must be finite N x 12 arrays')
    data['x_z'], data['y_z'] = xz, yz
    return data


def forecast_set(data, split):
    from .reconstruction import build_pairs
    if 'source_record_id' not in data:
        raise ValueError('This split-based forecast interface expects surrogate data')
    accepted, excluded, cal, target = build_pairs(data['source_record_id'], data['segment'],
                                                 data['ari'], data['split'] == split)
    if not accepted:
        raise ValueError('No complete five-calibration/one-target groups in split ' + split)
    return dict(cal_x=data['x_z'][cal], cal_y=data['y_z'][cal], cal_s=data['step'][cal],
                target_x=data['x_z'][target], target_y_relative=data['cbv'][target],
                cal_x_relative=data['abp'][cal], cal_y_relative=data['cbv'][cal],
                target_x_relative=data['abp'][target], source_record_id=data['source_record_id'][target],
                ari=data['ari'][target], cal_rows=cal, target_rows=target)
