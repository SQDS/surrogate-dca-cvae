"""Repository-relative configuration, tables and file integrity."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def config(name='models'):
    return json.loads((ROOT / 'configs' / (name + '.json')).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('No rows to write: ' + str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def output_dir(path):
    path = Path(path).expanduser().resolve()
    for name in ['data', 'weights', 'src', 'configs', 'simulation', 'scripts', 'results', 'docs']:
        frozen = ROOT / name
        if path == frozen or frozen in path.parents:
            raise ValueError('Choose an output directory outside the shipped assets: ' + str(path))
    if path == ROOT:
        raise ValueError('Use an output subdirectory, not the repository root')
    path.mkdir(parents=True, exist_ok=True)
    return path
