"""Verify bundled weights, or fetch the same files from an explicit release URL."""
import argparse
import os
from pathlib import Path
import sys
from urllib.parse import urlparse, quote
from urllib.request import urlopen
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src._io import ROOT, read_csv, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', help='Actual HTTPS release asset directory; no URL is assumed')
    parser.add_argument('--weights-dir', type=Path, default=ROOT / 'weights')
    parser.add_argument('--variant', choices=['M', 'S', 'S-Y', 'MS'])
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.base_url and urlparse(args.base_url).scheme != 'https':
        parser.error('--base-url must use HTTPS')
    entries = read_csv(args.weights_dir / 'manifest.csv')
    entries = [r for r in entries if (not args.variant or r['variant'] == args.variant)
               and (args.seed is None or int(r['seed']) == args.seed)]
    if not entries:
        parser.error('No matching weight manifest entries')
    for row in entries:
        dest = args.weights_dir / row['filename']
        if dest.resolve().parent != args.weights_dir.resolve():
            raise ValueError('Invalid manifest filename')
        if dest.exists() and sha256(dest) == row['sha256']:
            print('OK', row['filename'])
            continue
        if not args.base_url:
            raise RuntimeError(f'Missing or altered weight: {dest.name}; supply --base-url for a real release')
        temp = dest.with_suffix('.part')
        try:
            with urlopen(args.base_url.rstrip('/') + '/' + quote(dest.name), timeout=60) as response, temp.open('wb') as handle:
                while block := response.read(1024**2):
                    handle.write(block)
            if temp.stat().st_size != int(row['bytes']) or sha256(temp) != row['sha256']:
                raise ValueError('Downloaded weight failed size/SHA256 verification: ' + dest.name)
            os.replace(temp, dest)
        finally:
            temp.unlink(missing_ok=True)
        print('DOWNLOADED AND VERIFIED', dest.name)
    print(f'Verified {len(entries)} checkpoints')


if __name__ == '__main__':
    main()
