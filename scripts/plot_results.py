"""Draw figures from demo outputs or evaluation summary tables."""
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src._io import ROOT, read_csv


def plot_demo(folder):
    folder = Path(folder)
    with np.load(folder / 'demo_outputs.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    people = np.unique(data['person_ids'])
    fig, axes = plt.subplots(len(people), 1, figsize=(10, 3 * len(people)), squeeze=False)
    for ax, person in zip(axes[:, 0], people):
        rows = np.flatnonzero(data['person_ids'] == person)
        scores = data['correlation89'][rows]
        index = rows[np.argmin(abs(scores - np.nanmedian(scores)))]
        time = np.arange(100)
        ax.plot(time, data['true'][index], label='Measured CBFV', color='#253858', lw=1.8)
        ax.plot(time, data['pred'][index], label='MS-aux: mean of five seed outputs', color='#d66d30', lw=1.5)
        ax.axvspan(0, 10, color='grey', alpha=.1)
        ax.set(title=f'{person} | side {data["sides"][index]} | window start {data["starts"][index]} s',
               xlabel='Time within window (s)', ylabel='Relative CBFV')
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle('Curated JLU examples — same-window reconstruction', fontsize=13)
    fig.tight_layout()
    fig.savefig(folder / 'jlu_reconstruction.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(len(people), 3, figsize=(12, 3 * len(people)), squeeze=False)
    for i, person in enumerate(data['curve_person_ids']):
        for j, name in enumerate(['Relative gain', 'Phase (degrees)', 'Coherence']):
            ax = axes[i, j]
            ax.plot(data['frequencies'], data['traditional'][i, :, j], label='Conventional TFA', color='#253858')
            ax.plot(data['frequencies'], data['model_curves'][i, :, j], label='MS-aux probes', color='#d66d30')
            ax.set(title=f'{person}: {name}', xlabel='Frequency (Hz)', ylabel=name)
            if j == 2: ax.set_ylim(-.03, 1.03)
            ax.grid(alpha=.18)
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / 'jlu_tfa.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'outputs/demo')
    args = parser.parse_args()
    if (args.input / 'demo_outputs.npz').exists():
        plot_demo(args.input)
    else:
        rows = read_csv(args.input / 'reconstruction_summary.csv')
        groups = sorted(set(r['scope'] for r in rows))
        fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4), squeeze=False)
        for ax, scope in zip(axes[0], groups):
            selected = [r for r in rows if r['scope'] == scope]
            ax.bar([r['method'] for r in selected], [float(r['MSE89']) for r in selected], color='#427c9b')
            ax.set(title=scope, ylabel='Relative CBFV MSE, samples 11:100')
            ax.tick_params(axis='x', rotation=35)
        fig.tight_layout()
        fig.savefig(args.input / 'reconstruction_summary.png', dpi=170)
        plt.close(fig)
    print('Figures saved in', args.input)


if __name__ == '__main__':
    main()
