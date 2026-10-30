"""Matched training of four configurations and five seeds on a schema NPZ."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src._io import ROOT, config, output_dir, write_csv, write_json, sha256
from src.models import AblationNet, VARIANTS, SEEDS, SELECT_PATH, epoch_order, epoch_noise
from src.losses import LOSS_FIELDS
from src.preprocessing import load_dataset, forecast_set
from src.reconstruction import forecast, inverse_y, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/demo/surrogate_demo.npz')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/training')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true', help='Two epochs and at most 2,000 training pairs per run')
    args = parser.parse_args()
    if args.epochs < 10 and not args.smoke:
        parser.error('Use at least 10 epochs, or --smoke for a short execution check')
    torch.set_num_threads(args.threads)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    cfg = config()
    data = load_dataset(args.data)
    val = forecast_set(data, 'val')
    indices = np.flatnonzero(data['split'] == 'train')
    if args.smoke:
        indices = indices[:2000]
        val = {k: v[:20] for k, v in val.items()}
    device = torch.device(args.device)
    x = torch.as_tensor(data['x_z'][indices], device=device)
    y = torch.as_tensor(data['y_z'][indices], device=device)
    s = torch.as_tensor(data['step'][indices], device=device)
    out = output_dir(args.out)
    max_epochs = 2 if args.smoke else args.epochs
    runs = []
    for variant in args.variants:
        for seed in args.seeds:
            folder = out / variant / f'seed{seed}'
            if folder.exists():
                raise FileExistsError('Choose a fresh output directory; existing training is preserved: ' + str(folder))
            folder.mkdir(parents=True)
            net = AblationNet(variant, seed).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=cfg['training']['lr'],
                                         betas=tuple(cfg['training']['adam_betas']), eps=cfg['training']['adam_eps'],
                                         weight_decay=cfg['training']['weight_decay'])
            history, best = [], None
            for epoch in range(1, max_epochs + 1):
                net.train()
                beta = cfg['model']['beta'] * min(epoch / cfg['training']['warmup_epochs'], 1.)
                order = torch.as_tensor(epoch_order(len(x), seed, epoch), device=device)
                em = epoch_noise(len(x), seed, epoch, 'main', device) if net.em is not None else None
                es = epoch_noise(len(x), seed, epoch, 'aux', device) if net.es is not None else None
                totals = torch.zeros(len(LOSS_FIELDS), dtype=torch.float64, device=device)
                for start in range(0, len(x), cfg['training']['batch_size']):
                    ix = order[start:start + cfg['training']['batch_size']]
                    stop = start + len(ix)
                    losses = net.objective(y[ix], x[ix], s[ix], beta,
                                           None if em is None else em[start:stop], None if es is None else es[start:stop])
                    optimizer.zero_grad(set_to_none=True)
                    losses['total'].backward()
                    optimizer.step()
                    totals += torch.stack([losses[k].detach() for k in LOSS_FIELDS]).double() * len(ix)
                values = (totals / len(x)).cpu().numpy()
                if not np.isfinite(values).all():
                    raise FloatingPointError('Nonfinite training objective')
                row = dict(epoch=epoch, beta=beta, **dict(zip(LOSS_FIELDS, map(float, values))))
                if args.smoke or epoch % cfg['training']['checkpoint_every'] == 0:
                    net.eval()
                    pred = forecast(net, val['cal_x'], val['cal_y'], val['cal_s'], val['target_x'], SELECT_PATH[variant], str(device))
                    mse = errors(inverse_y(pred, cfg['scaler']), val['target_y_relative'])['mse89']
                    score = float(np.mean([mse[val['source_record_id'] == r].mean() for r in np.unique(val['source_record_id'])]))
                    row['val_cross_mse89_relative'] = score
                    checkpoint = folder / f'epoch{epoch:03d}.pt'
                    torch.save(dict(variant=variant, seed=seed, epoch=epoch, state_dict=net.state_dict(),
                                    scaler=cfg['scaler'], smoke=args.smoke, validation_MSE89_relative=score), checkpoint)
                    if best is None or score < best['score']:
                        best = dict(epoch=epoch, score=score, filename=checkpoint.name, sha256=sha256(checkpoint))
                    print(f'{variant} seed {seed}, epoch {epoch}: val MSE89={score:.8g}', flush=True)
                history.append(row)
                write_csv(folder / 'history.csv', history)
            record = dict(status='PASS', variant=variant, seed=seed, epochs=max_epochs,
                          training_pairs=len(indices), data_sha256=sha256(args.data), smoke=args.smoke,
                          selected=best, scope='Training on supplied data; bundled 2.95% examples do not reproduce full-data paper models')
            write_json(folder / 'run.json', record)
            runs.append(record)
    write_json(out / 'training_summary.json', dict(status='PASS', runs=runs))


if __name__ == '__main__':
    main()
