"""Evaluate waveform reconstruction/forecasting or person-level clinical AUC."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src._io import ROOT, config, output_dir, write_csv, write_json
from src.preprocessing import load_dataset, forecast_set
from src.models import VARIANTS, SEEDS, PATHS, load_weights
from src.reconstruction import reconstruct, forecast, inverse_y, errors, fit_baselines, predict_baselines
from src.fir import apply_fir, step_to_fir
from src.statistics import source_bootstrap, interval, auc_bootstrap, kernel, holm, delong_raw
from src.tfa import participant_curves
from src.classification import fit_loo


def reconstruction(args, data, out):
    if 'source_record_id' not in data:
        raise ValueError('For JLU examples use run_demo.py; this command evaluates surrogate splits')
    scaler = config()['scaler']
    selected = data['split'] == args.split
    x, y, s, true = [data[k][selected] for k in ['x_z', 'y_z', 'step', 'cbv']]
    target = forecast_set(data, args.split)
    ids = {'same': data['source_record_id'][selected], 'cross': target['source_record_id']}
    error_bank = {}
    fir = inverse_y(apply_fir(x, step_to_fir(s)), scaler)
    error_bank['same', 'F'] = [errors(fir, true)['mse89']]
    context = fit_baselines(target['cal_x_relative'], target['cal_y_relative'], target['cal_x'], target['cal_y'], target['cal_s'])
    baselines = predict_baselines(context, target['target_x_relative'], target['target_x'], scaler)
    for method, pred in baselines.items():
        error_bank['cross', method] = [errors(pred, target['target_y_relative'])['mse89']]
    by_seed = []
    for variant in args.variants:
        for seed in args.seeds:
            net, _ = load_weights(variant, seed, args.device, args.weights_dir)
            for path in PATHS[variant]:
                same = inverse_y(reconstruct(net, x, y, s, path, args.device)['pred'], scaler)
                cross = inverse_y(forecast(net, target['cal_x'], target['cal_y'], target['cal_s'], target['target_x'], path, args.device), scaler)
                for scope, prediction, actual in [('same', same, true), ('cross', cross, target['target_y_relative'])]:
                    mse = errors(prediction, actual)['mse89']
                    error_bank.setdefault((scope, path), []).append(mse)
                    score = float(np.mean([mse[ids[scope] == r].mean() for r in np.unique(ids[scope])]))
                    by_seed.append(dict(scope=scope, method=path, seed=seed, MSE89=score))
            print(f'Evaluated {variant}, seed {seed}', flush=True)
    summary, source_rows = [], []
    for (scope, method), per_seed in error_bank.items():
        mse = np.mean(per_seed, axis=0)
        sources = np.unique(ids[scope])
        values = np.asarray([mse[ids[scope] == r].mean() for r in sources])
        lo, hi = interval(source_bootstrap(values)[:, 0])
        summary.append(dict(scope=scope, method=method, MSE89=float(values.mean()), CI95_low=lo, CI95_high=hi,
                            pairs=len(mse), source_records=len(sources), seeds=len(per_seed), split=args.split,
                            unit='relative_CBFV_squared', aggregation='mean seed errors; equal source weighting'))
        source_rows.extend(dict(scope=scope, method=method, source_record_id=r, MSE89=v) for r, v in zip(sources, values))
    write_csv(out / 'reconstruction_summary.csv', summary)
    write_csv(out / 'reconstruction_by_seed.csv', by_seed)
    write_csv(out / 'source_errors.csv', source_rows)
    write_json(out / 'run.json', dict(status='PASS', task='reconstruction', split=args.split,
               pairs=int(selected.sum()), seeds=args.seeds, variants=args.variants,
               scope='Supplied dataset only; paper reference results use the full dataset'))


def classification(args, data, out):
    if 'person_ids' not in data:
        raise ValueError('Classification requires a clinical NPZ with person_ids')
    unique = sorted(set(zip(data['labels'].tolist(), data['person_ids'].tolist())))
    labels = np.asarray([r[0] for r in unique])
    if set(labels.tolist()) != {0, 1} or min(np.bincount(labels.astype(int))) < 2:
        raise ValueError('The two-person example cannot estimate clinical AUC. Supply a full clinical dataset obtained from hx.yang2@siat.ac.cn.')
    ids = [r[1] for r in unique]
    names = [f'{b}_{m}' for b in ['vlf', 'lf', 'hf'] for m in ['gain', 'phase', 'coherence']]
    gp6 = [0, 1, 3, 4, 6, 7]
    curves, features, people = participant_curves(data)
    assert people == unique
    predictions, records, audits, feature_rows = {}, [], [], []

    def run(name, seed, matrix, columns):
        active_names = [name + '_' + names[j] for j in columns]
        probabilities, folds, _ = fit_loo(labels, matrix[:, columns], active_names, ids, name, seed)
        predictions.setdefault(name, []).append(probabilities)
        records.extend(dict(method=name, seed=seed, person_id=person, label=int(label), probability=float(p))
                       for person, label, p in zip(ids, labels, probabilities))
        audits.extend(folds)
        feature_rows.extend(dict(method=name, seed=seed, person_id=person, label=int(labels[i]),
                                 **{names[j]: float(matrix[i, j]) for j in columns}) for i, person in enumerate(ids))

    run('T_GPC9', 0, features, list(range(9)))
    run('T_GP6', 0, features, gp6)
    for variant in args.variants:
        for seed in args.seeds:
            net, _ = load_weights(variant, seed, args.device, args.weights_dir)
            for path in PATHS[variant]:
                _, matrix, keys = participant_curves(data, net, path, args.device)
                assert keys == unique
                run(path + '_GP6', seed, matrix, gp6)
            print(f'Clinical LOPO {variant}, seed {seed}', flush=True)
    ensemble = {name: np.mean(values, axis=0) for name, values in predictions.items()}
    bootstrap = auc_bootstrap(labels, ensemble)
    rows = []
    for method, values in ensemble.items():
        lo, hi = interval(bootstrap[method])
        rows.append(dict(method=method, people=len(labels), AUC=float(kernel(labels, values).mean()),
                         CI95_low=lo, CI95_high=hi, n_seeds=len(predictions[method])))
    paired = []
    settings = config('evaluation')['statistics']
    for family, comparisons in [('primary_auc', settings['AUC_primary']), ('secondary_auc', settings['AUC_secondary'])]:
        members = []
        for left, right in comparisons:
            row = dict(family=family, left=left, right=right)
            if left in ensemble and right in ensemble:
                row.update(delong_raw(labels, ensemble[left], ensemble[right]))
                lo, hi = interval(bootstrap[left] - bootstrap[right])
                row.update(CI95_low=lo, CI95_high=hi)
            else:
                row.update(raw_p=np.nan, delong_status='comparison unavailable in selected methods')
            members.append(row)
        adjusted = holm([row['raw_p'] for row in members])
        paired.extend(dict(**row, holm_p=value) for row, value in zip(members, adjusted))
    write_csv(out / 'auc_summary.csv', rows)
    write_csv(out / 'paired_auc_Holm.csv', paired)
    write_csv(out / 'oof_by_seed.csv', records)
    write_csv(out / 'oof_ensemble.csv', [dict(person_id=person, label=int(labels[i]),
              **{name: float(p[i]) for name, p in ensemble.items()}) for i, person in enumerate(ids)])
    write_csv(out / 'fold_audit.csv', audits)
    write_csv(out / 'person_features.csv', feature_rows)
    write_json(out / 'run.json', dict(status='PASS', task='classification', people=len(labels),
               controls=int(sum(labels == 0)), patients=int(sum(labels == 1)), seeds=args.seeds,
               bootstrap_unit='person, stratified by label; fixed out-of-fold scores; no refitting'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['reconstruction', 'classification'], default='reconstruction')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--variants', choices=VARIANTS, nargs='+', default=list(VARIANTS))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--weights-dir', type=Path, default=ROOT / 'weights')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/evaluation')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    path = args.data or ROOT / 'data/demo' / ('jlu_demo.npz' if args.task == 'classification' else 'surrogate_demo.npz')
    data = load_dataset(path)
    out = output_dir(args.out)
    {'reconstruction': reconstruction, 'classification': classification}[args.task](args, data, out)


if __name__ == '__main__':
    main()
