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
    summary, source_rows, source_bank, bootstrap_bank = [], [], {}, {}
    for (scope, method), per_seed in error_bank.items():
        mse = np.mean(per_seed, axis=0)
        sources = np.unique(ids[scope])
        values = np.asarray([mse[ids[scope] == r].mean() for r in sources])
        draws = source_bootstrap(values)[:, 0]
        source_bank[scope, method] = values
        bootstrap_bank[scope, method] = draws
        lo, hi = interval(draws)
        seed_scores = [row['MSE89'] for row in by_seed if row['scope'] == scope and row['method'] == method]
        summary.append(dict(scope=scope, method=method, MSE89=float(values.mean()), CI95_low=lo, CI95_high=hi,
                            pairs=len(mse), source_records=len(sources), seeds=len(per_seed), split=args.split,
                            seed_MSE89_SD=float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else '',
                            unit='relative_CBFV_squared', aggregation='mean seed errors; equal source weighting'))
        source_rows.extend(dict(scope=scope, method=method, source_record_id=r, MSE89=v) for r, v in zip(sources, values))
    write_csv(out / 'reconstruction_summary.csv', summary)
    write_csv(out / 'reconstruction_by_seed.csv', by_seed)
    write_csv(out / 'source_errors.csv', source_rows)
    paired = []
    comparisons = config('evaluation')['statistics']['MSE_primary']
    for left, right in comparisons:
        row = dict(scope='cross', left=left, right=right, family_size=len(comparisons))
        a, b = ('cross', left), ('cross', right)
        if a in source_bank and b in source_bank:
            difference = bootstrap_bank[a] - bootstrap_bank[b]
            lo, hi = interval(difference)
            adjusted_lo, adjusted_hi = interval(difference, 1 - .05 / len(comparisons))
            row.update(delta_MSE89=float(np.mean(source_bank[a] - source_bank[b])),
                       CI95_low=lo, CI95_high=hi, CI98333_low=adjusted_lo, CI98333_high=adjusted_hi,
                       status='ok', direction='negative favors left')
        else:
            row['status'] = 'comparison unavailable in selected methods'
        paired.append(row)
    write_csv(out / 'paired_prediction_intervals.csv', paired)
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
    designs = {'GP6': gp6, 'G3': [0, 3, 6], 'P3': [1, 4, 7],
               'VLF_GP2': [0, 1], 'LF_GP2': [3, 4], 'HF_GP2': [6, 7]}
    curves, features, people = participant_curves(data)
    assert people == unique
    predictions, records, audits, feature_rows = {}, [], [], []

    def run(name, seed, matrix, columns, matrix_names=None):
        feature_names = matrix_names or names
        active_names = [name + '_' + feature_names[j] for j in columns]
        probabilities, folds, _ = fit_loo(labels, matrix[:, columns], active_names, ids, name, seed)
        predictions.setdefault(name, []).append(probabilities)
        records.extend(dict(method=name, seed=seed, person_id=person, label=int(label), probability=float(p))
                       for person, label, p in zip(ids, labels, probabilities))
        audits.extend(folds)
        feature_rows.extend(dict(method=name, seed=seed, person_id=person, label=int(labels[i]),
                                 **{feature_names[j]: float(matrix[i, j]) for j in columns}) for i, person in enumerate(ids))

    run('T_GPC9', 0, features, list(range(9)))
    for suffix, columns in designs.items():
        run('T_' + suffix, 0, features, columns)
    _, fir_features, fir_people = participant_curves(data, baseline='F')
    assert fir_people == unique
    for suffix, columns in designs.items():
        run('F_' + suffix, 0, fir_features, columns)
    for variant in args.variants:
        for seed in args.seeds:
            net, _ = load_weights(variant, seed, args.device, args.weights_dir)
            for path in PATHS[variant]:
                _, matrix, keys = participant_curves(data, net, path, args.device)
                assert keys == unique
                for suffix, columns in designs.items():
                    run(path + '_' + suffix, seed, matrix, columns)
                if path == 'MS-aux':
                    combined = np.column_stack([matrix[:, gp6], features])
                    combined_names = ['MS-aux_' + names[j] for j in gp6] + ['T_' + name for name in names]
                    run('Fusion15', seed, combined, list(range(15)), combined_names)
            print(f'Clinical LOPO {variant}, seed {seed}', flush=True)
    ensemble = {name: np.mean(values, axis=0) for name, values in predictions.items()}
    bootstrap = auc_bootstrap(labels, ensemble)
    rows = []
    for method, values in ensemble.items():
        lo, hi = interval(bootstrap[method])
        seed_aucs = [float(kernel(labels, p).mean()) for p in predictions[method]]
        rows.append(dict(method=method, people=len(labels), AUC=float(kernel(labels, values).mean()),
                         CI95_low=lo, CI95_high=hi, n_seeds=len(predictions[method]),
                         mean_seed_AUC=float(np.mean(seed_aucs)) if len(seed_aucs) > 1 else '',
                         seed_AUC_SD=float(np.std(seed_aucs, ddof=1)) if len(seed_aucs) > 1 else '',
                         scope='main' if method.endswith('GP6') or method == 'T_GPC9' else 'exploratory'))
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
                if family == 'primary_auc':
                    lo, hi = interval(bootstrap[left] - bootstrap[right], 1 - .05 / len(comparisons))
                    row.update(CI98333_low=lo, CI98333_high=hi)
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
