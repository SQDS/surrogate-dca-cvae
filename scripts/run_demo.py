"""Run the bundled two-person JLU example and surrogate cross-window example."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src._io import ROOT, output_dir, write_csv, write_json, config
from src.models import load_weights
from src.preprocessing import load_dataset, forecast_set
from src.reconstruction import reconstruct, inverse_y, errors, forecast
from src.tfa import participant_curves, FREQ
from src.statistics import aggregate
from plot_results import plot_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/demo')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = output_dir(args.out)
    data = load_dataset(ROOT / 'data/demo/jlu_demo.npz')
    surrogate = load_dataset(ROOT / 'data/demo/surrogate_demo.npz')
    test = forecast_set(surrogate, 'test')
    scaler = config()['scaler']
    predictions, curves, seed_errors, surrogate_mse = [], [], [], []
    for seed in range(42, 47):
        net, _ = load_weights('MS', seed, args.device)
        pred = inverse_y(reconstruct(net, data['x_z'], data['y_z'], data['step'], 'MS-aux', args.device)['pred'], scaler)
        predictions.append(pred)
        seed_errors.append(errors(pred, data['cbv']))
        curve, _, people = participant_curves(data, net, 'MS-aux', args.device)
        curves.append(curve)
        forecast_pred = inverse_y(forecast(net, test['cal_x'], test['cal_y'], test['cal_s'], test['target_x'], 'MS-aux', args.device), scaler)
        surrogate_mse.append(errors(forecast_pred, test['target_y_relative'])['mse89'])
        print(f'MS seed {seed}: reconstruction, probes and forecast complete', flush=True)
    traditional, _, traditional_people = participant_curves(data)
    assert people == traditional_people
    mean_pred = np.mean(predictions, axis=0)
    mean_curves = aggregate(np.stack(curves), axis=0)
    mean_correlation = np.mean([e['correlation89'] for e in seed_errors], axis=0)
    mean_error = np.mean([e['mse89'] for e in seed_errors], axis=0)
    rows = []
    for label, person in people:
        selected = data['person_ids'] == person
        def side_mean(values):
            return float(np.mean([values[selected & (data['sides'] == side)].mean()
                                  for side in np.unique(data['sides'][selected])]))
        rows.append(dict(person_id=person, label=label, windows=int(selected.sum()),
                         mean_correlation89=side_mean(mean_correlation), mean_mse89=side_mean(mean_error),
                         scope='curated same-window examples; mean individual-seed metrics'))
    write_csv(out / 'jlu_metrics.csv', rows)
    np.savez_compressed(out / 'demo_outputs.npz', true=data['cbv'], pred=mean_pred,
                        person_ids=data['person_ids'], sides=data['sides'], starts=data['window_start_sec'],
                        correlation89=mean_correlation, frequencies=FREQ, model_curves=mean_curves,
                        traditional=traditional, curve_person_ids=np.asarray([p[1] for p in people]))
    curve_rows = []
    for source, values in [('MS-aux', mean_curves), ('traditional', traditional)]:
        for i, (_, person) in enumerate(people):
            for j, frequency in enumerate(FREQ):
                curve_rows.append(dict(method=source, person_id=person, frequency_hz=frequency,
                                       gain=values[i, j, 0], phase_degrees=values[i, j, 1], coherence=values[i, j, 2]))
    write_csv(out / 'jlu_tfa.csv', curve_rows)
    write_json(out / 'run.json', dict(status='PASS', device=args.device, seeds=list(range(42, 47)),
               clinical_people=len(people), clinical_windows=len(data['abp']),
               surrogate_test_forecast_pairs=len(test['target_x']),
               surrogate_test_forecast_sources=len(np.unique(test['source_record_id'])),
               surrogate_MSE89_mean_seed_error=float(np.mean(surrogate_mse)),
               scope='Bundled examples only. Full paper results are in results/reference_summary.csv.'))
    plot_demo(out)
    print('Demo complete:', out)


if __name__ == '__main__':
    main()
