# Reproduction scope

This repository follows the author-supplied manuscript **Toward Digital Twin–Enabled Assessment of Dynamic Cerebral Autoregulation: Learning From Surrogate Pressure–Flow Data**. The public examples are deliberately smaller than the full study datasets.

| Manuscript analysis | Full-study population | Repository support |
|---|---|---|
| Matched neural development | Four configurations × five seeds; 393,700 total surrogate pairs split by source record | All 20 selected weights; `train_models.py` uses the supplied schema dataset |
| Stroke/control discrimination, Tables 1 and 2 | 139 participants: 68 controls, 71 stroke; 140 recordings | `evaluate.py --task classification`; full clinical windows needed; directly fitted FIR and spectral TFA comparators included |
| Same-window surrogate reconstruction | 59,080 pairs from 1,067 test sources | `evaluate.py --task reconstruction`, `scope=same` |
| Independent-window surrogate prediction, Table 2 | 354 test sources and 3,540 ARI-specific targets | Same command, `scope=cross`; five reference windows spanning 0–300 s, a 50-s gap, and target ABP at 350–450 s |
| Supplementary measured-CBFV prediction | 137 participants | Archived aggregate results in `results/clinical_prediction_summary.csv`; this separate raw-record pipeline is not implemented by the public evaluation command |
| Latent ARI analyses, excitation analyses and other supplemental sensitivity analyses | Analysis-specific source sets | Not exposed by the example CLI; the available reference tables do not imply full supplemental reproduction |

The 137-person prediction task uses a 150-s reference interval and a separate 200–300-s target interval. Its normalization uses reference-interval means only, with reference and target segments filtered independently and no full-record alignment. It must not reuse the full-record transforms used for the 139-person classification task. The two excluded prediction participants lacked a complete prescribed target interval; they remain in the classification cohort.

## Preprocessed clinical inputs

The public loader accepts already preprocessed relative-signal windows. It validates shapes, participant/record identities and side availability, then applies the frozen surrogate-training scaler. It does not implement raw clinical beat-to-beat preprocessing, record-level cross-correlation alignment, mirror padding, detrending, filtering or resampling. Those preceding steps must follow the manuscript's Clinical cohort and preprocessing section when preparing a full external clinical dataset.

For each control recording, include both sides. For each stroke recording, supply only the symptom-ipsilateral side determined from the clinical metadata. The loader rejects a bilateral stroke input rather than choosing a side from reconstruction performance. The physiological correctness of the supplied side designation cannot be inferred from the waveform arrays alone.

Relative normalization differs by task:

- Surrogate generation: each raw 125-Hz ABP window's mean.
- Clinical classification: complete-record means, followed by the documented signal processing and frozen training-set standardization.
- Supplementary measured-CBFV prediction: reference-interval means, with target CBFV reserved for scoring.

## Evaluation and statistics

The primary AUC comparison uses Joint (step latent) GP6 and spectral TFA GPC9. The primary family also compares Joint (step latent) with Waveform CVAE and Step VAE + waveform decoder. Six secondary comparisons retain a separate Holm family. Gain-only, phase-only, band-specific and 15-feature fusion results are exploratory. Neural AUC averages held-out probabilities across seeds; prediction MSE averages individual-model errors.

The evaluation command reports both 95% and Bonferroni-adjusted 98.33% intervals for the three primary paired contrasts. Source bootstraps retain methods together, and clinical bootstraps resample participants within class using fixed held-out scores. Seed SD is reported separately from bootstrap uncertainty. Missing methods in a user-selected subset retain their planned family positions and are marked unavailable.

## Figures and reference numbers

No generated demonstration figure is embedded in the README. Running the demo or plotting script can create local inspection plots; those are not manuscript figures. Manuscript Tables 1 and 2 are provided as machine-readable reference tables. The archived values are identified by their full-study population and must not be attributed to the bundled two-person or 2.95% examples.
