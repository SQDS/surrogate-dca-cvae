# Full-study reference results

These tables describe the completed full-data manuscript experiments. They are not calculated from the small datasets in `data/demo/`.

| File | Manuscript connection | Scope |
|---|---|---|
| `table1_clinical_discrimination.csv` | Table 1 | 139-person stroke/control classification; AUC and participant-bootstrap CI |
| `table2_matched_ablations.csv` | Table 2 | Five neural routes; clinical AUC and independent-window surrogate MSE with seed SD |
| `reference_summary.csv` | Main and supporting results | Clinical AUC, same-window surrogate MSE, independent-window surrogate MSE; task identified by `metric` |
| `paired_auc_Holm.csv` | Paired comparisons and supporting analyses | Three primary and six secondary AUC contrasts, separate Holm families |
| `cross_paired_intervals.csv` | Independent-window prediction | Paired source-bootstrap MSE differences; 354 sources |
| `clinical_prediction_summary.csv` | Supplementary measured-CBFV prediction | 137-person task; distinct preprocessing from clinical classification |
| `clinical_prediction_paired_intervals.csv` | Supplementary measured-CBFV prediction | Participant-bootstrap MSE differences for that 137-person task |

`table1_clinical_discrimination.csv` and `table2_matched_ablations.csv` retain the manuscript's displayed precision. `reference_summary.csv` and the paired tables retain the archived numerical precision. In Table 2, prediction MSE and its seed SD are expressed in units of 10⁻⁶ relative CBFV squared; the clinical AUC interval describes participant-bootstrap uncertainty.

The joint step-latent route's AUC is 0.9515, and the spectral TFA GPC9 AUC is 0.9145. Their paired difference is 0.0371 with a 95% CI crossing zero and Holm-adjusted p=0.3678. This matches the manuscript's conclusion that superiority to TFA was not established. The waveform-prediction tasks also did not establish an error advantage over TFA.
