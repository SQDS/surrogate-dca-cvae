# Release validation

The initial packaged release was checked on 2026-10-30 using Python 3.10.19 and the dependencies listed in `requirements.txt`. Exact versions and the initial machine-readable checks are in [validation.json](validation.json). The later manuscript-alignment revision is covered separately by [manuscript_alignment_validation.json](manuscript_alignment_validation.json).

- All requested architecture entries exist. Runtime Python imports resolve within this repository and the installed dependencies.
- The surrogate example contains 11,610 exact source pairs from 210 complete records. All ARI variants, windows and original split assignments are retained; no source record crosses splits.
- JLU examples contain exactly one control and one patient. Their 12 and 5 windows match the corresponding source analysis arrays. The original participant IDs, acquisition paths and other identity fields are excluded from the public files.
- All 20 portable checkpoints pass SHA256 verification. Every model parameter equals the corresponding source parameter exactly. For each checkpoint, fixed-noise loss values, gradients, same-window inference and cross-window inference match the original code exactly.
- The FIR implementation matches the source calculation exactly. The five-seed JLU TFA curves agree with their archived counterparts within numerical tolerances; the maximum absolute difference across gain/phase/coherence entries is recorded in the JSON report. Evaluation on CPU can differ slightly from archived GPU results.
- Generalized participant-level LOPO yields exactly the same probabilities as the archived 139-participant feature-table implementation. DeLong, Holm and bootstrap comparisons also match. The two-person demo is rejected as an AUC cohort.
- The five-seed demo, all 20-model surrogate evaluations, generated plots and two-epoch training checks for all four configurations completed successfully on CPU.
- The archive was extracted into a temporary directory and its file checksums were verified. The demo then ran successfully with a working directory outside the extracted repository and without a custom `PYTHONPATH`. The clinical AUC command correctly rejected the two-person example. See [portability_check.json](portability_check.json).

The MATLAB generator was inspected and packaged with its internal helper functions, but was not executed because MATLAB/Octave is unavailable in the validation environment. Full 400-epoch training was not repeated. Weight download verification was exercised on the bundled files; remote downloading was not exercised because no public release URL is configured.

`checksums.sha256` lists the final shipped files. The distribution archive excludes local runs, optimizer checkpoints, caches, full clinical data and private identity mappings.

## Manuscript-alignment revision

The author-supplied DOCX was read directly, and the displayed values in both manuscript tables were compared with the archived analysis tables. All AUCs, confidence bounds, prediction MSEs and seed SDs matched. Model parameter counts, selected epoch ranges and training settings also matched. The neural checkpoints and demonstration arrays were not changed.

Generated README preview figures were removed. The clinical evaluation entry point now includes the directly fitted FIR comparator and the manuscript's exploratory feature designs. Primary paired AUC and prediction-MSE reports include 98.33% intervals, and seed SD is separated from bootstrap uncertainty. Input checks enforce bilateral controls and one preselected ipsilateral side for each stroke recording.

Validation of the changes comprised all 44 clinical designs on the archived 139-person feature table, including exact agreement in AUC/CI/Holm results; direct calculation of all 139 participants' FIR features; a six-person real-window CLI execution check; and all 20 models on the bundled surrogate test subset. The full clinical neural feature extraction and 400-epoch training were not rerun. The six-person check is an execution test, not a new scientific result. Private participant data used for validation remain outside the repository.
