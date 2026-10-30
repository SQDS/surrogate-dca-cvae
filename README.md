# Surrogate dCA CVAE

Code, 20 frozen model checkpoints, and small example datasets for **Toward Digital Twin–Enabled Assessment of Dynamic Cerebral Autoregulation: Learning From Surrogate Pressure–Flow Data**.

The study uses recorded arterial blood pressure (ABP) to drive the Tiecks simulator and generate surrogate cerebral blood flow velocity (CBFV). A waveform conditional variational autoencoder is coupled to a step-response variational autoencoder. After surrogate-only training and checkpoint selection, frozen models encode measured clinical recordings and generate responses to prescribed sinusoidal ABP probes. Clinical evaluation uses participant-level stroke/control classification. Separate experiments assess CBFV reconstruction and independent-window prediction.

## Models and study protocol

| Manuscript configuration | Code name | Parameters | Frozen inference route |
|---|---|---:|---|
| Waveform CVAE | `M` | 336,760 | Waveform latent |
| Step VAE | `S` | 25,672 | Reconstructed step converted to FIR coefficients |
| Step VAE + waveform decoder | `S-Y` | 204,072 | Step latent |
| Joint CVAE–Step VAE | `MS` | 362,432 | `MS-main`: waveform latent; `MS-aux`: step latent |

All four configurations use seeds 42–46, giving 20 separately trained models. The two joint-model routes share the same MS weights. The full surrogate dataset contains 39,370 pressure windows and 393,700 ABP–CBFV pairs, split 70%/15%/15% by source record. This repository's clinical results use the manuscript's **139-participant cohort: 68 controls and 71 patients with stroke**, with all windows, sides and visits of a participant held out together. Neural weights remain frozen; the downstream classifier uses clinical labels within each training fold.

## Included data

| Dataset | Included | Purpose |
|---|---:|---|
| Surrogate pairs | **11,610 / 393,700 = 2.95%**, 210 source records | Training and evaluation examples; original train/validation/test membership retained |
| JLU controls | **1 participant**, 12 windows | Curated reconstruction and TFA example |
| JLU patients | **1 participant**, 5 windows | Curated reconstruction and TFA example |
| Model weights | **4 configurations × 5 seeds = 20 checkpoints** | Frozen inference without training or downloading |

JLU examples were selected for favorable same-window reconstruction using the five frozen MS-aux models. They use new example IDs and contain no original file paths or participant identifiers. They are illustrative examples; two people cannot estimate clinical AUC. See [selection details](data/demo/selection_manifest.json).

**For the full real/clinical dataset and access arrangements, contact [hx.yang2@siat.ac.cn](mailto:hx.yang2@siat.ac.cn).** See [data access](docs/data_access.md).

## Install and run

The pinned environment was checked with Python 3.10.19; the default demo runs on CPU. The manuscript's full training runs used PyTorch 2.7.1 with CUDA 11.8 on Tesla V100 GPUs.

```bash
git clone https://github.com/SQDS/surrogate-dca-cvae.git
cd surrogate-dca-cvae
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_weights.py
python scripts/run_demo.py
```

The demo runs the two clinical examples through frozen MS models and TFA, plus a surrogate independent-window prediction example. It writes numerical outputs and inspection plots under `outputs/demo/`. Those plots are generated examples, not manuscript figures.

```bash
# All four configurations and five seeds, on the bundled surrogate test subset.
python scripts/evaluate.py --out outputs/evaluation
python scripts/plot_results.py --input outputs/evaluation

# Short execution check for all four training configurations.
python scripts/train_models.py --smoke --seeds 42 --out outputs/training_smoke

# Training on the included 2.95% subset (new models, not paper-model reproduction).
python scripts/train_models.py --device cuda:0 --out outputs/subset_training

# With complete datasets obtained separately, using data/schema.json.
python scripts/train_models.py --data data/private/surrogate_full.npz --device cuda:0 --out outputs/full_training
python scripts/evaluate.py --task classification --data data/private/jlu_full.npz --out outputs/clinical
```

`train_models.py` defaults to 400 epochs, four configurations, and seeds 42–46. Checkpoint selection uses validation cross-window relative MSE on samples 11:100; the earlier saved epoch wins exact ties. There is no early stopping in this release's four-configuration protocol. Training on the example subset cannot reproduce results from the full dataset. The supplied weights are the original selected model tensors, repackaged without optimizer history.

The classification command includes spectral TFA, directly fitted FIR and the selected neural configurations. GP6 supports the main comparisons; gain-only, phase-only, band-specific and feature-fusion analyses are marked exploratory. See [reproduction scope](docs/reproduction_scope.md) for the commands covering the 139-participant clinical analysis and the surrogate experiments.

## Layout

```text
configs/      model, preprocessing, evaluation and statistical settings
src/          models, losses, preprocessing, FIR, TFA, reconstruction,
              participant-level classification and statistical functions
simulation/   MATLAB Tiecks data generator
scripts/      demo, weight verification/download, training, evaluation, plots
weights/      20 portable checkpoints and their SHA256 manifest
data/demo/    surrogate subset, two JLU examples and selection metadata
results/      aggregate reference results from the full archived experiment
docs/         model card, data access, provenance and validation notes
```

The MATLAB generator accepts explicit input/output directories and requires the original `Part_*.mat` ABP source files and MATLAB Signal Processing Toolbox. From MATLAB, add `simulation` to the path and call:

```matlab
generate_surrogate(100, '/path/to/source_abp', '/path/to/generated_output')
```

## Reference results and interpretation

The following clinical results use all **139 participants** and are from manuscript Tables 1 and 2. GP6 contains gain and phase in three frequency bands; GPC9 additionally contains coherence.

| Method | Features | AUC | 95% CI |
|---|---|---:|---|
| Spectral TFA | GPC9 | 0.9145 | [0.8596, 0.9600] |
| Spectral TFA | GP6 | 0.8863 | [0.8250, 0.9393] |
| Window-wise FIR | GP6 | 0.9316 | [0.8807, 0.9712] |
| Waveform CVAE | GP6 | 0.6553 | [0.5638, 0.7419] |
| Step VAE | GP6 | 0.8913 | [0.8335, 0.9420] |
| Step VAE + waveform decoder | GP6 | 0.9451 | [0.9024, 0.9785] |
| Joint (waveform latent) | GP6 | 0.7564 | [0.6703, 0.8364] |
| Joint (step latent) | GP6 | **0.9515** | [0.9111, 0.9832] |

Joint (step latent) exceeded Waveform CVAE, while superiority to conventional TFA or Step VAE + waveform decoder was not established. Against TFA GPC9, the paired AUC difference was 0.0371 (95% CI, −0.0193 to 0.0978; Holm-adjusted p = 0.3678).

The surrogate waveform experiments use the following test sets:

| Experiment | Full evaluation set | Joint (step latent) MSE | Spectral TFA MSE |
|---|---|---:|---:|
| Surrogate same-window reconstruction | 59,080 pairs from 1,067 test sources | 22.009 × 10⁻⁶ | Not the independent-window TFA comparison |
| Surrogate independent-window prediction | 3,540 ARI-specific targets from 354 test sources | 9.810 × 10⁻⁶ | 7.687 × 10⁻⁶ |

Surrogate prediction MSE averages the five models' errors, with equal source-record weighting. The surrogate prediction experiment found no error advantage over TFA. Clinical AUC and surrogate waveform MSE assess different outcomes.

Full reference tables are in [results/](results/README.md), including machine-readable versions of manuscript Tables 1 and 2. These are archived full-experiment results; the small included datasets are for running examples.

## Data convention

Clinical examples are already preprocessed 100-point ABP–CBFV windows at 1 Hz. Clinical relative normalization uses the complete-record mean; surrogate simulation uses each raw pressure window's mean. The frozen training-set means and SDs in `configs/models.json` provide the subsequent global standardization.

The auxiliary input is the cumulative response of a 12-coefficient ridge FIR model fitted in standardized coordinates. TFA gain is reported as relative CBFV/relative ABP (%/%), and phase in degrees. Controls contribute bilateral averages; stroke recordings use only the symptom-ipsilateral side. See [data schema](data/schema.json) and [model card](docs/model_card.md).

## License and citation

Code is released under the [MIT License](LICENSE), selected by the author for this package. See [CITATION.cff](CITATION.cff) for the software citation and repository URL. The manuscript title is recorded, but no unverified DOI or author list has been inserted. Contact [hx.yang2@siat.ac.cn](mailto:hx.yang2@siat.ac.cn) for the current manuscript citation and data access.

## 中文快速说明

本仓库以论文的 **139人临床队列（68名正常人、71名卒中患者）** 及模拟实验为依据。包含模拟数据 **2.95%**、JLU 正常人和卒中患者**各一人**、四配置五种子的 **20 个正式权重**。临床参考结果均采用139人队列；两人示例用于运行演示。完整真实数据请联系 **hx.yang2@siat.ac.cn**。
