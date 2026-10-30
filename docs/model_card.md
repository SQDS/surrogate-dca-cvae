# Model card

## Architecture and intended use

Research implementation for surrogate pressure–flow representation learning and frozen inference on measured ABP–CBFV windows. Four configurations share identically initialized corresponding modules within each seed:

| Configuration | Encoders / decoders | Inference route |
|---|---|---|
| M | Waveform encoder and ABP-conditioned waveform decoder | M |
| S | Step encoder and step decoder | S; reconstructed step is converted to FIR |
| S-Y | Step encoder, step decoder, waveform decoder | S-Y |
| MS | Both encoders and both decoders, with cross-reconstruction | MS-main and MS-aux |

Main hidden layers are 150, 150, 500; auxiliary hidden layers are 50, 50, 100. Hidden activations are Tanh, output layers are linear, and the latent size is 30. Input windows have 100 samples at 1 Hz. Step responses have 12 samples. Main input is `[CBFV, ABP]`; its decoder receives `[z, ABP]`.

The objective combines available direct waveform/step MSE terms, 0.5 times available cross-reconstruction MSE terms, and beta times the available KL terms. Beta reaches 0.001 over the first ten epochs. Training samples one noise vector per active branch and reuses that sample for its decoder paths. Inference uses the posterior mean.

## Data and inference

Train-only global normalization parameters are frozen in `configs/models.json`. Same-window reconstruction is conditioned on the observed window and is distinct from prediction. Cross-window inference encodes calibration windows, conditions the decoder on target ABP, and averages decoded predictions; target CBFV is used only for scoring. Waveform scores use indices 11:100 so FIR startup samples are excluded.

For clinical TFA, each observation window is encoded independently. Five overlapping 100-second windows from a 300-second sinusoidal probe are decoded at each frequency from 0.02 to 0.50 Hz. Their spectra are averaged before computing gain, phase and coherence. Window curves are then aggregated by side, visit and person. Phase averaging is circular. Conventional TFA averages measured-window spectra within a side. The Nyquist probe uses the archived RMS-matched cosine policy. Invalid coherence is retained as missing, with no additive denominator offset.

Clinical classification is participant-level leave-one-out. Repeated visits must share a person ID. Scaling and missing-feature imputation use the training fold only. Five-seed predictions are combined by averaging out-of-fold probabilities. Bootstrap resamples participants within class and does not refit the classifier. Paired DeLong comparisons use the planned Holm families in `configs/evaluation.json`.

## Scope

The release's two JLU examples were deliberately selected for favorable reconstruction. Their correlations and plots describe those examples only. They cannot reproduce clinical AUC or the full cohort confidence intervals. Full reference results are separately identified in `results/`.

The model was developed for research on the supplied signal convention and cohort. It is not a clinical diagnostic system. No superiority to conventional TFA, physiological ground-truth validation, or measured-CBFV prediction advantage is claimed by this package.

## Provenance

Algorithms and weights originate from the completed four-configuration, five-seed `cvae_step_ablation_v2_20261013` experiment. `code_provenance.csv` records the source module hashes. Public modules remove workspace-dependent paths and cohort-size assertions while retaining the numerical model, loss, FIR, spectra and inference implementations. See `validation.json` for the executed release checks.
