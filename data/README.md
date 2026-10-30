# Data

`demo/surrogate_demo.npz` contains 11,610 ABP–CBFV pairs from 210 source records, or 2.94895% of the 393,700 original pairs. The deterministic sampling seed is 20261030. Within each original split, records were stratified by their window counts; approximately 3% of records in each stratum were sampled. All windows and all ten ARI variants of each selected record were retained.

| Split | Source records | Pairs |
|---|---:|---:|
| Train | 150 | 8,310 |
| Validation | 30 | 1,650 |
| Test | 30 | 1,650 |

`demo/jlu_demo.npz` contains all 12 curated analysis windows of one control and all 5 curated analysis windows of one patient. The release IDs are `JLU_control_001` and `JLU_patient_001`. ABP and CBFV are preprocessed relative perturbations, not raw mmHg or cm/s. Side labels and relative window starts are retained for correct aggregation; raw acquisition files and the private identity mapping are excluded.

Selection used the five frozen MS-aux models. For each participant, Pearson correlations on samples 11:100 were averaged across seeds, windows within each side, and then sides. The highest correlation participant in each group was selected; lower variance-normalized MSE breaks ties. The repeated-visit participant was excluded to keep each example to one record. The selected mean correlations are 0.76896 (control) and 0.90902 (patient). These are favorable illustrative examples, not a representative validation sample.

The `step` arrays are the archived FIR-derived inputs associated with the selected rows. They use the frozen train-only global standardization in `configs/models.json`; they must not be interpreted as physical CBFV. `src.preprocessing.load_dataset` can compute `step` when it is omitted from an external dataset. No clinical examples enter neural training.

Clinical classification uses complete-record means for relative normalization; surrogate generation uses each 125-Hz pressure window's mean. Both subsequently use the frozen surrogate-training global means and SDs. The clinical loader expects signals after the manuscript's raw-record alignment, detrending, filtering and resampling; it does not perform those preceding steps. Controls must include both sides per record, while stroke records must contain only the symptom-ipsilateral side. For repeated visits, retain one shared `person_ids` value and separate `record_ids` values.

Clinical reference results use the complete 139-participant cohort (68 controls, 71 stroke). The two-person examples demonstrate the data format and frozen inference; see [reproduction scope](../docs/reproduction_scope.md).

The original surrogate construction used recorded ABP to drive Tiecks ARI 0–9 simulations. Source ABP is from Kachuee et al., [Cuff-Less Blood Pressure Estimation, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/340/cuff), DOI 10.24432/C5B602 (manuscript reference 23). UCI provides the source `Part_*.mat` files and identifies their license as CC BY 4.0. The generated surrogate pairs are derived from those pressure recordings; the repository's MIT software license does not replace the source-data attribution.

The complete MATLAB generator is in `simulation/generate_surrogate.m`; it needs the source records supplied separately. The 3% figure describes included pairs, not a newly simulated dataset or a fraction of file bytes.

For complete real/clinical data, contact **[hx.yang2@siat.ac.cn](mailto:hx.yang2@siat.ac.cn)**. See [access details](../docs/data_access.md). Full datasets belong under an ignored location such as `data/private/`.
