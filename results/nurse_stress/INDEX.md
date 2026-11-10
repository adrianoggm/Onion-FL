# Nurse Stress: dataset card and centralised baselines (real data)

`datasets/nurse_stress.yaml` read on 2026-11-10 from the CC0 files of Hosseini et al. (2022), downloaded from [Zenodo record 5514277](https://zenodo.org/records/5514277), the same two files as [Dryad](https://doi.org/10.5061/dryad.5hqbzkh6f): `Stress_dataset.zip` (MD5 `8829a9c0175e9afa614f95f12f210cea`) and `SurveyResults.xlsx` (MD5 `790b12dce346992b4a83f57a26c40bb8`). Issue #241.

```bash
python -I scripts/nurse_stress_extract.py data/NurseStress/Stress_dataset.zip data/NurseStress/Stress_dataset
onion_fl data inspect nurse_stress > inspect.json
onion_fl baseline experiments/nurse_stress_reference.yaml --models lr rf --seeds 0 1 2 3 4 > baselines_seeds.json
onion_fl baseline experiments/nurse_stress_reference.yaml --models lr rf --cv 5 > baselines_cv5.json
```

Only centralised baselines have run. The federated part of `experiments/nurse_stress_reference.yaml` has not.

## Setup

- **Labels.** They were declared before any model was trained (the descriptor's header): a window is stress inside a survey event rated 1 or 2, and no stress outside every event of an E4 session that holds an event. Windows in unrated (`na`) or level-0 events, and sessions without an event, are dropped. The survey's local times are read in America/Chicago; the descriptor's header gives the evidence.
- **Data.** 15 nurses (one subject each), 63,913 windows of 60 s with half overlap (532 hours), 15.0% stress. The 30 features are the harmonized E4 statistics of ACC, BVP, EDA and TEMP.
- **Dataset card.** `inspect.json` has no missing values and no warning. The feature most correlated with the label is `eda_min`, with |r| = 0.29.
- **Stress share per nurse.** It ranges from 3.9% (94) to 59.4% (6D, 155 windows).
- **Roles.** Each seed draws its own 4 test nurses (a quarter of 15). The other 11 train. Imputation, scaling and the removal of constant features are fitted on the training nurses.
- **Models.** Logistic regression and random forest with their default parameters (`onion_fl.baselines`), set before the run and not tuned.

| Identity | Value |
|---|---|
| `topology_id` | `9d37394cf720328e16e6695e171a5d8eb583e9792e531690afd60303bc7df74f` |
| `config_id` | `93db2e3c08065f6738fa63512c2a164995a9352e02ecb887219880c65fc7bcaa` |
| `data_id` | `ea32ad8dbd299838d8e51a43322c86daa9e6bb5fe0e69eac542185131f5d015d` |
| code | commit `36e3a25`, clean tree |

There is no `run_id` or `run_hash`: `onion_fl baseline` writes no run folder. The identity of these numbers is the config, the data and the commit above.

## Results on held-out nurses

Mean ± 95% Student-t half-width over seeds 0–4, each with its own 4 test nurses (`baselines_seeds.json`, which lists them):

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|
| Logistic regression | 0.851 ± 0.027 | 0.538 ± 0.065 | 0.528 ± 0.091 |
| Random forest | 0.834 ± 0.036 | 0.540 ± 0.057 | 0.533 ± 0.081 |
| Always "no stress" | 0.846 ± 0.041 | 0.500 | 0.458 ± 0.012 |

The "always no stress" row is computed from the test nurses' labels of each seed. Subject-level 5-fold CV (`baselines_cv5.json`, mean ± std over folds) gives balanced accuracy 0.530 ± 0.020 for LR and 0.553 ± 0.037 for RF, with macro-F1 of 0.521 ± 0.030 and 0.555 ± 0.053.

- **The models barely beat the majority class.** Their accuracy does not beat it, and their balanced accuracy is 0.54, with intervals whose lower ends are 0.47 and 0.48. Only seed 4 (test nurses 5C, 6B, 8B, E4) lifts LR to 0.63.
- **The labels limit what can be learnt.** The stress events are the epochs that the authors' detector (a Random Forest on EDA, HR and TEMP, trained on AffectiveROAD) flagged, which the nurses then rated. "No stress" is the time the detector did not flag, and the paper warns that unlabelled time "does not necessarily imply a lack of stress". Labels from the field, a 15% minority class and 15 nurses make this a hard, noisy target. It is not a leak: no feature has |r| > 0.3 with the label.
- **Comparison with WESAD.** These numbers are not comparable with WESAD's laboratory labels. They are the field reference for later comparisons (laboratory → hospital), not a benchmark to beat.

## Not run yet

- The `high` label strategy (level 2 only), declared as a sensitivity check.
- The federated runs of `experiments/nurse_stress_reference.yaml`.

## Files

- `inspect.json`: the `data inspect` card.
- `baselines_seeds.json`: each seed's test nurses and scores, and the summary with intervals.
- `baselines_cv5.json`: subject-level 5-fold CV per model.
