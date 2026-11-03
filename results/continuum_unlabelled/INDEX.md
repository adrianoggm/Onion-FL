# Unlabelled rows on the SWELL and WESAD streams: the selection on validation (new framework, real data)

`experiments/stream_unlabelled_select.yaml` was run on 2026-11-03 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #160 (continuum C5).

```bash
onion_fl run experiments/stream_unlabelled_select.yaml --workers 3   # run twice: the second is the determinism check
python scripts/score_final_models.py runs/<run_id> ... --subjects val > validation.csv
```

**This is a null result.** The plan was to choose each objective's settings on the validation subjects, then compare the chosen ones against `none` on test (`experiments/stream_unlabelled.yaml`). The selection cannot choose: in every configuration and seed, the final global model is a constant predictor, as in C3 and C4, so the validation macro-F1 ties at the majority-class value. The validation loss does not separate them either: its differences fall within the spread across seeds. The test comparison was therefore not run, and `stream_unlabelled.yaml` was not written. The user made this decision after seeing the selection.

What the runs do show is how the objectives behave on the real streams: the pseudo-label audit, and the prequential score.

## Setup

- **Stream and training.** As in [continuum_stream/](../continuum_stream/INDEX.md), with a tenth of the rows labelled:
  - SWELL + WESAD replayed in time order after a 20-minute bootstrap;
  - a round every 10 minutes of data, 18 in all, until the last label arrives;
  - `labels: {fraction: 0.1, delay: 10m}`: each edge labels a seeded tenth of its rows, 10 minutes late. The other nine tenths wait in its unlabelled pool;
  - FedAvg with SGD at lr 0.1 and 10 local epochs, on the segregated lossless topology `four_fogs_swell_wesad_lossless`, quorum 1.0.
- **Nine configurations:**
  - `none`: the unlabelled rows are not used (C3's behaviour);
  - `pseudo_label`, `threshold` 0.8, 0.9 or 0.95 × `weight` 0.5 or 1.0: each row whose max probability under the received model reaches the threshold adds weight·CE with that class;
  - `consistency`, `noise` 0.05 or 0.1 (weight 0.5): weight·KL(p(x) ‖ p(x + ε)) on the scaled features.
- **Scores.**
  - **Validation** (`validation.csv`): the final model (`model.npz`), scored offline by `scripts/score_final_models.py` on every row of the validation subjects, pooled per dataset. The test subjects were never evaluated.
  - **Prequential** (`run_metrics.csv`): every arrival scored by the model its edge was serving, pooled from the edges' confusion counts over the run. `prequential_labelled` keeps only the rows whose label arrived.
  - **Pseudo-labels** (`pseudo_labels.csv`): from `diagnostic.unlabelled`, summed over the trainings that worked. `pseudo_label_accuracy` compares the accepted pseudo-labels with the hidden labels, which only the simulation knows.
- **Seeds** 0–2.

| Identity | Value |
|---|---|
| `topology_id` | `ff55fe4d2ce1a807f764bcc5205a8d2d4d43ab3d533163a1d13e173a9a6725fa` |
| `data_id` | `9fd573754157d68b111f037ecd891bdb62abcbff1d5e5e818da789dde470257c` (the same as C4's) |
| code | commit `07fb2c8`, clean tree |
| `config_id`s | one per configuration, in `run_metrics.csv` |

## Results (mean ± 95% CI over 3 seeds)

**Validation, final model** (`validation.csv`). Every configuration and seed scores the same, because every final model predicts stress for every validation row (confusion matrices `[[0, 85], [0, 171]]` on SWELL and `[[0, 75], [0, 40]]` on WESAD):

| Configuration | Macro-F1 SWELL | Macro-F1 WESAD | Loss SWELL | Loss WESAD |
|---|---|---|---|---|
| none | 0.400 ± 0.000 | 0.258 ± 0.000 | 3.927 ± 0.673 | 4.097 ± 0.130 |
| pseudo_label 0.8 / 0.5 | 0.400 ± 0.000 | 0.258 ± 0.000 | 3.625 ± 0.334 | 4.171 ± 0.086 |
| pseudo_label 0.8 / 1.0 | 0.400 ± 0.000 | 0.258 ± 0.000 | 6.656 ± 5.068 | 4.954 ± 0.965 |
| pseudo_label 0.9 / 0.5 | 0.400 ± 0.000 | 0.258 ± 0.000 | 3.922 ± 0.972 | 4.211 ± 0.122 |
| pseudo_label 0.9 / 1.0 | 0.400 ± 0.000 | 0.258 ± 0.000 | 3.567 ± 0.397 | 4.266 ± 0.226 |
| pseudo_label 0.95 / 0.5 | 0.400 ± 0.000 | 0.258 ± 0.000 | 4.221 ± 1.027 | 4.309 ± 0.267 |
| pseudo_label 0.95 / 1.0 | 0.400 ± 0.000 | 0.258 ± 0.000 | 4.449 ± 0.972 | 4.533 ± 0.303 |
| consistency 0.05 | 0.400 ± 0.000 | 0.258 ± 0.000 | 3.464 ± 0.763 | 3.805 ± 0.130 |
| consistency 0.1 | 0.400 ± 0.000 | 0.258 ± 0.000 | 3.461 ± 0.763 | 3.804 ± 0.132 |

**Prequential macro-F1** (`run_metrics.csv`; WESAD is 0.000 ± 0.000 everywhere):

| Configuration | Prequential SWELL | Labelled only, SWELL |
|---|---|---|
| none | 0.800 ± 0.019 | 0.789 ± 0.057 |
| pseudo_label 0.8 / 0.5 | 0.701 ± 0.059 | 0.678 ± 0.108 |
| pseudo_label 0.8 / 1.0 | 0.657 ± 0.025 | 0.630 ± 0.082 |
| pseudo_label 0.9 / 0.5 | 0.744 ± 0.041 | 0.723 ± 0.083 |
| pseudo_label 0.9 / 1.0 | 0.699 ± 0.052 | 0.678 ± 0.103 |
| pseudo_label 0.95 / 0.5 | 0.770 ± 0.014 | 0.750 ± 0.067 |
| pseudo_label 0.95 / 1.0 | 0.749 ± 0.037 | 0.727 ± 0.075 |
| consistency 0.05 | 0.797 ± 0.016 | 0.786 ± 0.056 |
| consistency 0.1 | 0.797 ± 0.016 | 0.786 ± 0.056 |

**Pseudo-labels** (`pseudo_labels.csv`, mean over seeds of each run's totals per dataset). Each run sees about 2111 SWELL and 531 WESAD pool rows:

| Configuration | Accepted SWELL (rate) | Accuracy SWELL (min–max over seeds) | Accepted WESAD (rate) | Accuracy WESAD |
|---|---|---|---|---|
| 0.8 / 0.5 | 1429 (0.68) | 0.930 (0.875–0.985) | 182 (0.34) | 0.000 |
| 0.8 / 1.0 | 1400 (0.66) | 0.848 (0.813–0.891) | 182 (0.34) | 0.000 |
| 0.9 / 0.5 | 1259 (0.60) | 0.988 (0.978–0.998) | 169 (0.32) | 0.000 |
| 0.9 / 1.0 | 1193 (0.57) | 0.959 (0.915–0.994) | 177 (0.33) | 0.000 |
| 0.95 / 0.5 | 1080 (0.51) | 0.994 (0.986–1.000) | 75 (0.14) | 0.000 |
| 0.95 / 1.0 | 1077 (0.51) | 0.993 (0.987–1.000) | 85 (0.16) | 0.000 |

The mean confidence over the rows seen is about 0.85 on SWELL and 0.67 on WESAD in every configuration.

**Checks:**
- **No failed training.** Every run has 190–196 trainings that worked, none that failed, and no failed round.
- **Determinism** (`determinism.csv`): each configuration and seed, run twice, gives the same final model and the same events, 27 of 27. The reruns are a first pass at commit `76497f8`, whose tree was dirty only with the then-untracked `scripts/score_final_models.py`; nothing that runs differed.
- **The objectives act.** Every objective changes the final model against `none` with the same seed (largest weight difference between 0.07 and 1.0), so the tie is the setup's, not a hook that does nothing.
- `verify_run` passed on the 27 original run folders.

## Reading

- **No setting can be chosen on validation in this setup.** The training settings (lr 0.1, quorum 1.0) were set for FedAvg on whole datasets, and every stream run since C3 ends in a constant global model. With a tenth of the labels, so does every C5 configuration. Choosing between them needs a setup whose final model is not constant: learning rates chosen on the validation stream, which is C8's.
- **On SWELL the pseudo-labels are accurate, and still hurt the prequential score.**
  - The accepted pseudo-labels are right 85–99% of the time, more so at higher thresholds and lower weights.
  - Yet pseudo-labelling lowers the prequential score from 0.800 to 0.657–0.770, more as the threshold falls and the weight rises.
  - In these runs, the reading that fits is C4's: SWELL's prequential score mostly measures persistence, and training on the model's own confident predictions holds the previous condition's class for longer after a switch. The accuracy is high because most rows are not near a switch. These runs do not test this directly; the per-round prequential series would.
- **On WESAD every accepted pseudo-label is wrong.** After the bootstrap every WESAD window is stress, and no served model predicts it (C4). So the received model is confident in the wrong class: pseudo-labelling reinforces it, and the audit shows it (accuracy 0.000 in every run). This is the confirmation bias the audit (`generated_by_version`) exists to expose.
- **Consistency changes almost nothing.** Its prequential score is within 0.003 of `none`'s, and both noise levels give nearly the same result.
- **What C8 needs from this:**
  - training settings chosen on the validation stream, so that the selection has a non-constant model to choose with;
  - a threshold that adapts, or a check on the class balance of the accepted pseudo-labels, since a confident model on a shifted stream accepts only the wrong class;
  - more than three seeds.

## Files

- `<run_id>/`: `run.json`, `summary.json` and `model.npz` of the 27 cited runs. The 27 determinism reruns stay in `runs/`.
- `validation.csv`: per run and dataset, the final model's loss, accuracy, macro-F1 and confusion counts on the validation subjects.
- `run_metrics.csv`: per run, the trainings, failed trainings and failed cloud rounds, and the prequential and labelled-only macro-F1 per dataset with their sample counts.
- `pseudo_labels.csv`: per run and dataset, the pool rows seen, accepted, the acceptance rate, the mean confidence and the accuracy of the accepted pseudo-labels.
- `determinism.csv`: each cited run against its rerun.

`events.jsonl` and `bundle/` are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
