# onion-bench-stress: multi-dataset stress benchmark (issue #244)

The benchmark's design is in the datasets dossier (§A.3). This file holds the protocol, written and committed before the benchmark runs (only the learning-rate selection ran first, on validation data). The results follow once they have run.

## Protocol

### Two tracks, never in one table

- **E4 track.** Four Empatica E4 studies read at the native E4 rates with the same 30 harmonized features: `acc_x`, `acc_y`, `acc_z`, `bvp`, `eda`, `temp` × `mean`, `std`, `min`, `max`, `median`, over 60 s windows with 50% overlap.
  - `wesad_e4` (WESAD's wrist), `physionet_e4`, `nurse_stress` and `stress_predict`.
  - One edge per subject and one fog per study (`topologies/stress_e4_studies.yaml`).
- **Lab track.** SWELL-KW (computer interaction, per minute) and WESAD (`datasets/wesad.yaml`, wrist at the label rate), the pair of the existing baselines (`techniques_*`). SWELL has no E4, so its features differ and `adapters: shared` is impossible there.
  - Topology `four_fogs_swell_wesad_lossless`, with the roles of the existing baselines.

### The binary stress label of each dataset

All E4 datasets use task `stress_binary`, so one head serves them all. The labels are not the same construct:

| Dataset | 0 | 1 | Source |
|---|---|---|---|
| `wesad_e4` | baseline | TSST (stress) | protocol condition; amusement and the rest are dropped |
| `physionet_e4` | baseline and rests | stress tasks | protocol tags (STRESS sessions); exercise sessions are dropped in the binary label |
| `nurse_stress` | outside every survey event of a surveyed session | inside an event the nurse rated ≥ 1 | the nurses' survey on events their detector flagged; unrated and level-0 events are dropped |
| `stress_predict` | baseline, rests, questionnaire | Stroop, interview, hyperventilation | `Time_logs.xlsx` phases |
| `swell` (lab track) | N (neutral) | T, I, R (time pressure, interruptions, combined) | condition per minute |
| `wesad` (lab track) | baseline | TSST | protocol condition |

Nurse Stress is field data labelled by self-report, the others are laboratory protocols. Hyperventilation (Stress-Predict) is a physiological stressor, not a psychological one.

### Subjects and preprocessing

- **Roles** (`roles.seed: 0`): 25% of each study's subjects for test, 10% as validation (zone evaluators), and the last 20% of each class of every training subject as its `local_val`. The test subjects are drawn per dataset with their own stream, so they are the same in every scenario and every seed of the track:

| Dataset | Test subjects |
|---|---|
| `wesad_e4` | S8, S13, S16, S17 |
| `physionet_e4` | S04, S09, S10, S11, S16, S18, f10, f15, f17 |
| `nurse_stress` | 6D, 8B, 83, DF |
| `stress_predict` | S04, S08, S15, S17, S24, S28, S33, S35 |
| `swell` (lab track) | 1, 2, 12, 18, 19 |
| `wesad` (lab track) | S6, S7, S10 |

- **Scaling (E4 track):** `scaler: local`. Every holder (edge or evaluator) standardises with the statistics of its own rows, without labels. A study held out of training has no training rows to fit a scaler on, so the whole track uses the same per-subject scaling. `drop_constant: false`, so the shared adapter sees the same 30 inputs in every study.
- **Scaling (lab track):** `scaler: global` per dataset on the training bag, as in the existing baselines.

### Model and training

- `modular_mlp`: adapter width 64, trunk [64, 32], dropout 0.2, one head per task.
- SGD, batch 32, 20 rounds. Local epochs: 5 (E4 track), 10 (lab track, as the existing baselines).
- **Seeds 0–4.** The seed changes the initialisation and the batches. The test subjects do not change.

### Learning rates, chosen before the benchmark on validation only

`benchmarks/stress_select.yaml` (E4) and `benchmarks/stress_lab_select.yaml` (lab) run every technique at lr 0.03, 0.1 and 0.3 on seed 0. The rule:
- FedAvg, SCAFFOLD and FedDyn: the global model's macro-F1 on the validation subjects at round 20.
- Ditto, FedRep and FedBABU: the edges' macro-F1 on their `local_val` rows at round 20, for the model the technique delivers (Ditto personal, FedRep local, FedBABU fine-tuned).
- Ties within 0.005 go to the smaller lr. A diverged run (lost rounds, no score) is not chosen.

### Comparisons

| # | Comparison | Where |
|---|---|---|
| 1 | Majority class; logistic regression per study and pooled over the four studies | `scripts/stress_bench.py baselines` |
| 2 | Local-only: a logistic regression per training edge, scored on its study's test subjects; FedAvg within each study alone | `baselines` (`local_only`), `benchmarks/stress_per_study.yaml` |
| 3 | FedAvg flat (one aggregator) vs hierarchical (one fog per study) | `benchmarks/stress.yaml` |
| 4 | `adapters: per_dataset` vs `shared` | `benchmarks/stress.yaml` |
| 5 | Ditto, FedRep, FedBABU; SCAFFOLD, FedDyn | `benchmarks/stress.yaml` |
| 6 | Leave-one-dataset-out, above all lab → Nurse | `benchmarks/stress_lodo.yaml`; logistic regression `lr_lodo` |

- The reference scenario is FedAvg, hierarchical, `per_dataset` adapters.
- The leave-one-dataset-out reference is FedAvg, hierarchical, `shared` adapter, with every study training. A study that never trains has no adapter of its own, so only the shared adapter applies.
- **Expected before running:** with full participation, lossless links and sample-weighted FedAvg at both levels, the hierarchical average equals the flat one. Flat vs hierarchical should then differ only by floating-point order, and in the bytes on the cloud link.

### Metrics

- **Primary:** the macro-F1 over every test window of a study, at round 20, for the final global model (`model.npz`). It is scored offline by `scripts/stress_bench.py score`, with the same function as the classical references.
  - This is not the `global/<dataset>` value of `summary.json`. That one is the evaluators' macro-F1 averaged by samples, not the F1 of the pooled windows.
- **Secondary:** balanced accuracy, AUROC, and per-subject F1. Per-subject F1 is each test subject's macro-F1 over the classes present in its windows; tables give its median and 10th percentile over the study's test subjects. Edge scores on `local_val` at round 20 are given for the personalisation techniques; bytes and wall time per run.
- **Statistics:** mean ± 95% Student-t interval over the 5 seeds, and the differences to the reference paired by seed (same test subjects), with their own interval.
- **Personalisation:** FedRep keeps the head on the edges, so the global model holds an untrained head, and its global column does not measure FedRep. Ditto's global model is FedAvg's. Their edge columns are the technique.
