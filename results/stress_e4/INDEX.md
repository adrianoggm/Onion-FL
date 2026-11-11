# onion-bench-stress: multi-dataset stress benchmark (issue #244)

The benchmark's design is in the datasets dossier (§A.3). This file holds the protocol, written and committed before the benchmark ran (only the learning-rate selection ran first, on validation data), then what ran and its results.

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

## What ran

All runs are simulated (`SimRuntime`), lossless, and finished with no lost round (`quorum_failed` 0), except the selection runs of SCAFFOLD at higher lr noted below. Every run of the benchmark ran on a clean tree; the framework code (`src/`) is the same at every cited commit (`8bf2343` to `e965216` only add benchmark files, scripts and docs).

| Experiment | Runs | What |
|---|---|---|
| `benchmarks/stress_select.yaml` | 18 | E4 lr selection, seed 0 |
| `benchmarks/stress.yaml` | 45 | 9 scenarios × 5 seeds |
| `benchmarks/stress_lodo.yaml` | 20 | 4 held-out studies × 5 seeds |
| `benchmarks/stress_per_study.yaml` | 20 | 4 studies alone × 5 seeds |
| `benchmarks/stress_lab_select.yaml` | 18 | lab lr selection, seed 0 |
| `benchmarks/stress_lab.yaml` | 35 | 7 scenarios × 5 seeds |
| `scripts/stress_bench.py baselines` | — | majority, logistic regression (per study, pooled, leave-one-out), local-only |

- **Reruns.** 15 E4 selection runs and 6 runs of `stress.yaml` first ran while the tree held untracked benchmark files (`dirty: true`). They were rerun on a clean tree. All 21 reruns gave bit-identical `model.npz`. Only the clean runs are committed.
- **Cost.** About 1.5 h of wall time (13:25–14:42 UTC, reruns included) on one CPU-only machine with 6 workers. One E4 run takes 2.5–6 min (Ditto is the slowest) and one lab run 0.5–1.5 min.

### What did not run, against the dossier's design (§A.3)

- The stratified bootstrap by subject (1,000 replicas) of the test macro-F1. The intervals are the Student-t over seeds and the paired differences.
- `modular_mlp` vs `feature_transformer`: #229 is not merged.
- The `mixing` α ∈ {0, 0.5, 1} scenario and the secondary task `stress_vs_exercise` (PhysioNet, 3 classes).
- The F1 per fog and the curve of F1 against rounds. The global model was scored every round, but those events stay in `events.jsonl`, which is not committed. CPU time is not measured, only wall time.
- **Leave-one-dataset-out scores the held-out study on its usual test subjects**, not on all its subjects as the dossier proposed. The in-domain and held-out numbers then come from the same subjects and can be paired.

## Learning rates (validation only)

`selection.csv` and `lab/selection.csv` hold every selection run, with its validation score and identity.

| Technique | E4: lr chosen | E4: score at 0.03 / 0.1 / 0.3 | Lab: lr chosen | Lab: score at 0.03 / 0.1 / 0.3 |
|---|---|---|---|---|
| FedAvg (global, val subjects) | 0.03 | 0.552 / 0.534 / 0.550 | 0.1 | 0.558 / 0.634 / 0.628 |
| SCAFFOLD (global, val subjects) | 0.03 | 0.671 / diverged / diverged | 0.1 | 0.586 / 0.631 / diverged |
| FedDyn (global, val subjects) | 0.03 | 0.658 / 0.538 / 0.480 | 0.1 | 0.631 / 0.648 / 0.618 |
| Ditto (personal, `local_val`) | 0.1 | 0.580 / 0.604 / 0.589 | 0.1 | 0.549 / 0.617 / 0.592 |
| FedRep (local, `local_val`) | 0.1 | 0.560 / 0.582 / 0.579 | 0.3 | 0.470 / 0.537 / 0.577 |
| FedBABU (fine-tuned, `local_val`) | 0.3 | 0.584 / 0.597 / 0.611 | 0.3 | 0.467 / 0.516 / 0.610 |

- "Diverged": the run lost rounds to non-finite weights (E4 SCAFFOLD: 11 and 16 lost rounds; lab SCAFFOLD at 0.3: 8) and has no score at round 20.
- In the E4 track, FedAvg, SCAFFOLD and FedDyn chose 0.03, the smallest lr of the grid. A smaller one might do better. The grid was fixed before running and was not extended.
- In the E4 track Ditto trains at 0.1 and FedAvg at 0.03, so Ditto's global model is not FedAvg's. In the lab track both use 0.1, and Ditto's global model is exactly FedAvg's.

## Results: E4 track

Identity: `data_id` `a5610d4e…` (the four studies; per study `321e8a45…` WESAD wrist, `c6dbe7c6…` PhysioNet, `b45d93b9…` Nurse, `4d64cbd9…` Stress-Predict). `topology_id` `5f5cd72e…` (studies), and per run in the appendix. Every number below is from `scores.csv` and `baselines.json`, scored by `scripts/stress_bench.py` on the test subjects of the protocol.

### Classical references

Logistic regression is deterministic on these splits, so each is a single number.

| Reference | WESAD wrist | PhysioNet | Nurse | Stress-Predict |
|---|---|---|---|---|
| Majority class | 0.391 | 0.457 | 0.459 | 0.399 |
| LR per study (centralised) | 0.877 | 0.563 | 0.542 | 0.641 |
| LR pooled over the four studies | 0.386 | 0.550 | 0.543 | 0.459 |
| Local-only LR, mean over training edges | 0.735 | 0.511 | 0.537 | 0.575 |
| Local-only LR, median over training edges | 0.761 | 0.509 | 0.531 | 0.579 |
| Training windows | 410 | 722 | 27,045 | 1,846 |

- Per-subject F1 of LR per study (median / p10): WESAD 0.934 / 0.745, PhysioNet 0.560 / 0.448, Nurse 0.515 / 0.342, Stress-Predict 0.663 / 0.508.
- Nurse holds 90% of the training windows, so the pooled regression is close to a Nurse model. On the laboratory studies it falls to the majority class.

### Federated scenarios: macro-F1 over the test windows (mean ± 95% CI, 5 seeds)

| Scenario | WESAD wrist | PhysioNet | Nurse | Stress-Predict |
|---|---|---|---|---|
| FedAvg, hierarchical, per_dataset | 0.850 ± 0.032 | 0.495 ± 0.016 | 0.552 ± 0.014 | 0.660 ± 0.036 |
| FedAvg, hierarchical, shared | 0.493 ± 0.066 | 0.556 ± 0.020 | 0.557 ± 0.010 | 0.498 ± 0.026 |
| FedAvg, flat, per_dataset | 0.850 ± 0.031 | 0.495 ± 0.016 | 0.550 ± 0.015 | 0.660 ± 0.039 |
| FedAvg, flat, shared | 0.492 ± 0.067 | 0.558 ± 0.013 | 0.557 ± 0.013 | 0.497 ± 0.027 |
| Ditto | 0.874 ± 0.029 | 0.577 ± 0.052 | 0.545 ± 0.018 | 0.667 ± 0.034 |
| FedRep | 0.833 ± 0.041 | 0.487 ± 0.065 | 0.578 ± 0.032 | 0.573 ± 0.098 |
| FedBABU | 0.888 ± 0.011 | 0.603 ± 0.038 | 0.556 ± 0.015 | 0.673 ± 0.015 |
| SCAFFOLD | 0.862 ± 0.030 | 0.544 ± 0.130 | 0.545 ± 0.086 | 0.648 ± 0.055 |
| FedDyn | 0.853 ± 0.014 | 0.640 ± 0.072 | 0.552 ± 0.044 | 0.663 ± 0.011 |

### Differences to FedAvg, hierarchical, `per_dataset`, paired by seed (macro-F1)

| Scenario | WESAD wrist | PhysioNet | Nurse | Stress-Predict |
|---|---|---|---|---|
| FedAvg, hierarchical, shared | -0.357 ± 0.064 | 0.061 ± 0.024 | 0.005 ± 0.021 | -0.162 ± 0.037 |
| FedAvg, flat, per_dataset | 0.000 ± 0.004 | 0.000 ± 0.001 | -0.002 ± 0.002 | -0.000 ± 0.003 |
| FedAvg, flat, shared | -0.358 ± 0.064 | 0.063 ± 0.019 | 0.005 ± 0.023 | -0.163 ± 0.036 |
| Ditto | 0.024 ± 0.044 | 0.082 ± 0.048 | -0.007 ± 0.021 | 0.007 ± 0.021 |
| FedRep | -0.016 ± 0.051 | -0.008 ± 0.055 | 0.027 ± 0.029 | -0.087 ± 0.113 |
| FedBABU | 0.038 ± 0.041 | 0.108 ± 0.047 | 0.005 ± 0.025 | 0.013 ± 0.030 |
| SCAFFOLD | 0.013 ± 0.016 | 0.049 ± 0.136 | -0.007 ± 0.096 | -0.012 ± 0.080 |
| FedDyn | 0.004 ± 0.046 | 0.145 ± 0.074 | 0.001 ± 0.045 | 0.003 ± 0.035 |

An interval that excludes 0 is a difference on these test subjects.

### Per-subject F1: median and 10th percentile over the test subjects

Median:

| Scenario | WESAD wrist | PhysioNet | Nurse | Stress-Predict |
|---|---|---|---|---|
| FedAvg, hierarchical, per_dataset | 0.952 ± 0.050 | 0.460 ± 0.006 | 0.534 ± 0.019 | 0.672 ± 0.040 |
| FedAvg, hierarchical, shared | 0.435 ± 0.047 | 0.452 ± 0.014 | 0.543 ± 0.016 | 0.474 ± 0.044 |
| FedAvg, flat, per_dataset | 0.952 ± 0.050 | 0.460 ± 0.006 | 0.530 ± 0.019 | 0.672 ± 0.044 |
| FedAvg, flat, shared | 0.435 ± 0.047 | 0.455 ± 0.012 | 0.544 ± 0.015 | 0.476 ± 0.043 |
| Ditto | 0.977 ± 0.019 | 0.535 ± 0.077 | 0.528 ± 0.017 | 0.659 ± 0.058 |
| FedRep | 0.924 ± 0.070 | 0.429 ± 0.043 | 0.566 ± 0.034 | 0.579 ± 0.096 |
| FedBABU | 0.979 ± 0.019 | 0.539 ± 0.071 | 0.539 ± 0.020 | 0.677 ± 0.020 |
| SCAFFOLD | 0.953 ± 0.044 | 0.550 ± 0.160 | 0.533 ± 0.076 | 0.642 ± 0.064 |
| FedDyn | 0.959 ± 0.013 | 0.591 ± 0.097 | 0.523 ± 0.051 | 0.684 ± 0.037 |

10th percentile:

| Scenario | WESAD wrist | PhysioNet | Nurse | Stress-Predict |
|---|---|---|---|---|
| FedAvg, hierarchical, per_dataset | 0.617 ± 0.069 | 0.430 ± 0.011 | 0.376 ± 0.024 | 0.548 ± 0.044 |
| FedAvg, hierarchical, shared | 0.367 ± 0.042 | 0.416 ± 0.006 | 0.357 ± 0.012 | 0.404 ± 0.020 |
| FedAvg, flat, per_dataset | 0.620 ± 0.066 | 0.430 ± 0.010 | 0.379 ± 0.023 | 0.550 ± 0.048 |
| FedAvg, flat, shared | 0.365 ± 0.045 | 0.414 ± 0.007 | 0.355 ± 0.013 | 0.407 ± 0.026 |
| Ditto | 0.681 ± 0.062 | 0.433 ± 0.026 | 0.361 ± 0.013 | 0.577 ± 0.043 |
| FedRep | 0.587 ± 0.096 | 0.376 ± 0.070 | 0.379 ± 0.021 | 0.465 ± 0.115 |
| FedBABU | 0.716 ± 0.029 | 0.430 ± 0.032 | 0.347 ± 0.011 | 0.574 ± 0.012 |
| SCAFFOLD | 0.651 ± 0.094 | 0.440 ± 0.023 | 0.364 ± 0.032 | 0.539 ± 0.071 |
| FedDyn | 0.643 ± 0.033 | 0.422 ± 0.024 | 0.360 ± 0.016 | 0.561 ± 0.018 |

### Balanced accuracy and AUROC

Balanced accuracy:

| Scenario | WESAD wrist | PhysioNet | Nurse | Stress-Predict |
|---|---|---|---|---|
| FedAvg, hierarchical, per_dataset | 0.841 ± 0.038 | 0.516 ± 0.007 | 0.547 ± 0.009 | 0.652 ± 0.030 |
| FedAvg, hierarchical, shared | 0.541 ± 0.050 | 0.550 ± 0.016 | 0.551 ± 0.007 | 0.533 ± 0.018 |
| FedAvg, flat, per_dataset | 0.842 ± 0.038 | 0.516 ± 0.007 | 0.546 ± 0.010 | 0.652 ± 0.032 |
| FedAvg, flat, shared | 0.540 ± 0.051 | 0.552 ± 0.010 | 0.551 ± 0.008 | 0.532 ± 0.018 |
| Ditto | 0.874 ± 0.030 | 0.566 ± 0.034 | 0.542 ± 0.011 | 0.661 ± 0.034 |
| FedRep | 0.820 ± 0.048 | 0.505 ± 0.052 | 0.588 ± 0.059 | 0.594 ± 0.071 |
| FedBABU | 0.887 ± 0.010 | 0.584 ± 0.028 | 0.550 ± 0.010 | 0.668 ± 0.016 |
| SCAFFOLD | 0.860 ± 0.046 | 0.594 ± 0.155 | 0.549 ± 0.052 | 0.667 ± 0.032 |
| FedDyn | 0.859 ± 0.014 | 0.620 ± 0.061 | 0.549 ± 0.029 | 0.657 ± 0.012 |

AUROC:

| Scenario | WESAD wrist | PhysioNet | Nurse | Stress-Predict |
|---|---|---|---|---|
| FedAvg, hierarchical, per_dataset | 0.944 ± 0.018 | 0.706 ± 0.036 | 0.738 ± 0.006 | 0.748 ± 0.015 |
| FedAvg, hierarchical, shared | 0.786 ± 0.021 | 0.650 ± 0.010 | 0.740 ± 0.003 | 0.702 ± 0.010 |
| FedAvg, flat, per_dataset | 0.944 ± 0.018 | 0.706 ± 0.037 | 0.738 ± 0.006 | 0.748 ± 0.015 |
| FedAvg, flat, shared | 0.786 ± 0.021 | 0.650 ± 0.011 | 0.740 ± 0.004 | 0.702 ± 0.009 |
| Ditto | 0.946 ± 0.014 | 0.746 ± 0.016 | 0.719 ± 0.012 | 0.736 ± 0.018 |
| FedRep | 0.943 ± 0.013 | 0.522 ± 0.160 | 0.731 ± 0.013 | 0.766 ± 0.021 |
| FedBABU | 0.958 ± 0.004 | 0.798 ± 0.020 | 0.725 ± 0.009 | 0.731 ± 0.008 |
| SCAFFOLD | 0.947 ± 0.023 | 0.779 ± 0.025 | 0.751 ± 0.014 | 0.777 ± 0.026 |
| FedDyn | 0.943 ± 0.011 | 0.803 ± 0.021 | 0.748 ± 0.009 | 0.730 ± 0.010 |

### Edge scores and cost

Each edge scores its own `local_val` rows at round 20. The table gives each run's sample-weighted mean over its edges (`summary.json`, the cloud's combination), then mean ± CI over seeds. These rows are close in time to the training rows, so the scores are optimistic in the same way for every scenario. Bytes count every message of a run.

| Scenario | Received | Trained (local) | Personal | Fine-tuned | MB per run |
|---|---|---|---|---|---|
| FedAvg, hierarchical, per_dataset | 0.543 ± 0.024 | 0.586 ± 0.009 | — | 0.581 ± 0.011 | 161.7 |
| FedAvg, hierarchical, shared | 0.537 ± 0.020 | 0.583 ± 0.004 | — | 0.577 ± 0.011 | 114.5 |
| FedAvg, flat, per_dataset | 0.542 ± 0.026 | 0.587 ± 0.010 | — | 0.582 ± 0.009 | 156.4 |
| FedAvg, flat, shared | 0.538 ± 0.024 | 0.582 ± 0.005 | — | 0.576 ± 0.011 | 110.2 |
| Ditto | 0.545 ± 0.022 | 0.606 ± 0.011 | 0.606 ± 0.011 | 0.605 ± 0.008 | 161.7 |
| FedRep | 0.559 ± 0.008 | 0.588 ± 0.008 | — | 0.580 ± 0.004 | 159.5 |
| FedBABU | 0.551 ± 0.013 | 0.614 ± 0.009 | — | 0.603 ± 0.010 | 161.7 |
| SCAFFOLD | 0.543 ± 0.048 | 0.578 ± 0.023 | — | 0.580 ± 0.013 | 317.9 |
| FedDyn | 0.549 ± 0.054 | 0.585 ± 0.014 | — | 0.578 ± 0.009 | 161.7 |

### Each study alone

FedAvg over one study's edges under one fog (`stress_per_study.yaml`), at the lr chosen for the joint federation (0.03):

| Study alone | WESAD wrist | PhysioNet | Nurse | Stress-Predict |
|---|---|---|---|---|
| macro-F1 | 0.864 ± 0.016 | 0.457 ± 0.000 | 0.548 ± 0.014 | 0.436 ± 0.048 |

PhysioNet alone predicts the majority class in every seed, and Stress-Predict alone is close to it. The lr was not selected for these runs: at 0.03 the two small studies may be under-trained, so the gap to the joint federation is an upper bound of what joining brings.

### Leave-one-dataset-out

FedAvg with the shared adapter on three studies; the fourth trains nothing and is scored on its usual test subjects. The in-domain counterpart is "FedAvg, hierarchical, shared" of `stress.yaml`, with the same seeds and test subjects.

| Held-out study | FL held out | FL in-domain (shared) | Paired difference | Per-subject F1 median / p10 (held out) | AUROC (held out) | LR pooled, held out | LR pooled, in-domain | Majority |
|---|---|---|---|---|---|---|---|---|
| Nurse | 0.478 ± 0.031 | 0.557 ± 0.010 | -0.079 ± 0.039 | 0.472 ± 0.025 / 0.341 ± 0.004 | 0.632 ± 0.018 | 0.561 | 0.543 | 0.459 |
| WESAD wrist | 0.479 ± 0.062 | 0.493 ± 0.066 | -0.013 ± 0.019 | 0.425 ± 0.041 / 0.365 ± 0.042 | 0.784 ± 0.020 | 0.386 | 0.386 | 0.391 |
| PhysioNet | 0.558 ± 0.023 | 0.556 ± 0.020 | 0.001 ± 0.004 | 0.455 ± 0.013 / 0.415 ± 0.006 | 0.650 ± 0.009 | 0.547 | 0.550 | 0.457 |
| Stress-Predict | 0.493 ± 0.028 | 0.498 ± 0.026 | -0.006 ± 0.007 | 0.472 ± 0.038 / 0.402 ± 0.019 | 0.700 ± 0.009 | 0.463 | 0.459 | 0.399 |

"LR pooled, held out" is logistic regression on the pooled windows of the other three studies (`lr_lodo` in `baselines.json`).

## Reading (E4 track)

- **Against the centralised regression per study, plain FedAvg never wins; FedBABU and FedDyn do on two studies.** The regression is one deterministic number per study. A federated scenario beats it here only if its whole interval is above that number:
  - WESAD wrist: FedAvg (0.850 ± 0.032) is below 0.877. The best scenario, FedBABU (0.888 ± 0.011), has its lower bound at 0.877: a tie.
  - PhysioNet: FedAvg (0.495 ± 0.016) is below 0.563. FedDyn (0.640 ± 0.072) and FedBABU (0.603 ± 0.038) have their lower bounds at 0.568 and 0.565, just above it. The margin is small and comes from five seeds.
  - Nurse: every scenario is within 0.54–0.58. Every interval but FedRep's contains 0.542, and FedRep's lower bound (0.546) clears it by 0.004, with a global model whose head is untrained. All are near the majority class (0.459): nothing here is a usable detector.
  - Stress-Predict: FedBABU (0.673 ± 0.015) and FedDyn (0.663 ± 0.011) are above 0.641, with lower bounds of 0.658 and 0.652. FedAvg (0.660 ± 0.036) contains it.
- **Against one regression pooled over all four studies, the federation with per-study adapters wins on the laboratory studies.** For WESAD wrist it scores 0.850 against 0.386, and for Stress-Predict 0.660 against 0.459. Nurse's 27,045 windows dominate the pooled data.
- **Flat and hierarchical FedAvg give the same model, as expected.** The paired differences are within ±0.004, from floating-point order. The hierarchy only changes the traffic: 162 MB against 156 MB per run, because four fogs send to the cloud instead of one.
- **The shared adapter costs the laboratory studies a lot.** With harmonized features and `adapters: shared`, WESAD wrist loses 0.357 ± 0.064 and Stress-Predict 0.162 ± 0.037. PhysioNet gains 0.061 ± 0.024, and Nurse is unchanged. Sample-weighted FedAvg makes the shared model a Nurse model, as in the pooled regression. The per-dataset adapters absorb the differences between studies.
- **Techniques (global model).** Against FedAvg, three intervals exclude 0, all on PhysioNet: FedDyn (+0.145 ± 0.074), FedBABU (+0.108 ± 0.047) and Ditto (+0.082 ± 0.048).
  - Ditto's global model is FedAvg at lr 0.1, so its gain may be the lr and not Ditto.
  - FedRep's global model keeps an untrained head (the head stays on the edges), so its global column does not measure it.
  - SCAFFOLD's intervals are wide: it is unstable at these learning rates, and it diverged above 0.03 in the selection.
- **Techniques (edges).** FedBABU's trained model (0.614 ± 0.009) and Ditto's personal model (0.606 ± 0.011) are above FedAvg's trained model (0.586 ± 0.009) on the edges' own rows.
- **Fairness.** The 10th percentile of per-subject F1 is low in every scenario: 0.35–0.38 on Nurse and 0.38–0.44 on PhysioNet. That is about the F1 of predicting a single class. FedBABU raises WESAD's 10th percentile from 0.617 to 0.716 ± 0.029.
- **Laboratory → hospital.** Trained on the three laboratory studies, the model scores 0.478 ± 0.031 on the test nurses, against 0.557 ± 0.010 when Nurse trains (paired −0.079 ± 0.039). That is barely above the majority class (0.459). The logistic regression pooled over the laboratories transfers better (0.561). For the other studies, holding them out changes little. But with the shared adapter they were already served poorly in-domain, so this says little about their transfer.
- **Joining helps the small studies.** Alone at lr 0.03, PhysioNet collapses to the majority class (0.457) and Stress-Predict reaches 0.436. In the joint federation they reach 0.495 and 0.660. WESAD wrist (0.864 alone, 0.850 joint) and Nurse (0.548 alone, 0.552 joint) do not change.

## Results: lab track (SWELL and WESAD)

A separate track: SWELL has no E4, the features and descriptors differ, and these numbers are not comparable with the E4 table. Identity: `data_id` `9fd57375…`, `topology_id` `ff55fe4d…` (hierarchical, the same as `techniques_drift`) and `adbaffcb…` (flat). The test subjects are those of the existing baselines.

| Reference | SWELL | WESAD |
|---|---|---|
| Majority class | 0.404 | 0.392 |
| LR per dataset (centralised) | 0.433 | 0.918 |
| Local-only LR, mean over training edges | 0.452 | 0.533 |

Federated, macro-F1 over the test windows (mean ± 95% CI, 5 seeds):

| Scenario | SWELL | WESAD |
|---|---|---|
| FedAvg, hierarchical | 0.555 ± 0.033 | 0.779 ± 0.008 |
| FedAvg, flat | 0.557 ± 0.039 | 0.778 ± 0.009 |
| Ditto | 0.555 ± 0.033 | 0.779 ± 0.008 |
| FedRep | 0.439 ± 0.026 | 0.799 ± 0.055 |
| FedBABU | 0.584 ± 0.032 | 0.777 ± 0.003 |
| SCAFFOLD | 0.491 ± 0.040 | 0.829 ± 0.055 |
| FedDyn | 0.579 ± 0.029 | 0.856 ± 0.048 |

Paired by seed against FedAvg, hierarchical:

| Scenario | SWELL | WESAD |
|---|---|---|
| FedAvg, flat | 0.002 ± 0.007 | -0.001 ± 0.003 |
| Ditto | 0.000 ± 0.000 | 0.000 ± 0.000 |
| FedRep | -0.115 ± 0.051 | 0.020 ± 0.048 |
| FedBABU | 0.029 ± 0.013 | -0.002 ± 0.006 |
| SCAFFOLD | -0.064 ± 0.037 | 0.050 ± 0.049 |
| FedDyn | 0.024 ± 0.034 | 0.077 ± 0.045 |

Per-subject F1, median:

| Scenario | SWELL | WESAD |
|---|---|---|
| FedAvg, hierarchical | 0.557 ± 0.033 | 0.973 ± 0.013 |
| FedAvg, flat | 0.554 ± 0.044 | 0.970 ± 0.013 |
| Ditto | 0.557 ± 0.033 | 0.973 ± 0.013 |
| FedRep | 0.424 ± 0.037 | 0.974 ± 0.034 |
| FedBABU | 0.592 ± 0.029 | 0.977 ± 0.010 |
| SCAFFOLD | 0.485 ± 0.040 | 0.977 ± 0.011 |
| FedDyn | 0.596 ± 0.024 | 0.950 ± 0.013 |

10th percentile:

| Scenario | SWELL | WESAD |
|---|---|---|
| FedAvg, hierarchical | 0.488 ± 0.041 | 0.411 ± 0.026 |
| FedAvg, flat | 0.491 ± 0.049 | 0.411 ± 0.027 |
| Ditto | 0.488 ± 0.041 | 0.411 ± 0.026 |
| FedRep | 0.415 ± 0.030 | 0.471 ± 0.198 |
| FedBABU | 0.518 ± 0.030 | 0.398 ± 0.002 |
| SCAFFOLD | 0.433 ± 0.031 | 0.574 ± 0.169 |
| FedDyn | 0.503 ± 0.047 | 0.705 ± 0.136 |

- **SWELL: the federation beats the centralised regression.** FedAvg reaches 0.555 ± 0.033, against 0.433 for the regression, which is itself barely above the majority class (0.404). FedBABU adds +0.029 ± 0.013. FedRep (−0.115 ± 0.051) and SCAFFOLD (−0.064 ± 0.037) lose.
- **WESAD: the regression wins.** It reaches 0.918, against 0.779 ± 0.008 for FedAvg and 0.856 ± 0.048 for FedDyn, the best scenario (+0.077 ± 0.045 paired). This matches the existing baselines (README §7).
- Flat and hierarchical are the same again (±0.007). Ditto's global model is FedAvg's, since both train at lr 0.1.

## Files

- `scores.csv` (E4: `stress_e4`, `stress_e4_lodo`, `stress_e4_per_study`) and `lab/scores.csv`: one row per run and test study. They hold the pooled macro-F1, balanced accuracy, AUROC, per-subject F1 (median, p10 and every subject), the round-20 edge scores, bytes and wall time, and the identity (`run_id`, `config_id`, `topology_id`, `data_id`, `run_hash`, commit).
- `baselines.json`, `lab/baselines.json`: the classical references, with per-subject F1.
- `selection.csv`, `lab/selection.csv`: the selection runs and their validation scores.
- `names.json`: scenario name → the label used in these tables.
- `runs/<run_id>/`, `lab/runs/<run_id>/`: `run.json` (identity, config, `run_hash`) and `summary.json` of every run, and `model.npz` for the scored runs (not for the selection runs).
  - `events.jsonl` (5–10 MB per run) is left out, and `host` and `pid` are removed from `run.json`, so `verify_run` cannot check these copies.
  - `scripts/stress_bench.py score` rebuilds every number of the tables from `model.npz` and the data.

```bash
python scripts/stress_bench.py baselines benchmarks/stress.yaml --out results/stress_e4/baselines.json
python scripts/stress_bench.py baselines benchmarks/stress_lab_select.yaml --out results/stress_e4/lab/baselines.json
python scripts/stress_bench.py score results/stress_e4/runs --out results/stress_e4/scores.csv
python scripts/stress_bench.py tables results/stress_e4/scores.csv --names results/stress_e4/names.json \
    --reference "FedAvg, hierarchical, per_dataset"
```

## Appendix: run identities

E4 track (`stress.yaml`, `stress_lodo.yaml`, `stress_per_study.yaml`):

| Scenario | `config_id` | `topology_id` | seed: `run_id` / `run_hash` / commit |
|---|---|---|---|
| FedAvg, hierarchical, shared | `24c4012618bb…` | `5f5cd72e08be…` | 0: `20261110T134304Z-67ecaa1b61fa` / `623755448485…` / `8bf2343`<br>1: `20261110T142801Z-05223671238b` / `35f50049bed0…` / `e965216`<br>2: `20261110T142801Z-3b9e7fce6196` / `130e9c062fdf…` / `e965216`<br>3: `20261110T142801Z-bab724f1134f` / `6b298034e0e0…` / `e965216`<br>4: `20261110T142801Z-22dcf643cf97` / `217dc26aae36…` / `e965216` |
| FedAvg, hierarchical, per_dataset | `4b062e38cda1…` | `5f5cd72e08be…` | 0: `20261110T134304Z-c4f3afd22650` / `bd76b21f551e…` / `8bf2343`<br>1: `20261110T134304Z-e33ac3a8ebee` / `bbda86cfadd3…` / `8bf2343`<br>2: `20261110T134304Z-6e2a82fe48fc` / `957aab2a5edf…` / `8bf2343`<br>3: `20261110T134304Z-afb03f0291ba` / `0b960f1339f8…` / `8bf2343`<br>4: `20261110T134304Z-69c55f87bb5e` / `5d1dcd63b4eb…` / `8bf2343` |
| FedAvg, flat, per_dataset | `981155bb02ba…` | `c4bb91105436…` | 0: `20261110T142801Z-f4be68fcc0ed` / `7b69a5b767ed…` / `e965216`<br>1: `20261110T142801Z-f1ce2218c4a2` / `68791e50a244…` / `e965216`<br>2: `20261110T134805Z-f7988355b7b8` / `953ebfada15e…` / `4d34d9f`<br>3: `20261110T134806Z-ccecd0b3f429` / `dbbff7e7d414…` / `4d34d9f`<br>4: `20261110T134807Z-e3007de6abef` / `8784906d362d…` / `4d34d9f` |
| FedAvg, flat, shared | `35ed0341a422…` | `c4bb91105436…` | 0: `20261110T134807Z-ee41e79d7844` / `6d8b5a1b99b3…` / `4d34d9f`<br>1: `20261110T134810Z-632af50cc935` / `a0cbc252d754…` / `4d34d9f`<br>2: `20261110T134811Z-c01474fe9486` / `45d0ed3c235c…` / `4d34d9f`<br>3: `20261110T135049Z-0d3f2e348456` / `69df60ae5658…` / `4d34d9f`<br>4: `20261110T135049Z-c942a16022cf` / `5e1276422ba2…` / `4d34d9f` |
| Ditto | `ea10404507ef…` | `5f5cd72e08be…` | 0: `20261110T135050Z-04041c4925ca` / `d0b10c453b68…` / `4d34d9f`<br>1: `20261110T135050Z-78e33033c076` / `f545c7332c46…` / `4d34d9f`<br>2: `20261110T135053Z-24ed56c77744` / `4efba61ed246…` / `4d34d9f`<br>3: `20261110T135057Z-aa7ba799b6c3` / `089415260437…` / `4d34d9f`<br>4: `20261110T135333Z-a74057d19e8e` / `b4e008228371…` / `4d34d9f` |
| FedRep | `20addc14a881…` | `5f5cd72e08be…` | 0: `20261110T135335Z-4c439cf9acaf` / `2996d53fcb30…` / `4d34d9f`<br>1: `20261110T135653Z-97d91dd020be` / `bfcdf3e9982c…` / `4d34d9f`<br>2: `20261110T135653Z-fad430a26c14` / `cee7fe127303…` / `4d34d9f`<br>3: `20261110T135658Z-3d7d8cca6305` / `64ec07477aac…` / `4d34d9f`<br>4: `20261110T135701Z-eb5d28b2fe92` / `86e187833a76…` / `4d34d9f` |
| FedBABU | `5a889e5635ef…` | `5f5cd72e08be…` | 0: `20261110T135713Z-1b74cd637b8e` / `772d38d670c1…` / `4d34d9f`<br>1: `20261110T135915Z-8a47d40bc4be` / `4f8daeb306c9…` / `4d34d9f`<br>2: `20261110T135930Z-e69a98079ace` / `857e4c81a8ff…` / `4d34d9f`<br>3: `20261110T140033Z-57406552988d` / `131ab03f7923…` / `4d34d9f`<br>4: `20261110T140034Z-d6b741478a5e` / `929a4567140c…` / `4d34d9f` |
| SCAFFOLD | `17d722a3ec8a…` | `5f5cd72e08be…` | 0: `20261110T140038Z-649d6d0f1e46` / `85d4e522c330…` / `4d34d9f`<br>1: `20261110T140041Z-0525a7d46106` / `9e738b3054ae…` / `4d34d9f`<br>2: `20261110T140132Z-2066137f0cbd` / `1fdb764647e2…` / `4d34d9f`<br>3: `20261110T140151Z-b8a971cea875` / `864663401b63…` / `4d34d9f`<br>4: `20261110T140257Z-5cabbfb15e08` / `f9546f00dfc7…` / `4d34d9f` |
| FedDyn | `08cac73a3dff…` | `5f5cd72e08be…` | 0: `20261110T140259Z-4f7471079f68` / `aeb76f9f6688…` / `4d34d9f`<br>1: `20261110T140355Z-713e4980f408` / `535c33b37490…` / `4d34d9f`<br>2: `20261110T140400Z-e8a1cf7d03cd` / `73331d8d0611…` / `4d34d9f`<br>3: `20261110T140444Z-bae509b1cae2` / `73114c1ae6bc…` / `4d34d9f`<br>4: `20261110T140507Z-9c493184aaee` / `136444a8af26…` / `4d34d9f` |
| LODO: Nurse held out | `da26301fdcfe…` | `511bd041634a…` | 0: `20261110T140844Z-4d22116aab27` / `3d22a04b8c80…` / `4d34d9f`<br>1: `20261110T140844Z-8394b19a469d` / `26fc53fd616b…` / `4d34d9f`<br>2: `20261110T140844Z-4c4f20ec31c3` / `fa890f78eb33…` / `4d34d9f`<br>3: `20261110T140844Z-061826c7bfee` / `dd4c4b8b8487…` / `4d34d9f`<br>4: `20261110T140844Z-908a17d1189f` / `459be7914fea…` / `4d34d9f` |
| LODO: WESAD held out | `cc5867539206…` | `904451ba68c9…` | 0: `20261110T140844Z-77b22008c42b` / `a356fbc50df7…` / `4d34d9f`<br>1: `20261110T140917Z-c924cb0c66e3` / `fd8effaa5472…` / `4d34d9f`<br>2: `20261110T140917Z-614897a795c2` / `a4d08d077598…` / `4d34d9f`<br>3: `20261110T140917Z-d3ec8b3a271e` / `3f0ff351e32c…` / `4d34d9f`<br>4: `20261110T140917Z-603506765785` / `81f37e2b716c…` / `4d34d9f` |
| LODO: PhysioNet held out | `ee1438c10aff…` | `ff2f72b44810…` | 0: `20261110T140917Z-77c1bdd7476c` / `a62b976b34c6…` / `4d34d9f`<br>1: `20261110T141114Z-d44ca55ba518` / `5b30e5dfa7fd…` / `4d34d9f`<br>2: `20261110T141139Z-3e833299a107` / `b4f9f27fe5d3…` / `4d34d9f`<br>3: `20261110T141144Z-70a35fc9c5b3` / `ecfd7ece5954…` / `4d34d9f`<br>4: `20261110T141144Z-0a720fa1eb35` / `26f033bf2085…` / `4d34d9f` |
| LODO: Stress-Predict held out | `efe7b12dea92…` | `fec4e2d79de3…` | 0: `20261110T141146Z-6acfaa11416e` / `bd1fca28eee6…` / `4d34d9f`<br>1: `20261110T141146Z-3860aa725081` / `e036389eba2c…` / `4d34d9f`<br>2: `20261110T141329Z-9ce22522e6a0` / `3ed17bb877e8…` / `4d34d9f`<br>3: `20261110T141350Z-c4a9a6eb0d8d` / `2733dcc6bf22…` / `4d34d9f`<br>4: `20261110T141351Z-d218ef568b5b` / `cff9a7b176c0…` / `4d34d9f` |
| Per study: WESAD | `3198fabf6498…` | `18086603f5b1…` | 0: `20261110T141520Z-65b7b20ff7e0` / `738906eb96d3…` / `4d34d9f`<br>1: `20261110T141520Z-731e2790cf32` / `442b51ecb47d…` / `4d34d9f`<br>2: `20261110T141520Z-da8a81e2801b` / `210d10b93d85…` / `4d34d9f`<br>3: `20261110T141520Z-8e57ac028516` / `24d1d09064cc…` / `4d34d9f`<br>4: `20261110T141520Z-be1cb3c9877f` / `c4bb536a0b25…` / `4d34d9f` |
| Per study: PhysioNet | `0327d484b46b…` | `18086603f5b1…` | 0: `20261110T141520Z-f2d34da2f3bd` / `0e4dbca3a868…` / `4d34d9f`<br>1: `20261110T141527Z-400ea2ae79f7` / `6c559b4c7817…` / `4d34d9f`<br>2: `20261110T141527Z-ed85a2a9b2db` / `a2819a4376f0…` / `4d34d9f`<br>3: `20261110T141527Z-c926ded3c4f5` / `a6d5aa03fa4e…` / `4d34d9f`<br>4: `20261110T141527Z-2101c137d107` / `4f0b68557915…` / `4d34d9f` |
| Per study: Nurse | `b96c682eebe9…` | `18086603f5b1…` | 0: `20261110T141527Z-50e98c6ddab7` / `ad9b51748bbf…` / `4d34d9f`<br>1: `20261110T141532Z-3af26f1b09d3` / `81e04f4c4709…` / `4d34d9f`<br>2: `20261110T141536Z-7fa73e741306` / `97a4b8101add…` / `4d34d9f`<br>3: `20261110T141536Z-c9043ed2269d` / `e09f75335129…` / `4d34d9f`<br>4: `20261110T141537Z-ecdecdd56cb1` / `633682958844…` / `4d34d9f` |
| Per study: Stress-Predict | `2ab246c7d2f0…` | `18086603f5b1…` | 0: `20261110T141537Z-af0d90d3dde6` / `75f5a6565eb5…` / `4d34d9f`<br>1: `20261110T141549Z-fed7f2fec958` / `62f86d212837…` / `4d34d9f`<br>2: `20261110T141603Z-9deac63c400e` / `8aae3cddf2f1…` / `4d34d9f`<br>3: `20261110T141617Z-f14ae84f67ee` / `945fdf0bcab6…` / `4d34d9f`<br>4: `20261110T141631Z-1eaa9f312263` / `2ff02cd22b1a…` / `4d34d9f` |

Lab track (`stress_lab.yaml`):

| Scenario | `config_id` | `topology_id` | seed: `run_id` / `run_hash` / commit |
|---|---|---|---|
| FedAvg, hierarchical | `b8bb7eec26b6…` | `ff55fe4d2ce1…` | 0: `20261110T142157Z-7e0a01cc5488` / `f81313839c1a…` / `e965216`<br>1: `20261110T142157Z-bf2a369b7b30` / `e6fef191bd31…` / `e965216`<br>2: `20261110T142157Z-a84f07f02993` / `f173cc014062…` / `e965216`<br>3: `20261110T142157Z-882068e6d597` / `9b6bd43aae13…` / `e965216`<br>4: `20261110T142157Z-0bb02e0b0a8b` / `f34dcd028349…` / `e965216` |
| FedAvg, flat | `bd48e60cb6b7…` | `adbaffcb9b18…` | 0: `20261110T142157Z-b222f12189fa` / `a0acfe6aa2d7…` / `e965216`<br>1: `20261110T142243Z-344a9c6add00` / `9174ef0e7ee1…` / `e965216`<br>2: `20261110T142245Z-fa0aca6bf35b` / `c9909330d853…` / `e965216`<br>3: `20261110T142245Z-3ed73aed7434` / `5c2e11656d7e…` / `e965216`<br>4: `20261110T142246Z-b8b4a9a0eed3` / `d3debd1097d7…` / `e965216` |
| Ditto | `d2602b0fe13d…` | `ff55fe4d2ce1…` | 0: `20261110T142247Z-d55b93f89e9e` / `1d04f944d841…` / `e965216`<br>1: `20261110T142247Z-3d11de7aa410` / `5d30f4624d03…` / `e965216`<br>2: `20261110T142326Z-002c7182a013` / `e62dce720eae…` / `e965216`<br>3: `20261110T142327Z-67c12d223a32` / `d0a5de293b1a…` / `e965216`<br>4: `20261110T142327Z-d59a717da78f` / `aa5ebd714b98…` / `e965216` |
| FedRep | `d53f8b08011e…` | `ff55fe4d2ce1…` | 0: `20261110T142328Z-d8733b567fd7` / `d791a864552a…` / `e965216`<br>1: `20261110T142416Z-90a21f64cd40` / `c3733007d085…` / `e965216`<br>2: `20261110T142416Z-fb96693b3f4c` / `79617c92a3b5…` / `e965216`<br>3: `20261110T142425Z-6729351d0399` / `38ccc1abe219…` / `e965216`<br>4: `20261110T142455Z-dfed00120cb5` / `0d5b6a20f1c2…` / `e965216` |
| FedBABU | `dc5dda865f0d…` | `ff55fe4d2ce1…` | 0: `20261110T142455Z-57baf5405642` / `99ae0a4dec3f…` / `e965216`<br>1: `20261110T142457Z-66bf1b44edc3` / `d83a72203e5b…` / `e965216`<br>2: `20261110T142508Z-4d7f291ddf1a` / `68692303574c…` / `e965216`<br>3: `20261110T142509Z-37f92b4ea4d0` / `05c92ff31e15…` / `e965216`<br>4: `20261110T142517Z-89b7401f90f6` / `a7db7c57a346…` / `e965216` |
| SCAFFOLD | `db7e16bee927…` | `ff55fe4d2ce1…` | 0: `20261110T142535Z-3a7f93a3f97a` / `04debccd9604…` / `e965216`<br>1: `20261110T142536Z-e2eba0f5ae1b` / `b68f600af53b…` / `e965216`<br>2: `20261110T142544Z-0f3397dc2a6d` / `9ae174e2bc0f…` / `e965216`<br>3: `20261110T142548Z-5746509224a1` / `61df237a9fa8…` / `e965216`<br>4: `20261110T142549Z-33334b3486e5` / `fa900fbe0472…` / `e965216` |
| FedDyn | `a707bc6ff5dc…` | `ff55fe4d2ce1…` | 0: `20261110T142557Z-16a81d5629de` / `ec836a5cde78…` / `e965216`<br>1: `20261110T142644Z-9a5fbe53c50f` / `6d274cf654b2…` / `e965216`<br>2: `20261110T142645Z-23941ba05db1` / `49d42b90c1cd…` / `e965216`<br>3: `20261110T142652Z-75e70ab0dfbb` / `a6101727b9a2…` / `e965216`<br>4: `20261110T142658Z-92d9c6df5711` / `1c4c16e87cec…` / `e965216` |
