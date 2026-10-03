# Non-IID drift techniques (new framework, real data)

`experiments/techniques_drift.yaml` was run on 2026-10-03 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #148.

```bash
onion_fl run experiments/techniques_drift_select.yaml --workers 3   # learning rates, validation only
onion_fl run experiments/techniques_drift.yaml --workers 3
onion_fl report experiments/techniques_drift.yaml --out report.html --metric macro_f1 --by scenario --by model
```

This replaces the first version of the comparison, which ran over lossy links and with SCAFFOLD and FedDyn as they were before the review of PR #154. The same techniques over lossy links are in [techniques_drift_lossy/](../techniques_drift_lossy/INDEX.md).

## Setup

- **Topology and placement.** `four_fogs_swell_wesad_lossless`: the latencies and bandwidths of `four_fogs_swell_wesad`, without message loss. No cloud round was lost in any run. The `mixing` placement at α = 0 is segregated: two fogs hold only SWELL edges and two hold only WESAD edges. Each edge holds one subject.
- **Test subjects.** SWELL tests on subjects 1, 2, 12, 18 and 19, and WESAD on S6, S7 and S10.
- **Edge validation.** Each edge holds out the last 20% of each class.
- **Model and training.** The model is a modular MLP with an adapter per dataset and a shared trunk and head. Every scenario trains 20 rounds of 10 local epochs with SGD, on seeds 0–2.
- **How the learning rates were chosen.**
  - `experiments/techniques_drift_select.yaml` ran every technique at 0.03, 0.1 and 0.3 on seed 0. The score was the macro-F1 of the fogs' zone evaluators (validation subjects) at round 20. The test subjects played no part.
  - Every technique scored best at 0.1. At 0.3, SCAFFOLD diverged: edges were left with non-finite weights, six rounds failed, and round 20 has no validation score.
  - MOON keeps μ = 0.1. At lr 0.3, its default μ = 1 scored 0.584 against 0.629.
  - The other parameters keep their defaults: FedProx μ = 0.01, FedDyn α = 0.01, MOON temperature 0.5.
  - Every selection run is in `validation.csv`. They ran at commits `6139e9d` and `9ca377d`, which differ only in documentation.
- **What the review of PR #154 changed.**
  - SCAFFOLD edges send the change in their control variate, and the `scaffold` server optimizer keeps `c`. It moves `c` by the share of each key's holders that trained.
  - FedDyn's server step uses the same per-key share.
  - Both average edges uniformly, as their papers do; the other techniques weight edges by examples.

| Scenario | Trainer | Server optimizer | lr | Other |
|---|---|---|---|---|
| FedAvg | `standard` | `replace` | 0.1 | — |
| FedProx | `fedprox` | `replace` | 0.1 | μ = 0.01 |
| SCAFFOLD | `scaffold` | `scaffold` | 0.1 | — |
| FedNova | `fednova` | `fednova` | 0.1 | — |
| FedDyn | `feddyn` | `feddyn` | 0.1 | α = 0.01 |
| MOON | `moon` | `replace` | 0.1 | μ = 0.1 |

## Identity

| Identity | Value |
|---|---|
| `topology_id` | `ff55fe4d2ce1a807f764bcc5205a8d2d4d43ab3d533163a1d13e173a9a6725fa` |
| `data_id` | `06ac5ff0d43080f42ccd968956b1f1e666a651cf8caecfa4ecd4e2b695e1f64e` |
| code | commit `90ef785`, clean tree |
| `config_id` FedAvg | `dee7e258cd04201d4f35f68eee71ad4379d9b4d9c76db3b2a0f2dfd8ff3a4f37` |
| `config_id` FedProx | `b3aa3714891d9e19809fac2a3498ba393d3f8712dba0f01f4a1e6c241bcc0499` |
| `config_id` SCAFFOLD | `adf22c57993120cf42f063a4f8dbeed5546dd50485464ad01db5c42f4da97d49` |
| `config_id` FedNova | `e0a52b678458d5dd416fc2154697bff85e3b5f65e04683b4613112d751d88c24` |
| `config_id` FedDyn | `3c889f55c5ffeb8b78a1b20faaedb7227f733cecfd3224f6a688c44e42363e71` |
| `config_id` MOON | `e63faf22629824803d6f53acf5896db50a3769d0ccda0a7acd97f59d8d383d6c` |

## Results (mean ± 95% CI over 3 seeds)

- **Global:** the final global model's macro-F1 on the test subjects (`run_metrics.csv`).
- **Edge:** each edge's macro-F1 on its own held-out rows at round 20, for the model it received and for the model after its local training. Each run's value is the mean over its 28 edges, weighted by samples (`edge_scores.csv`).
- **Trunk alignment:** the mean cosine between the trunk updates of a fog's children (`diagnostic.divergence_cos`), averaged over rounds and fogs (`run_metrics.csv`).

| Scenario | Global SWELL | Global WESAD | Edge received | Edge trained | Trunk alignment |
|---|---|---|---|---|---|
| FedAvg | 0.549 ± 0.081 | 0.752 ± 0.005 | 0.587 ± 0.055 | 0.581 ± 0.040 | 0.177 ± 0.073 |
| FedProx | 0.545 ± 0.095 | 0.750 ± 0.012 | 0.584 ± 0.036 | 0.584 ± 0.042 | 0.177 ± 0.075 |
| SCAFFOLD | 0.487 ± 0.072 | 0.828 ± 0.027 | 0.555 ± 0.032 | 0.559 ± 0.045 | 0.171 ± 0.024 |
| FedNova | 0.561 ± 0.085 | 0.754 ± 0.014 | 0.598 ± 0.030 | 0.595 ± 0.056 | 0.151 ± 0.051 |
| FedDyn | 0.593 ± 0.056 | 0.876 ± 0.010 | 0.615 ± 0.031 | 0.616 ± 0.046 | 0.055 ± 0.010 |
| MOON | 0.551 ± 0.078 | 0.754 ± 0.014 | 0.587 ± 0.061 | 0.575 ± 0.049 | 0.170 ± 0.063 |

## Reading

- **On WESAD, FedDyn and SCAFFOLD beat FedAvg in these runs.**
  - FedDyn reaches 0.876 ± 0.010 and SCAFFOLD 0.828 ± 0.027, against 0.752 ± 0.005 for FedAvg. The three intervals do not overlap.
  - Both correct the client drift with state kept on both sides of the link.
- **On SWELL, no technique separates from FedAvg.** All intervals overlap. FedDyn has the highest mean (0.593) and SCAFFOLD the lowest (0.487).
- **FedProx at μ = 0.01 is nearly FedAvg**, and MOON's contrast at μ = 0.1 does not change the outcome on this tabular task.
- **Trunk alignment does not track accuracy.** FedDyn has the least aligned trunk updates (0.055) and the best scores. Its linear term changes the local objective, so the cosine of the raw updates is not a clean measure of drift for it.
- **Next step.** Three seeds leave the SWELL intervals wide. More seeds, or tuning FedDyn's α and FedProx's μ on validation, would separate them.

## Files

- `<run_id>/`: `run.json` (identity, config, `run_hash`), `summary.json` and `model.npz` of each of the 18 runs.
- `run_metrics.csv`: per run, the final global macro-F1 per dataset, the lost cloud rounds (all 0) and the trunk alignment.
- `edge_scores.csv`: round-20 edge macro-F1 per run and model.
- `validation.csv`: the 19 selection runs, with their validation score and commit.
- `report.html`: macro-F1 per round by scenario and model, mean ± CI over the seeds.

`events.jsonl` (about 3 MB per run) is left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
