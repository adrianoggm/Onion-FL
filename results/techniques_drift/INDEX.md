# Non-IID drift techniques (new framework, real data)

`experiments/techniques_drift.yaml` was run on 2026-10-03 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #148.

```bash
onion_fl run experiments/techniques_drift.yaml --workers 3
onion_fl report runs --out report.html --metric macro_f1 --by scenario --by model
```

## Setup

- **Topology and placement.** `four_fogs_swell_wesad` with a `mixing` placement at α = 0 (segregated): two fogs hold only SWELL edges and two hold only WESAD edges. Each edge holds one subject.
- **Test subjects.** SWELL tests on subjects 1, 2, 12, 18 and 19, and WESAD on S6, S7 and S10.
- **Edge validation.** Each edge holds out the last 20% of each class.
- **Model and training.** The model is a modular MLP with an adapter per dataset and a shared trunk and head. Every scenario trains 20 rounds of 10 local epochs with SGD, on seeds 0–2.
- **How the learning rates were chosen.**
  - Each technique's rate was picked from 0.03, 0.1 and 0.3 on the validation subjects (the fogs' zone evaluators, round 20, seed 0). MOON's contrast weight μ was picked the same way. The test subjects were scored only in the final run.
  - Every selection run is in `validation.csv`.
  - SCAFFOLD and FedNova were re-selected after the review fixes changed their code (`drift_lr_reselect`), and kept the same rates.

| Scenario | Trainer | Server optimizer | lr | Other |
|---|---|---|---|---|
| FedAvg | `standard` | `replace` | 0.3 | — |
| FedProx | `fedprox` | `replace` | 0.3 | μ = 0.01 (default) |
| SCAFFOLD | `scaffold` | `replace` | 0.1 | 0.3 is a tie on validation (0.585 against 0.586). In an earlier run at 0.3, one WESAD edge diverged |
| FedNova | `fednova` | `fednova` | 0.3 | — |
| FedDyn | `feddyn` | `feddyn` | 0.1 | α = 0.01 (default) |
| MOON | `moon` | `replace` | 0.3 | μ = 0.1. On validation μ = 1 gave 0.413, μ = 0.1 gave 0.611 and μ = 0 (plain training) gave 0.661 |

## Identity

| Identity | Value |
|---|---|
| `topology_id` | `9033d1cf2c9e348b4067c92173a36fd151b3077cf0c93b5a592d8d188df05f32` |
| `data_id` | `06ac5ff0d43080f42ccd968956b1f1e666a651cf8caecfa4ecd4e2b695e1f64e` |
| code | commit `f62390b`, clean tree |
| `config_id` FedAvg | `84f280e6adb490455bab0c6785d82dea67c7b12ba3dc1e2f18d0fccb9d670978` |
| `config_id` FedProx | `05ac2e30c6809a087e8b7a703425bbb4da9321228c4562c3e0bb6b854acdb6f7` |
| `config_id` SCAFFOLD | `b3729eb93d1257d856a9785a86a52f5239a4724f01815a23174deb6d93a4f263` |
| `config_id` FedNova | `fe509f0227287b3f3de51a9ccb6cf07c3fd50261ead779e7d997cd3e8daa16a7` |
| `config_id` FedDyn | `a427173245ca049af7a6fe322f40891a57ca6b741e82e473f1a8d956cc91acf1` |
| `config_id` MOON | `742664f00bcc1a32f12653da2173c7533086ec82bb486ec0a3c0e9b3675e927d` |

## Results (mean ± 95% CI over 3 seeds)

- **Global:** the final global model's macro-F1 on the test subjects (`run_metrics.csv`).
- **Edge:** each edge's macro-F1 on its own held-out rows at round 20. Each run's value is the mean over its edges, weighted by samples (28 edges, or 27 when one edge had not scored). These are computed from the edges' own events (`edge_scores.csv`).
- **Trunk alignment:** the mean cosine between the trunk updates of a fog's children (`diagnostic.divergence_cos`), averaged over rounds and fogs (`run_metrics.csv`). Higher means the clients pull in more similar directions, i.e. less drift.

| Scenario | Global SWELL | Global WESAD | Edge received | Edge trained | Trunk alignment |
|---|---|---|---|---|---|
| FedAvg | 0.579 ± 0.034 | 0.749 ± 0.009 | 0.599 ± 0.032 | 0.581 ± 0.028 | 0.159 ± 0.087 |
| FedProx | 0.578 ± 0.046 | 0.751 ± 0.000 | 0.595 ± 0.065 | 0.582 ± 0.016 | 0.160 ± 0.087 |
| SCAFFOLD | 0.403 ± 0.000 | 0.772 ± 0.056 | 0.485 ± 0.026 | 0.500 ± 0.017 | 0.204 ± 0.029 |
| FedNova | 0.595 ± 0.039 | 0.751 ± 0.000 | 0.605 ± 0.051 | 0.589 ± 0.023 | 0.138 ± 0.082 |
| FedDyn | 0.533 ± 0.078 | 0.775 ± 0.045 | 0.568 ± 0.084 | 0.604 ± 0.053 | 0.134 ± 0.139 |
| MOON | 0.542 ± 0.074 | 0.737 ± 0.059 | 0.578 ± 0.053 | 0.555 ± 0.009 | 0.153 ± 0.073 |

## Reading

- **No drift technique clearly beats FedAvg.**
  - FedNova (0.595) and FedProx (0.578) are within FedAvg's interval on SWELL.
  - FedProx at its default μ = 0.01 is nearly FedAvg.
  - FedNova's per-key step normalisation matters because SWELL edges take about 39 steps per round and WESAD edges 20.
- **SCAFFOLD aligns the clients but stalls SWELL.**
  - Its trunk updates are the most aligned (0.204 against FedAvg's 0.159), which is what it is designed to do, and it is among the best on WESAD (0.772).
  - But at lr 0.1 its global model stays at "always stress" on SWELL (0.403 in every seed) for all 20 rounds, while its training loss falls more slowly than FedAvg's.
- **FedDyn gives the best WESAD score (0.775) and the best trained-edge score (0.604).** Its intervals overlap FedAvg's.
  - Its global and edge state drift apart because the cloud loses rounds, as its plugin text states.
  - Its participation share is the round's, not each key's.
- **MOON's contrast does not help on this tabular task.** Every μ > 0 scored below plain training on validation, and its test intervals overlap FedAvg's.
- **Lost rounds.** The lossy links cost the cloud 7–11 of its 20 rounds per run (`run_metrics.csv`). The pattern is the same for every technique, because packet loss depends only on the seed.
- **Next step.** With three seeds most intervals overlap. More seeds, or per-technique tuning of μ and α, would separate them.

## Files

- `<run_id>/`: `run.json` (identity, config, `run_hash`), `summary.json` and `model.npz` of each run.
- `run_metrics.csv`: per run, the final global macro-F1 per dataset, the lost cloud rounds and the trunk alignment.
- `edge_scores.csv`: round-20 edge macro-F1 per run and model.
- `validation.csv`: every learning-rate and μ selection run, with its validation score and commit.
- `report.html`: macro-F1 per round by scenario and model, mean ± CI over the seeds.

`events.jsonl` (about 3 MB per run) is left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
