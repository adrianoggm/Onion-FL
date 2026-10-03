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
- **How the learning rates were chosen.** Each technique's rate was picked from 0.03, 0.1 and 0.3 on the validation subjects (the fogs' zone evaluators, round 20, seed 0). MOON's contrast weight μ was picked the same way. The test subjects were scored only in the final run.

| Scenario | Trainer | Server optimizer | lr | Other |
|---|---|---|---|---|
| FedAvg | `standard` | `replace` | 0.3 | — |
| FedProx | `fedprox` | `replace` | 0.3 | μ = 0.01 (default) |
| SCAFFOLD | `scaffold` | `replace` | 0.1 | at 0.3 one WESAD edge diverged and its overflowing weights reached the global model |
| FedNova | `fednova` | `fednova` | 0.3 | — |
| FedDyn | `feddyn` | `feddyn` | 0.1 | α = 0.01 (default) |
| MOON | `moon` | `replace` | 0.3 | μ = 0.1; on validation μ = 1 gave 0.41, μ = 0.1 gave 0.61 and μ = 0 (plain training) gave 0.66 |

## Identity

| Identity | Value |
|---|---|
| `topology_id` | `9033d1cf2c9e348b4067c92173a36fd151b3077cf0c93b5a592d8d188df05f32` |
| `data_id` | `06ac5ff0d43080f42ccd968956b1f1e666a651cf8caecfa4ecd4e2b695e1f64e` |
| code | commit `2564b81`, clean tree |
| `config_id` FedAvg | `84f280e6adb490455bab0c6785d82dea67c7b12ba3dc1e2f18d0fccb9d670978` |
| `config_id` FedProx | `05ac2e30c6809a087e8b7a703425bbb4da9321228c4562c3e0bb6b854acdb6f7` |
| `config_id` SCAFFOLD | `b3729eb93d1257d856a9785a86a52f5239a4724f01815a23174deb6d93a4f263` |
| `config_id` FedNova | `fe509f0227287b3f3de51a9ccb6cf07c3fd50261ead779e7d997cd3e8daa16a7` |
| `config_id` FedDyn | `a427173245ca049af7a6fe322f40891a57ca6b741e82e473f1a8d956cc91acf1` |
| `config_id` MOON | `742664f00bcc1a32f12653da2173c7533086ec82bb486ec0a3c0e9b3675e927d` |

## Results (mean ± 95% CI over 3 seeds)

- **Global:** the final global model's macro-F1 on the test subjects, from `summary.json`.
- **Edge:** each edge's macro-F1 on its own held-out rows at round 20. Each run's value is the mean over its edges, weighted by samples (28 edges, or 27 when one edge had not scored). These are computed from the edges' own events and kept in `edge_scores.csv`.
- **Trunk alignment:** the mean cosine between the trunk updates of a fog's children (`diagnostic.divergence_cos`), averaged over rounds and fogs. Higher means the clients pull in more similar directions, i.e. less drift.

| Scenario | Global SWELL | Global WESAD | Edge received | Edge trained | Trunk alignment |
|---|---|---|---|---|---|
| FedAvg | 0.579 ± 0.034 | 0.749 ± 0.009 | 0.599 ± 0.032 | 0.581 ± 0.028 | 0.159 ± 0.087 |
| FedProx | 0.578 ± 0.046 | 0.751 ± 0.000 | 0.595 ± 0.065 | 0.582 ± 0.016 | 0.160 ± 0.087 |
| SCAFFOLD | 0.403 ± 0.000 | 0.789 ± 0.126 | 0.478 ± 0.032 | 0.503 ± 0.011 | 0.208 ± 0.028 |
| FedNova | 0.585 ± 0.031 | 0.751 ± 0.000 | 0.605 ± 0.041 | 0.588 ± 0.033 | 0.133 ± 0.094 |
| FedDyn | 0.533 ± 0.078 | 0.775 ± 0.045 | 0.568 ± 0.084 | 0.604 ± 0.053 | 0.134 ± 0.139 |
| MOON | 0.542 ± 0.074 | 0.737 ± 0.059 | 0.578 ± 0.053 | 0.555 ± 0.009 | 0.153 ± 0.073 |

## Reading

- **No drift technique beats FedAvg on SWELL.**
  - FedNova (0.585) and FedProx (0.578) match it.
  - FedProx at its default μ = 0.01 is nearly FedAvg, and FedNova's normalisation matters little when edges take similar numbers of steps (20 to 40).
- **SCAFFOLD trades SWELL for WESAD.**
  - It is the best on WESAD (0.789 ± 0.126), and its trunk updates are the most aligned (0.208), which is what it is designed to do.
  - But at lr 0.1 its global model stays at "always stress" on SWELL (0.403 in every seed) for all 20 rounds. Its training loss falls more slowly than FedAvg's.
  - At lr 0.3 one WESAD edge diverged, which no aggregator here filters.
- **FedDyn helps WESAD and the edges a little.** It reaches 0.775 on WESAD and the best trained-edge score (0.604), but its intervals overlap FedAvg's.
- **MOON's contrast does not help on this tabular task.** On validation every μ > 0 scored below plain training.
- **Lost rounds.** The lossy links cost the cloud 7–11 of its 20 rounds per run. The pattern is the same for every technique, because packet loss depends only on the seed.
- **Next step.** With three seeds most intervals overlap; more seeds, or per-technique tuning of μ and α, would separate them.

## Files

- `<run_id>/`: `run.json` (identity, config, `run_hash`), `summary.json` and `model.npz` of each run.
- `edge_scores.csv`: round-20 edge macro-F1 per run and model.
- `report.html`: macro-F1 per round by scenario and model, mean ± CI over the seeds.

`events.jsonl` (about 3 MB per run) is left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
