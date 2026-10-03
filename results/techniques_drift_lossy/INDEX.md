# Non-IID drift techniques over lossy links (new framework, real data)

`experiments/techniques_drift_lossy.yaml` was run on 2026-10-03. Issue #148.

```bash
onion_fl run experiments/techniques_drift_lossy.yaml --workers 3
onion_fl report experiments/techniques_drift_lossy.yaml --out report.html --metric macro_f1 --by scenario --by model
```

It is the comparison of [techniques_drift/](../techniques_drift/INDEX.md) over `four_fogs_swell_wesad`, whose links lose messages: the fog → cloud links (wifi) lose 0.5% and the edge → fog links (4g) 1%. Data, placement, model, techniques and learning rates are the same. The rates were chosen on lossless links, so they are not each technique's best rate on lossy ones.

## Identity

| Identity | Value |
|---|---|
| `topology_id` | `9033d1cf2c9e348b4067c92173a36fd151b3077cf0c93b5a592d8d188df05f32` |
| `data_id` | `06ac5ff0d43080f42ccd968956b1f1e666a651cf8caecfa4ecd4e2b695e1f64e` |
| code | commit `90ef785`, clean tree |
| `config_id` FedAvg | `8c6ce8f724732f8c9d38deae440fd9962daf7c136662f58de59c84ceb4258ce7` |
| `config_id` FedProx | `32e1e7597ab8d4ef6d6d8e3e9f74dd24ca2f8e38ceb216385db8676f7262e7e0` |
| `config_id` SCAFFOLD | `0614926eabc56e8dde6c31fa98d3ac2baa2030a05713f1707b60d4ad9dc59c47` |
| `config_id` FedNova | `45288d73b599ca8d38a4b83a897178634461968ab8cd80440d357d269bd0f5d5` |
| `config_id` FedDyn | `3a72779346907d759265723c5d4248f4e224785904bdcb3d253e9a920263c14d` |
| `config_id` MOON | `9cd03f03bd6addd7641df7f1b60b75e78dfa60c6106e57d24cfa1f626492f9c1` |

## Results (mean ± 95% CI over 3 seeds)

The columns are those of the lossless comparison. Edge scores average 27 or 28 edges, because an edge whose round-20 model never arrived did not score.

| Scenario | Global SWELL | Global WESAD | Edge received | Edge trained | Trunk alignment | Lost cloud rounds (seeds 0, 1, 2) |
|---|---|---|---|---|---|---|
| FedAvg | 0.416 ± 0.017 | 0.727 ± 0.044 | 0.486 ± 0.032 | 0.554 ± 0.031 | 0.219 ± 0.071 | 11, 10, 7 |
| FedProx | 0.415 ± 0.019 | 0.727 ± 0.044 | 0.484 ± 0.027 | 0.549 ± 0.030 | 0.220 ± 0.071 | 11, 10, 7 |
| SCAFFOLD | 0.405 ± 0.007 | 0.799 ± 0.103 | 0.467 ± 0.034 | 0.507 ± 0.020 | 0.286 ± 0.135 | 11, 10, 7 |
| FedNova | 0.510 ± 0.029 | 0.744 ± 0.015 | 0.528 ± 0.107 | 0.559 ± 0.040 | 0.215 ± 0.081 | 11, 10, 7 |
| FedDyn | 0.510 ± 0.035 | 0.775 ± 0.013 | 0.558 ± 0.059 | 0.602 ± 0.049 | 0.113 ± 0.094 | 11, 10, 7 |
| MOON | 0.405 ± 0.006 | 0.702 ± 0.148 | 0.470 ± 0.058 | 0.534 ± 0.027 | 0.185 ± 0.056 | 11, 10, 7 |

## Reading

- **Lost rounds cost every technique.** The cloud applies only 9–13 of its 20 rounds. On SWELL, FedAvg falls from 0.549 (lossless) to 0.416, and its received-model edge score from 0.587 to 0.486.
- **The techniques do not degrade alike, which is why the main comparison is lossless.**
  - FedNova and FedDyn hold best on SWELL (0.510 each, against 0.416 for FedAvg).
  - SCAFFOLD keeps part of its WESAD advantage (0.799) but stays at "always stress" on SWELL (0.405). So does MOON.
  - FedDyn loses most of its lossless WESAD lead (0.876 → 0.775). An edge updates its remembered gradient even when its update is then lost, so edge and server state drift apart; its plugin text states this limitation.
- **The lost-round pattern depends only on the seed**, so every technique lost the same number of rounds per seed.
- **The learning rate matters here.** The first lossy comparison (commit `f62390b`, before the review fixes) ran FedAvg at lr 0.3 and reached 0.579 on SWELL. A likely reason is that lr 0.1 was chosen with all 20 rounds applied, and with about half of them lost it trains too little. This experiment did not test that.

## Files

- `<run_id>/`: `run.json`, `summary.json` and `model.npz` of each of the 18 runs.
- `run_metrics.csv`: per run, the final global macro-F1 per dataset, the lost cloud rounds and the trunk alignment.
- `edge_scores.csv`: round-20 edge macro-F1 per run and model.
- `report.html`: macro-F1 per round by scenario and model.

`events.jsonl` is left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
