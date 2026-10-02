# SWELL reference run (new framework, real data)

`experiments/swell_reference.yaml` run on 2026-10-03 with the SWELL-KW feature dataset from DANS (doi:10.17026/dans-x55-69zp).

```bash
onion_fl run experiments/swell_reference.yaml --workers 4
onion_fl baseline experiments/swell_reference.yaml --models lr rf > baselines.json
```

The setup has three fogs and 18 training subjects. It uses physiology only (HR, RMSSD, SCL per minute), and 999 is read as missing. The test subjects are 19, 20, 23, 24 and 25 (646 minutes, 67.3% labelled stress). The run trains for 12 rounds of 14 local epochs over seeds 0–9, in simulation.

| Identity | Value |
|---|---|
| `topology_id` | `38ee627b304aa62b196ccc8ebe1474f2eeac815b1a008b635a357ad0a5d9eb33` |
| `config_id` | `dbf79b9114ab49086d368a69ac598bbae3370fcc5bc1f0d52ae6da5e27c240ae` |
| `data_id` | `4cb6e8304363b781637fb153b1b4aedcc3fea14c5e698827a74767e69397bb97` |
| code | commit `1696c05`, clean tree |

## Results on the test subjects

| Model | Accuracy | Macro-F1 | Balanced accuracy |
|---|---|---|---|
| Federated MLP (10 seeds, mean ± 95% CI) | 0.669 ± 0.009 | 0.489 ± 0.011 | — |
| Centralised logistic regression | 0.689 | 0.463 | 0.527 |
| Centralised random forest | 0.613 | 0.495 | 0.506 |
| Always "stress" | 0.673 | 0.402 | 0.500 |

No model beats the majority class on this split. `onion_fl baseline --cv 5` gives the random forest 0.547 ± 0.040 balanced accuracy over subject-level folds, so the fixed split is among the harder ones. Per-minute physiology barely separates stress across unseen subjects, and HR and RMSSD are missing in 53% of the minutes.

## Files

- `<run_id>/`: `run.json` (identity, config, `run_hash`), `summary.json` and `model.npz` of each seed.
- `report.html`: accuracy per round, mean ± CI over the seeds.
- `baselines.json`: the centralised baselines on the same split.

`events.jsonl` (1.2 MB per run) is left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. Re-running the command on commit `1696c05` with the same data gives the same metrics per seed. Two runs on 2026-10-03, at commits `3c3844b` and `1696c05`, gave identical metrics for every seed.
