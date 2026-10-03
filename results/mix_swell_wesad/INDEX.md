# SWELL and WESAD mixing sweep (new framework, real data)

`experiments/mix_swell_wesad.yaml` run on 2026-10-03 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows).

```bash
onion_fl run experiments/mix_swell_wesad.yaml --workers 3
onion_fl baseline experiments/mix_swell_wesad.yaml --models lr rf > baselines.json
```

**Setup:**
- **Topology.** `four_fogs_swell_wesad` has four fogs, two with SWELL as their home dataset and two with WESAD. The links are lossy: wifi from fog to cloud and 4g from edge to fog.
- **Placement.** The `mixing` placement sends each dataset to its home fogs at α = 0 and spreads it over all fogs at α = 1.
- **Subjects.** One edge per subject. SWELL tests on subjects 1, 2, 12, 18 and 19, and WESAD on S6, S7 and S10. These are the same in every scenario.
- **Model.** A modular MLP with an adapter per dataset and a shared trunk and head, trained with FedAvg.
- **Training.** 20 rounds of 10 local epochs, seeds 0–2.

| Identity | Value |
|---|---|
| `topology_id` | `9033d1cf2c9e348b4067c92173a36fd151b3077cf0c93b5a592d8d188df05f32` |
| `config_id` α = 0.0 | `48deeaaa456ce6e29a30b39fbc27ee9d5d2ce4c438426d06f76502930eedd946` |
| `config_id` α = 0.5 | `e3d1a2bd94aa35af3b1fc832469be47456467109b60ad8805019f24df0733f7a` |
| `config_id` α = 1.0 | `73dcf56593d1c77ae8def9359740bcfc185ff77c8ba16ebc954b6b3c6ab2f3c6` |
| `data_id` | `06ac5ff0d43080f42ccd968956b1f1e666a651cf8caecfa4ecd4e2b695e1f64e` |
| code | commit `3d4db6f`, clean tree |

## Global model on the test subjects (mean ± 95% CI over 3 seeds)

| α | SWELL accuracy | SWELL macro-F1 | WESAD accuracy | WESAD macro-F1 |
|---|---|---|---|---|
| 0.0 (segregated) | 0.582 ± 0.058 | 0.576 ± 0.048 | 0.775 ± 0.017 | 0.748 ± 0.021 |
| 0.5 | 0.584 ± 0.052 | 0.578 ± 0.043 | 0.777 ± 0.008 | 0.750 ± 0.012 |
| 1.0 (mixed) | 0.592 ± 0.071 | 0.584 ± 0.056 | 0.766 ± 0.030 | 0.736 ± 0.032 |

Centralised baselines on the same test subjects:

| Model | SWELL accuracy | SWELL macro-F1 | WESAD accuracy | WESAD macro-F1 |
|---|---|---|---|---|
| Logistic regression | 0.661 | 0.460 | 0.913 | 0.899 |
| Random forest | 0.640 | 0.612 | 0.779 | 0.735 |

**Reading.**
- **Placement.** It has no measurable effect with three seeds: the intervals overlap at every α.
- **WESAD.** The federated model reaches the random forest's level, but stays well below logistic regression.
- **Lost rounds.** The lossy links cost the cloud 7–11 of its 20 rounds per run, which failed quorum (`summary.json`, `quorum_failed`).
- **Under-training.** One local epoch, as in `mix_ab`, does not learn: every run then predicted "stress" for every sample, WESAD included. The experiment's comment records this.

## Files

- `<run_id>/`: `run.json` (identity, config, `run_hash`), `summary.json` and `model.npz` of each run.
- `report.html`: accuracy per round by scenario, mean ± CI over seeds.
- `baselines.json`: the centralised baselines.

`events.jsonl` (about 3 MB per run) is left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies; it passed on the original run folders.
