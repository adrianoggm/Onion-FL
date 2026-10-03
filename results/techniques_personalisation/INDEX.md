# Personalisation techniques (new framework, real data)

`experiments/techniques_personalisation.yaml` was run on 2026-10-03 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #147.

```bash
onion_fl run experiments/techniques_personalisation.yaml --workers 3
onion_fl report runs --out report.html --metric macro_f1 --by scenario
```

## Setup

- **Topology and placement.** `four_fogs_swell_wesad` with a `mixing` placement at α = 0.5. Each edge holds one subject.
- **Test subjects.**
  - SWELL tests on subjects 1, 2, 12, 18 and 19.
  - WESAD tests on S6, S7 and S10.
  - These are the same in every scenario.
- **Edge validation.** Each edge keeps the last 20% of each class as `local_val` (`local_val_split: class_tail`).
- **Training.** Each scenario runs 20 rounds of 10 local epochs (lr 0.003) with seeds 0–2. The scenarios are:

| Scenario | Sharing | Trainer | What it is |
|---|---|---|---|
| FedAvg | `fedavg` | `standard` | One global model |
| FedPer | `fedper` | `standard` | Heads stay on the edge |
| LG-FedAvg | `lg_fedavg` | `standard` | Adapters and trunk stay on the edge; heads are global |
| Ditto | `fedavg` | `ditto` (λ = 0.1) | Global model plus a personal model tied to it |
| APFL | `fedavg` | `apfl` (α₀ = 0.5, learnt) | Personal model mixed with the global one |
| FedRep | `fedper` | `fedrep` (5 head epochs, 10 body epochs) | Local head trained first, then the shared body |
| FedBABU | `fedavg` | `fedbabu` | Only the body learns; the head stays as initialised |

- **What each edge scores** on its `local_val` rows every 5 rounds:
  - `received`: the model it received;
  - `local`: the model it trained;
  - `personal`: the personal model, for Ditto and APFL only;
  - `finetuned`: the received model after 2 epochs of standard training on its own data.
- **Aggregation of edge scores.** The fogs combine these scores by samples, and the cloud combines the fogs.

## Identity

| Identity | Value |
|---|---|
| `topology_id` | `9033d1cf2c9e348b4067c92173a36fd151b3077cf0c93b5a592d8d188df05f32` |
| `data_id` | `06ac5ff0d43080f42ccd968956b1f1e666a651cf8caecfa4ecd4e2b695e1f64e` |
| code | commit `5c2c121`, clean tree |
| `config_id` FedAvg | `8868d837be8dff1e9583ff8dc746ef57ff4b642be470e75bb7843e08d2ad0818` |
| `config_id` FedPer | `2af4045c65369a7cc85de62e5b4f4e3e034b27627ecd248dae05f393f5d592e9` |
| `config_id` LG-FedAvg | `ae8eef84f611345cdc5ba18d6514907d623faac06b08f5ff3f4a979e67c72d07` |
| `config_id` Ditto | `57d8dbe23d15c2ee74ded0303974932498a0daa05043b0c96a59cb61428173af` |
| `config_id` APFL | `9b03184410886c881d5e4b3a688142e5663f3324ce287af0d2e8ddeef91b6f3a` |
| `config_id` FedRep | `c4940b1747948b788531f74e731727ae73b03018e7d2f86d3bc346f9be692b4e` |
| `config_id` FedBABU | `dc5452530247ed51da7015eca4771b99b071cc764f734a1716fb876da1913ab0` |

## Results (macro-F1, mean ± 95% CI over 3 seeds)

**Global model** on the test subjects, from `summary.json` (the last global evaluation; for seed 1 that is round 19, because its round 20 failed quorum at the cloud):

| Scenario | Global SWELL | Global WESAD |
|---|---|---|
| FedAvg | 0.620 ± 0.012 | 0.746 ± 0.009 |
| FedPer | 0.574 ± 0.052 ¹ | 0.686 ± 0.090 ¹ |
| LG-FedAvg | 0.246 ± 0.008 ¹ | 0.521 ± 0.279 ¹ |
| Ditto | 0.627 ± 0.011 | 0.746 ± 0.009 |
| APFL | 0.625 ± 0.006 | 0.747 ± 0.018 |
| FedRep | 0.502 ± 0.005 ¹ | 0.440 ± 0.556 ¹ |
| FedBABU | 0.610 ± 0.004 | 0.749 ± 0.009 |

¹ These techniques keep groups on the edge, so the global model holds untrained versions of them. Their global columns do not measure the technique; read their edge columns.

**Edges** score their own held-out rows at round 20. Each run's score is the mean over its edges, weighted by samples as the aggregators do. 28 edges score in each run, except 27 in seed 1. The numbers come from the edges' own events, kept in `edge_scores.csv`. The cloud's combined scores can't be used: seed 1 failed quorum at the cloud on every evaluation round, so they never formed. That has since been fixed: a failed round now keeps the scores that arrived.

| Scenario | Received | Trained (local) | Personal | Fine-tuned |
|---|---|---|---|---|
| FedAvg | 0.593 ± 0.014 | 0.641 ± 0.023 | — | 0.607 ± 0.045 |
| FedPer | 0.591 ± 0.009 | 0.632 ± 0.033 | — | 0.611 ± 0.074 |
| LG-FedAvg | 0.683 ± 0.018 | 0.694 ± 0.007 | — | 0.691 ± 0.030 |
| Ditto | 0.600 ± 0.040 | 0.634 ± 0.013 | 0.651 ± 0.004 | 0.609 ± 0.068 |
| APFL | 0.598 ± 0.019 | 0.646 ± 0.024 | 0.695 ± 0.019 | 0.604 ± 0.036 |
| FedRep | 0.585 ± 0.050 | 0.621 ± 0.021 | — | 0.596 ± 0.088 |
| FedBABU | 0.608 ± 0.032 | 0.629 ± 0.018 | — | 0.611 ± 0.060 |

## Reading

- **APFL and LG-FedAvg win on the edges.** APFL's personal model (0.695 ± 0.019) and LG-FedAvg's local model (0.694 ± 0.007) are the only edge scores whose intervals clear FedAvg's own trained model (0.641 ± 0.023).
- **Ditto falls in between.** Its personal model (0.651 ± 0.004) is better than its own trained model, but its interval overlaps FedAvg's.
- **The global model is not hurt.** Ditto, APFL and FedBABU keep it as good as FedAvg's: 0.61–0.63 on SWELL and 0.75 on WESAD. In these runs Ditto's and APFL's personal passes still shared the node's random stream. Since #152 they use a child stream, so a re-run gives them exactly FedAvg's global model.
- **Fine-tuning helps a little, but less than training.** Two epochs on the received model beat the received model in every scenario, and stay below the edge's own trained model in every one.
- **Lost rounds.** The lossy links cost the cloud 7–9 of its 20 rounds per run (`quorum_failed`).
- **Optimistic edge scores.** The held-out rows are the last of each class, so they are close in time to the training rows. That makes every edge score optimistic in the same way. The scores rank the techniques, but don't compare with the global test scores.
- **Next step.** More seeds would tighten the intervals, and subjects with more data than SWELL's 106–130 minutes each would help.

## Files

- `<run_id>/`: `run.json` (identity, config, `run_hash`), `summary.json` and `model.npz` of each run.
- `edge_scores.csv`: round-20 edge macro-F1 per run and model, from the edges' events (the source of the edge table).
- `report.html`: macro-F1 per round by scenario and model, mean ± CI over the seeds.

`events.jsonl` (about 3 MB per run) is left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
