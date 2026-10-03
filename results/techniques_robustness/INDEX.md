# Robust aggregation under attack (new framework, real data)

`experiments/techniques_robustness.yaml` was run on 2026-10-03 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #149.

```bash
onion_fl run experiments/techniques_robustness.yaml --workers 3
onion_fl report experiments/techniques_robustness.yaml --by scenario
```

## Setup

- **Topology and placement.** `four_fogs_swell_wesad` with a `mixing` placement at α = 0.5. Each edge holds one subject.
- **Aggregators.** Every fog (a leaf aggregator) runs the scenario's aggregator; the cloud runs FedAvg.
- **Training.** Training is as in the drift comparison: SGD, lr 0.3 (chosen on validation for FedAvg), 10 local epochs, 20 rounds, seeds 0–2.
- **Attacks.** In each dataset, 20% of the edges (picked with the run seed) flip the sign of their update: they send x − s·(y − x). At s = 1 the honest majority still dominates the mean. At s = 5 the averaged update, 0.8Δ − 0.2·5Δ, points against learning; s was set by that arithmetic, not tuned on test scores.
- **Clipping bound.** `norm_clip` uses bound 1.0, the median edge update norm under FedAvg without attack (seed 0: 464 updates, median 1.02, interquartile range 0.82–1.34). It is a training statistic, not a test score.
- **Code.** The runs without attack and with s = 1 ran at commit `5c91004`; those with s = 5 ran at commit `e346733`. The two commits differ only in the experiment file and in an error for unknown scenario names, so training is identical. All 72 runs were on a clean tree.

`topology_id` `9033d1cf2c9e348b4067c92173a36fd151b3077cf0c93b5a592d8d188df05f32`, `data_id` `06ac5ff0d43080f42ccd968956b1f1e666a651cf8caecfa4ecd4e2b695e1f64e`. Each run's `config_id` is in its `run.json`.

## Results (global macro-F1, mean ± 95% CI over 3 seeds; `run_metrics.csv`)

| Fog aggregator | SWELL, no attack | WESAD, no attack | SWELL, s = 1 | WESAD, s = 1 | SWELL, s = 5 | WESAD, s = 5 |
|---|---|---|---|---|---|---|
| FedAvg | 0.568 ± 0.092 | 0.751 ± 0.000 | 0.540 ± 0.119 | 0.749 ± 0.198 | 0.244 ± 0.000 | 0.392 ± 0.000 |
| Median | 0.578 ± 0.020 | 0.741 ± 0.043 | 0.517 ± 0.036 | 0.740 ± 0.270 | 0.541 ± 0.113 | 0.500 ± 0.915 |
| Trimmed mean (β = 0.1) | 0.577 ± 0.103 | 0.751 ± 0.000 | 0.533 ± 0.141 | 0.747 ± 0.177 | 0.244 ± 0.000 | 0.389 ± 0.013 |
| Krum | 0.556 ± 0.108 | 0.617 ± 0.575 | 0.569 ± 0.079 | 0.436 ± 0.677 | 0.548 ± 0.066 | 0.369 ± 0.468 |
| Multi-Krum | 0.577 ± 0.076 | 0.759 ± 0.109 | 0.566 ± 0.143 | 0.780 ± 0.304 | 0.489 ± 0.143 | 0.660 ± 0.772 |
| Geometric median | 0.567 ± 0.116 | 0.747 ± 0.018 | 0.560 ± 0.149 | 0.769 ± 0.292 | 0.555 ± 0.135 | 0.647 ± 0.541 |
| Bulyan | 0.583 ± 0.044 | 0.738 ± 0.026 | 0.523 ± 0.073 | 0.767 ± 0.189 | 0.453 ± 0.215 | 0.296 ± 0.178 |
| Norm clip (1.0) | 0.573 ± 0.069 | 0.747 ± 0.018 | 0.548 ± 0.131 | 0.761 ± 0.300 | 0.548 ± 0.134 | 0.763 ± 0.324 |

Detection by the aggregators that select. Malicious updates dropped / malicious updates present is recall; malicious dropped / all dropped is precision. Both are summed over fogs, rounds and seeds, from the `aggregation.dropped` events in `run_metrics.csv`:

| Aggregator | Recall, s = 1 | Precision, s = 1 | Recall, s = 5 | Precision, s = 5 |
|---|---|---|---|---|
| Krum | 0.927 | 0.231 | 1.000 | 0.250 |
| Multi-Krum | 0.278 | 0.418 | 0.546 | 0.822 |
| Bulyan | 0.284 | 0.416 | 0.567 | 0.829 |

## Reading

- **The weak attack (s = 1) barely hurts FedAvg** (SWELL 0.568 → 0.540, WESAD unchanged), so it does not separate the defences.
- **The strong attack (s = 5) collapses FedAvg and the trimmed mean.** Both fall to SWELL 0.244 in every seed and to about 0.39 on WESAD. The trimmed mean drops 10% per side, fewer than the 20% attackers.
- **Norm clipping withstands the strong attack best.** It keeps the no-attack scores, 0.548 and 0.763. Clipped to the honest median norm, a ×5 reversed update weighs no more than an honest one, so the 80% honest majority still wins.
- **The geometric median holds on SWELL (0.555) but is unstable on WESAD.** The coordinate median is also unstable on WESAD; its 0.500 ± 0.915 means one seed collapsed.
- **Selection detects the strong attackers.**
  - Multi-Krum and Bulyan drop about 55% of the malicious updates, with 82% of what they drop being malicious. The weak attackers are much harder to tell apart (recall about 28%).
  - Krum keeps a single update per fog. It catches every attacker, but also discards most honest updates, and it hurts WESAD even without an attack (0.617 ± 0.575).
  - Bulyan collapses on WESAD under the strong attack (0.296).
- **Next step.** With three seeds several intervals are wide. The WESAD columns depend on how few WESAD edges each fog holds at α = 0.5.

## Files

- `<run_id>/`: `run.json`, `summary.json` and `model.npz` of each of the 72 runs.
- `run_metrics.csv`: per run, the global macro-F1 per dataset, `quorum_failed`, and dropped / malicious-dropped / malicious-present counts and clipped counts at the fogs.
- `report.html`: macro-F1 per round by scenario.

`events.jsonl` is left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
