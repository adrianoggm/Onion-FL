# Differential privacy (new framework, real data)

`experiments/techniques_privacy.yaml` was run on 2026-10-03 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #149.

```bash
onion_fl run experiments/techniques_privacy.yaml --workers 3
onion_fl report experiments/techniques_privacy.yaml --out report.html --metric macro_f1 --by scenario
```

## Setup

- **Topology, placement and training.** The same as the robustness comparison: four fogs over lossless links, α = 0.5, SGD with lr 0.3, 20 rounds, seeds 0–2. No attack.
- **Central DP (`dp_fedavg` at every fog).**
  - The fog clips each child's update to C = 1.0, averages the updates uniformly and adds N(0, (σ·C/m)²) per coordinate, where m is the number of holders of the key.
  - The fog is trusted with its children's raw updates.
- **Local DP (`local_dp` at every edge).**
  - Each edge clips its own update to C = 1.0 and adds N(0, (σ·C)²) before sending.
  - No aggregator is trusted.
- **Clipping bound.** C = 1.0 is the median edge update norm under FedAvg (`../techniques_robustness/INDEX.md`). It is a training statistic.
- **How ε is computed.**
  - ε is for δ = 1e-5, from an RDP accountant for the Gaussian mechanism composed over the rounds actually applied.
  - It has no amplification by subsampling, so it is a conservative upper bound.
  - Central DP uses add/remove-one-child adjacency, with sensitivity C on the sum. Local DP uses replace-one adjacency: any update in the C-ball may become any other, so the sensitivity is 2C and the accountant uses σ/2.
  - Central ε counts the aggregations a fog performed; local ε counts the updates an edge released. The table gives the largest final value per run, averaged over seeds.
- **Code.** Commit `5995082`, clean tree. See "Reproducibility" below for the rerun at the merge of #148's fixes (`0a4a6ad`).

`topology_id` `ff55fe4d2ce1a807f764bcc5205a8d2d4d43ab3d533163a1d13e173a9a6725fa`, `data_id` `06ac5ff0d43080f42ccd968956b1f1e666a651cf8caecfa4ecd4e2b695e1f64e`. Each run's `config_id` is in its `run.json`.

## Results (global macro-F1, mean ± 95% CI over 3 seeds; `run_metrics.csv`)

| Scenario | SWELL | WESAD | ε after 20 rounds (δ = 1e-5) | Lost cloud rounds (seeds 0, 1, 2) |
|---|---|---|---|---|
| FedAvg (no DP) | 0.578 ± 0.109 | 0.751 ± 0.000 | — | 0, 0, 0 |
| Central DP, σ = 0.5 | 0.582 ± 0.074 | 0.629 ± 0.147 | 82.9 | 0, 0, 0 |
| Central DP, σ = 1.0 | 0.449 ± 0.120 | 0.311 ± 0.084 | 30.8 | 10, 8, 9 |
| Local DP, σ = 0.5 | 0.480 ± 0.150 | 0.317 ± 0.080 | 245.9 | 11, 9, 9 |
| Local DP, σ = 1.0 | 0.424 ± 0.072 | 0.391 ± 0.314 | 82.9 | 17, 18, 18 |

The links lose nothing. Rounds are lost when the noise drives some edges to non-finite weights: they roll back and send an empty update, and a fog with quorum 1.0 fails the round. Central ε at σ = 1.0 averages 30.4, 31.5 and 30.4 over the seeds: in seeds 0 and 2, no fog aggregated in all 20 rounds.

## Reading

- **In this configuration, stronger privacy costs a lot of utility, and local DP costs much more than central DP.**
  - Central DP at σ = 0.5 keeps SWELL (0.582 against 0.578 without DP) and costs part of WESAD (0.629 against 0.751).
  - Local DP at the same σ falls to 0.480 and 0.317, with a much larger ε (245.9 against 82.9). Each edge adds its full noise to its own update, and the replace-one accounting counts that noise as σ/2.
- **The accumulated budget is high at every level tested** (ε of 30.8–245.9 over 20 rounds). These are upper bounds without subsampling amplification: every round counts in full.
- **At the higher noise levels, training itself breaks down.** Edges diverge and the cloud loses up to 18 of 20 rounds (local DP, σ = 1.0). Part of the utility loss there comes from rounds that never happened.
- **WESAD suffers most.** Each fog holds few WESAD edges, so the noise per WESAD key is relatively larger.
- **Next step.** Useful privacy at this data size would need fewer rounds, larger cohorts or subsampled participation, which the accountant does not yet credit.

## Reproducibility

The runs are being repeated at the merge commit `0a4a6ad`, which brings #148's fixes into this branch, on a clean tree. The first 8 of 15 give the same final model bit for bit, with the same final metrics and lost cloud rounds.

## Files

- `<run_id>/`: `run.json`, `summary.json` and `model.npz` of each of the 15 runs.
- `run_metrics.csv`: per run, the global macro-F1 per dataset, `quorum_failed`, and the final central and local ε.
- `report.html`: macro-F1 per round by scenario.

`events.jsonl` is left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
