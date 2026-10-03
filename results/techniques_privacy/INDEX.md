# Differential privacy (new framework, real data)

`experiments/techniques_privacy.yaml` was run on 2026-10-03 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #149.

```bash
onion_fl run experiments/techniques_privacy.yaml --workers 3
onion_fl report experiments/techniques_privacy.yaml --by scenario
```

## Setup

- **Topology, placement and training.** The same as the robustness comparison: four fogs, α = 0.5, SGD with lr 0.3, 20 rounds, seeds 0–2. No attack.
- **Central DP (`dp_fedavg` at every fog).**
  - The fog clips each child's update to C = 1.0, averages them uniformly and adds N(0, (σ·C/m)²) per coordinate, where m is the number of holders of the key.
  - The fog is trusted with its children's raw updates.
- **Local DP (`local_dp` at every edge).**
  - Each edge clips its own update to C = 1.0 and adds N(0, (σ·C)²) before sending.
  - No aggregator is trusted.
- **Clipping bound.** C = 1.0 is the median edge update norm under FedAvg (`results/techniques_robustness/INDEX.md`). It is a training statistic.
- **How ε is computed.**
  - ε is for δ = 1e-5, from an RDP accountant for the Gaussian mechanism composed over the rounds actually applied.
  - It has no amplification by subsampling, so it is a conservative upper bound.
  - Central ε counts the aggregations a fog performed; local ε counts the updates an edge released. The table gives the largest final value per run, averaged over seeds.
- **Code.** Commit `5c91004`, clean tree.

`topology_id` `9033d1cf2c9e348b4067c92173a36fd151b3077cf0c93b5a592d8d188df05f32`, `data_id` `06ac5ff0d43080f42ccd968956b1f1e666a651cf8caecfa4ecd4e2b695e1f64e`.

| Scenario | `config_id` |
|---|---|
| FedAvg | `0e56a9a7e6ede0db050b96174eb207e7eb775752ae162cd461062e11eddc1624` |
| Central DP, σ = 0.5 | `dd6ce16ad894e5b408058a7486fb5ebec2805c23828852a15d323e0d7c92b03e` |
| Central DP, σ = 1.0 | `9f6ce4dc4db04d799abdac74e853b9637cf15050094030f0699f864f31c85940` |
| Local DP, σ = 0.5 | `11ee80c7cc93a033d75b04005e4f622b59441b6c312d6ab2f7d39c977a9f704a` |
| Local DP, σ = 1.0 | `caae51f49c60ce9ebf606cf4c2e0149e53d6f77ddc5a1b1c2d3d78b177a7fda5` |

## Results (global macro-F1, mean ± 95% CI over 3 seeds; `run_metrics.csv`)

| Scenario | SWELL | WESAD | ε after 20 rounds (δ = 1e-5) |
|---|---|---|---|
| FedAvg (no DP) | 0.568 ± 0.092 | 0.751 ± 0.000 | — |
| Central DP, σ = 0.5 | 0.544 ± 0.048 | 0.529 ± 0.318 | 76.7 |
| Central DP, σ = 1.0 | 0.466 ± 0.122 | 0.282 ± 0.048 | 28.6 |
| Local DP, σ = 0.5 | 0.420 ± 0.037 | 0.356 ± 0.227 | 82.9 |
| Local DP, σ = 1.0 | 0.437 ± 0.133 | 0.345 ± 0.365 | 31.5 |

## Reading

- **Privacy costs a lot of accuracy here, and buys little formal protection.**
  - At σ = 1, ε is still about 29–31 after 20 rounds, and WESAD falls from 0.751 to about 0.28–0.35.
  - These ε are upper bounds: without subsampling amplification, every round counts in full.
- **Central DP keeps more utility than local DP at the same σ.** Central DP's noise is divided by the number of children it averages, while each local edge adds its own full noise. Central DP at σ = 0.5 keeps SWELL near FedAvg (0.544).
- **Central ε is lower than local ε at the same σ** (28.6 against 31.5) because fogs that failed quorum did not aggregate, so they applied fewer rounds.
- **WESAD suffers most.** Each fog holds few WESAD edges, so the noise per WESAD key is relatively larger.
- **Next step.** Useful privacy at this data size would need fewer rounds, larger cohorts or subsampled participation, which the accountant does not yet credit.

## Files

- `<run_id>/`: `run.json`, `summary.json` and `model.npz` of each of the 15 runs.
- `run_metrics.csv`: per run, the global macro-F1 per dataset, `quorum_failed`, and the final central and local ε.
- `report.html`: macro-F1 per round by scenario.

`events.jsonl` is left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
