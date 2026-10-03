# Server optimizers and asynchrony (new framework, real data)

`experiments/techniques_server.yaml` was run on 2026-10-04 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #150.

```bash
onion_fl run experiments/techniques_server_select.yaml --workers 3   # server rates, validation only
onion_fl run experiments/techniques_server.yaml --workers 3
onion_fl report experiments/techniques_server.yaml --out report.html --metric macro_f1 --by scenario
```

## Setup

- **Data, placement and training.** The same as the drift comparison: SWELL + WESAD, segregated placement (α = 0; the fogs hold 9, 9, 5 and 5 edges), lossless links, 20 rounds of 10 local epochs with SGD at lr 0.1 (FedAvg's rate on validation there), seeds 0–2.
- **What changes is the server side.** Every scenario except the FedBuff-style ones runs on `four_fogs_swell_wesad_lossless`; only the cloud's optimizer changes.
- **FedBuff-style buffering** runs on `four_fogs_swell_wesad_lossless_fedbuff`, a hierarchical adaptation of FedBuff (Nguyen et al., 2022) made of round settings. It is not the paper's protocol: the fogs buffer, and the cloud stays synchronous.
  - each fog closes its round as soon as K updates are in, late ones included (K = 5 of 9 SWELL edges and 3 of 5 WESAD edges, just over half; a design choice, not tuned);
  - an edge still training gets no newer model;
  - a late update joins a later round as its change from the model it trained on, weighted by (1 + staleness)^-0.5;
  - the cloud runs FedAvg, or FedAsync's mixing rule (`fedasync_mix`), which only sees staleness on this topology. That rule is applied once per round to the aggregate; it is not the FedAsync protocol, which updates the global model on every arrival and adds a proximal term at the edge.
- **How the server rates were chosen** (`validation.csv`).
  - `experiments/techniques_server_select.yaml` ran each optimizer over a grid on seed 0, with `evaluation.global.subjects: val`: the final global model, which is what the server optimizer produces, was scored on the validation subjects. The test subjects were not evaluated.
  - Where the best value sat on an edge of its grid, the grid was widened by one step (FedAvgM 0.03, FedYogi 0.3, FedAdagrad 1.0, mixing α 1.0). After that, every choice is inside its grid.
  - WESAD's validation score is 1.0 for most settings: its validation subject is easy to classify. So SWELL decided.
  - The selection ran at commits `e90e698` and `56ffaf9`, which differ only in the selection file. The mixing-rule runs were repeated at `47c482f` under the new name `fedasync_mix`, with identical results.

| Scenario | Topology | Cloud optimizer | Chosen value | Validation macro-F1 |
|---|---|---|---|---|
| FedAvg | lossless | `replace` | — | — |
| FedAvgM | lossless | `fedavgm` | server_lr 0.1 (momentum 0.9) | 0.654 |
| FedAdam | lossless | `fedadam` | server_lr 0.03 | 0.708 |
| FedYogi | lossless | `fedyogi` | server_lr 0.1 | 0.666 |
| FedAdagrad | lossless | `fedadagrad` | server_lr 0.3 | 0.741 |
| FedBuff-style | buffering fogs | `replace` | — | — |
| FedBuff-style + FedAsync mixing | buffering fogs | `fedasync_mix` | α 0.9 (a = 0.5) | 0.714 |

## Identity

| Identity | Value |
|---|---|
| `topology_id` | `ff55fe4d2ce1a807f764bcc5205a8d2d4d43ab3d533163a1d13e173a9a6725fa` (both topologies: it ignores round settings) |
| `data_id` | `06ac5ff0d43080f42ccd968956b1f1e666a651cf8caecfa4ecd4e2b695e1f64e` |
| code | commit `47c482f`, clean tree |
| `config_id` FedAvg | `034a778638910bd3ae1115d3bfbbd0ea1b26bf342aae012732c6843f32ad1dcb` |
| `config_id` FedAvgM | `a9bf50b4c41e1630e16c687eb7c994e251c868fb00c2c628793a829f88eb7af8` |
| `config_id` FedAdam | `57997aa8c1dee4da5aee94a75213ba10d22a3c754d09611454cd59b003ef627c` |
| `config_id` FedYogi | `702eead7c2272cd1c07c6179ed7ac0d496e4c60caf6df2a000077fac8776d6a0` |
| `config_id` FedAdagrad | `29cff4e30da811eeda9473367a4d7e5a79e1e38d9515912766a336b64f190e01` |
| `config_id` FedBuff-style | `90777f5c94c7b17af5ef2f4293848c9d13c2c6521f37e8fee35d143f79450b28` |
| `config_id` FedBuff-style + FedAsync mixing | `a2431ff4ceaf0aadc9c5868b3cf1f8d9c4c793a0d4447f87b3b65f0490e589db` |

## Results (mean ± 95% CI over 3 seeds; `run_metrics.csv`)

- **Global:** the final global model's macro-F1 on the test subjects.
- **Edge:** each edge's macro-F1 on its own held-out rows at round 20, for the model it received and after its local training, weighted by samples.
- **Simulated time:** virtual seconds to finish the 20 rounds. **Buffered:** late updates kept for a later round, per run.

| Scenario | Global SWELL | Global WESAD | Edge received | Edge trained | Simulated time (s) | Buffered |
|---|---|---|---|---|---|---|
| FedAvg | 0.549 ± 0.081 | 0.752 ± 0.005 | 0.587 ± 0.055 | 0.581 ± 0.040 | 21 | 0 |
| FedAvgM | 0.506 ± 0.018 | 0.754 ± 0.014 | 0.549 ± 0.022 | 0.541 ± 0.048 | 21 | 0 |
| FedAdam | 0.612 ± 0.038 | 0.853 ± 0.079 | 0.628 ± 0.043 | 0.614 ± 0.042 | 21 | 0 |
| FedYogi | 0.602 ± 0.041 | 0.875 ± 0.076 | 0.614 ± 0.014 | 0.618 ± 0.016 | 21 | 0 |
| FedAdagrad | 0.627 ± 0.013 | 0.898 ± 0.045 | 0.626 ± 0.032 | 0.649 ± 0.090 | 21 | 0 |
| FedBuff-style | 0.417 ± 0.031 | 0.749 ± 0.045 | 0.576 ± 0.250 | 0.639 ± 0.066 | 12 | 283 |
| FedBuff-style + FedAsync mixing | 0.512 ± 0.186 | 0.751 ± 0.000 | 0.587 ± 0.106 | 0.628 ± 0.142 | 12 | 283 |

No round was lost in any run. FedAvg here is the same configuration as FedAvg in the drift comparison, and its scores match.

## Reading

- **In these runs, the adaptive server optimizers beat FedAvg on WESAD.**
  - FedAdagrad reaches 0.898 ± 0.045, FedYogi 0.875 ± 0.076 and FedAdam 0.853 ± 0.079, against 0.752 ± 0.005. None of the intervals overlaps FedAvg's.
  - On SWELL they score higher (0.602–0.627 against 0.549), but the intervals overlap.
  - A plausible reason, not tested here: their step adapts per coordinate, and with segregated fogs each dataset's adapter is updated by two fogs only.
- **FedAvgM does not help at its chosen rate** (SWELL 0.506, WESAD 0.754).
- **FedBuff-style buffering finishes the 20 rounds in 43% less simulated time** (12 s against 21 s), because each fog stops waiting after K updates. Its final SWELL score is lower (0.417).
  - About 283 updates per run arrive late and join a later round, so the model each edge trains from is often a round or more behind.
  - The comparison is per round, not per unit of time. Within the same 21 s, it would have run more rounds; a time-to-accuracy comparison is not in this experiment.
- **FedAsync's mixing rule at the cloud raises the buffered SWELL mean (0.512), but varies a lot between seeds** (0.579, 0.431 and 0.524). It discounts each round's aggregate by the mean staleness reported by the fogs.
- **Next step.** Three seeds leave the SWELL intervals wide. A time-to-accuracy comparison would show the buffering trade-off better than per-round scores. The canonical FedAsync and FedBuff protocols (the global model updated on arrival) are a possible later extension.

## Reproducibility

The 21 runs were first made at `e7fb7fd`. Two changes followed: keeping the base model of any late update a policy may still keep (no 16-round cap), and renaming `fedasync` to `fedasync_mix`. All 21 runs and the four mixing-rule selection runs were repeated at `47c482f`, on a clean tree. Every final model was identical bit for bit, and so were the final metrics. The cap never applied in this benchmark.

## Files

- `<run_id>/`: `run.json`, `summary.json` and `model.npz` of each of the 21 runs.
- `run_metrics.csv`: per run, the global and edge macro-F1, lost rounds, buffered and dropped late updates, and the simulated time.
- `validation.csv`: the 20 selection runs, with their validation scores on the final global model and their commit.
- `report.html`: macro-F1 per round by scenario.

`events.jsonl` is left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
