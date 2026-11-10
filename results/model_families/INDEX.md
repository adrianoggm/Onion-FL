# Two model families: modular MLP against a feature transformer (real data)

`experiments/model_families.yaml` was run on 2026-11-10 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #229; the design of the families is in `docs/superpowers/specs/2026-11-10-onion-fl-model-families-design.md`.

```bash
onion_fl run experiments/model_families_select.yaml --workers 3   # learning rates, validation only
onion_fl run experiments/model_families.yaml --workers 3
onion_fl report experiments/model_families.yaml --out report.html --metric macro_f1 --by scenario
```

## Setup

- **Data, placement and training.** These are FedAvg's settings in `techniques_server`: SWELL + WESAD, a segregated placement (α = 0; the fogs hold 9, 9, 5 and 5 edges), lossless links, FedAvg with `replace` at the cloud, and 20 rounds of 10 local epochs with SGD. Seeds 0–2.
- **Only the model family changes.**

| Family | Options | Parameters (global model) | of which trunk |
|---|---|---|---|
| `modular_mlp` | adapter 64, trunk [64, 32], dropout 0.2 | 9,378 | 6,240 |
| `feature_transformer` | token_dim 16, 1 block, 2 heads, ffn ×2, dropout 0.1 | 3,778 | 2,272 |

  Both families use the same keys (`adapter.swell`, `adapter.wesad`, `trunk`, `head.stress_binary`) and the same sharing (`fedavg`: everything global).
- **How the learning rates were chosen** (`validation.csv`).
  - `experiments/model_families_select.yaml` ran each family at SGD lr 0.03, 0.1 and 0.3 on seed 0, with `evaluation.global.subjects: val`. The final global model was scored on the validation subjects; the test subjects were not evaluated.
  - The transformer scored best at 0.3, the edge of its grid, so its grid was widened to 1.0. At 1.0 it diverged: 19 of the 20 rounds failed and there is no score.
  - The choices are **0.1 for the MLP** (validation macro-F1 0.634) and **0.3 for the transformer** (0.615).
  - WESAD's validation subject is easy: the MLP scores 1.0 on it at 0.1 and 0.3. SWELL decided.
  - The selection ran at commit `2c09e77` on a clean tree. A first pass at `a0d2d28`/`9b68998` had uncommitted files in the tree (a draft of `model_families.yaml` and the spec), but the code was the same. Its six shared runs gave the same scores, and it is not kept.

## Identity

| Identity | Value |
|---|---|
| `topology_id` | `ff55fe4d2ce1a807f764bcc5205a8d2d4d43ab3d533163a1d13e173a9a6725fa` |
| `data_id` | `9fd573754157d68b111f037ecd891bdb62abcbff1d5e5e818da789dde470257c` |
| code | commit `5b658b0`, clean tree |
| `config_id` modular_mlp | `f81c7216559409be29329742a2c9529335513648159f75aa23480c4525411c9f` |
| `config_id` feature_transformer | `ab0cab7d425ed4ab82a247c8ef1b18b9c137eeb6e0c4bcaf861d4c1eeba592f5` |

| `run_id` | Family | Seed | `run_hash` |
|---|---|---|---|
| `20261110T102235Z-4e37f42d4a6b` | modular_mlp | 0 | `0176760cf1fb4daec5a0e063fa30f0ac0b360192e37e2449e3574a28b0fd4249` |
| `20261110T102235Z-b2c7de058c07` | modular_mlp | 1 | `2082dc8df76de055425fc862065f81bc11f01c704caaf438eb97937f4b0ec76c` |
| `20261110T102235Z-3df2c91af2ed` | modular_mlp | 2 | `38be036d3bd021a0dad0e660d9c1e5cd685bc71ba4d6853b2fd6e8cdfa6cca7e` |
| `20261110T102435Z-3d53201b1d99` | feature_transformer | 0 | `358ad436c8111eef793d2334eca84b0634ec8dc39cf9303dc9fb068b20e87fcb` |
| `20261110T102440Z-3604cb58f2ad` | feature_transformer | 1 | `45e73e52c7e096e8b0a2270355c3e583ffa8029fff109ae8b80172a6c3b3c2f4` |
| `20261110T102440Z-499bfe59bc52` | feature_transformer | 2 | `d9942cc753ae112bcce285b448440e4459b926efaf2149c2f55ba29a76f0e240` |

## Results (mean ± 95% CI over 3 seeds, Student t; `run_metrics.csv`)

- **Global:** the final global model's macro-F1 on the test subjects.
- **Edge:** each edge's macro-F1 on its own held-out rows at round 20, for the model it received and after its local training, weighted by samples.
- **Traffic** is the bytes sent per run. **Wall time** is the time on one CPU worker, with three runs in parallel.

| Family | Global SWELL | Global WESAD | Edge received | Edge trained | Traffic (MB) | Wall time (s) |
|---|---|---|---|---|---|---|
| `modular_mlp` | 0.549 ± 0.081 | 0.752 ± 0.005 | 0.587 ± 0.055 | 0.581 ± 0.040 | 261 | 122 |
| `feature_transformer` | 0.534 ± 0.240 | 0.752 ± 0.014 | 0.518 ± 0.133 | 0.543 ± 0.066 | 105 | 325 |

Per seed (0, 1, 2), global SWELL: MLP 0.563, 0.572, 0.512; transformer 0.613, 0.426, 0.562. No round was lost in any run, and `verify_run` passed on the six original run folders.

The MLP rows match FedAvg in `results/techniques_server` digit for digit (same configuration), although the `data_id` changed in between.

## Reading

- **On this benchmark, the two families cannot be told apart.**
  - The global scores tie: WESAD 0.752 against 0.752, and SWELL 0.549 against 0.534, with intervals that overlap widely.
  - The transformer's SWELL interval (± 0.240) is three times the MLP's: across seeds it ranges from 0.426 to 0.613.
  - Three seeds cannot settle a difference this small.
- **The transformer is worse at the edges in these runs** (received 0.518 against 0.587, trained 0.543 against 0.581). The intervals overlap, so this is not an established difference.
- **The transformer sends 60% fewer bytes** (105 MB against 261 MB per run), because it has 40% of the MLP's parameters. Its trunk is shared as it is, whatever the number of features of each dataset.
- **It costs 2.7 times as much compute** (325 s against 122 s on this CPU), because attention runs over 17 or 31 tokens per row.
- **Neither model reaches a strong WESAD score.** With a segregated placement, WESAD is trained by two fogs only. 0.752 is the same plateau as FedAvg in `techniques_server`, where adaptive server optimizers lifted it to 0.85–0.90.
- **What this experiment does not test:**
  - a larger transformer (more blocks, a wider token_dim);
  - another optimizer (Adam is the usual choice for transformers; both families use SGD here, to keep the comparison the same);
  - an adaptive server optimizer;
  - mixed placements.
  The learning rate is the only setting that was tuned per family.

## Files

- `<run_id>/`: `run.json`, `summary.json` and `model.npz` of each of the six runs.
- `run_metrics.csv`: per run, the global and edge macro-F1, traffic, wall time, lost rounds, `config_id`, `run_hash` and commit.
- `validation.csv`: the seven selection runs, with their validation scores, `config_id`, `run_hash` and commit.
- `report.html`: macro-F1 per round by scenario.

`events.jsonl` and `bundle/` are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
