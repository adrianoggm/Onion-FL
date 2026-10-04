# Continuing a trained model with new data (new framework, real data)

`experiments/continuum_parent.yaml`, `continuum_warm_start.yaml` and `continuum_exactness.yaml` were run on 2026-10-04 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #157 (continuum C2).

```bash
onion_fl run experiments/continuum_parent.yaml --workers 3        # SWELL only
onion_fl run experiments/continuum_warm_start.yaml --workers 3    # SWELL + WESAD
onion_fl run experiments/continuum_exactness.yaml --workers 3
```

## Setup

- **Parent.** A federation trained on SWELL only for 20 rounds, on `two_fogs_swell_lossless`: the two SWELL fogs of the four-fog topology, without loss. Each run leaves its model bundle in `runs/<run_id>/bundle/`.
- **Continuation ("warm start").** SWELL + WESAD on `four_fogs_swell_wesad_lossless` (segregated placement), with `learning.init: {name: run, run: "experiment:continuum_parent"}`. Each seed continues the parent run of the same seed. The continuation:
  - verifies the parent's `run_hash`;
  - keeps SWELL's frozen preprocessing (re-applied bit for bit) and fits WESAD's;
  - keeps the global model, the server state and each SWELL edge's model and stream;
  - builds WESAD's adapter fresh and starts the two WESAD fogs fresh;
  - trains rounds 21–40 and records the parent in `run.json`.
- **Variant.** The same with the SWELL adapter frozen (`frozen: [adapter.swell]`). This is a design choice, not tuned.
- **Baselines.** SWELL + WESAD from scratch for 20 rounds (the rounds the continuation adds) and for 40 rounds (parent and continuation together).
- **Training.** FedAvg with SGD at lr 0.1, 10 local epochs, seeds 0–2. lr 0.1 is FedAvg's rate on validation in the lossless drift comparison, and the same configuration as its FedAvg.
- **Same test subjects everywhere.** The parent and every child share `data.roles`: the continuation is refused if a child's test or validation subject trained the parent.
- **Exactness check.** `continuum_exactness.yaml` continues each "from scratch, 20 rounds" run for 20 more rounds, to compare it with the "from scratch, 40 rounds" run of the same seed.

| Identity | Value |
|---|---|
| `topology_id` parent | `6de5106032a2c0c70e3e0afe3619a02eaa1df7c77f2c3fdb0d8a3ffab7b11ab2` |
| `topology_id` children | `ff55fe4d2ce1a807f764bcc5205a8d2d4d43ab3d533163a1d13e173a9a6725fa` |
| `data_id` parent (SWELL) | `c1bce4b610e309d484f249852bf2324de082161b59ab908860b4873a4f853db1` |
| `data_id` children (SWELL + WESAD) | `06ac5ff0d43080f42ccd968956b1f1e666a651cf8caecfa4ecd4e2b695e1f64e` |
| code | commit `63b0e47` (parent, warm start); `915bbe6` (exactness, differing only by its experiment file); clean tree |
| `config_id` parent | `d7e3dff389f03537e39cad0a91eea6129a2aeaab262890b9da821c1bba8e57c3` |
| `config_id` from scratch, 20 rounds | `a1de5d60275ef923fa1428a08b7b447c6ab22368174e7404e5ea6bca2ba917d6` |
| `config_id` from scratch, 40 rounds | `0f6daf3c41e7af432e375e57cdf90e27667212d8f8bd2f628331d6da9e47819e` |
| `config_id` warm start | `0d59969ed6a6a4a55541278f76d55b5fab5620e49d350d38e79913111fa58868` |
| `config_id` warm start, SWELL adapter frozen | `25f415a233e4f4064d19d5f52291a01d685e271b43e6615561f93c4c8ba5636d` |
| `config_id` exactness | `9fc5a928980471579e2572dc370a65ffba399bd980e581704145f60b9f0a14eb` |

## Results (global macro-F1 on the test subjects, mean ± 95% CI over 3 seeds; `run_metrics.csv`)

- **WESAD after 5 rounds:** the score 5 rounds after WESAD joined (round 25 for a continuation, round 5 from scratch).
- **SWELL change from parent:** the continuation's final SWELL score minus its parent's.

| Scenario | Rounds | Global SWELL | Global WESAD | WESAD after 5 rounds | SWELL change from parent |
|---|---|---|---|---|---|
| Parent (SWELL only) | 1–20 | 0.464 ± 0.137 | — | — | — |
| Warm start | 21–40 | 0.599 ± 0.041 | 0.751 ± 0.000 | 0.721 ± 0.068 | +0.135 ± 0.118 |
| Warm start, SWELL adapter frozen | 21–40 | 0.561 ± 0.060 | 0.760 ± 0.024 | 0.722 ± 0.060 | +0.097 ± 0.078 |
| From scratch, 20 rounds | 1–20 | 0.549 ± 0.081 | 0.752 ± 0.005 | 0.326 ± 0.141 | — |
| From scratch, 40 rounds | 1–40 | 0.608 ± 0.058 | 0.769 ± 0.023 | 0.326 ± 0.141 | — |

**Exactness** (`exactness.csv`): in all three seeds, continuing a 20-round run for 20 more rounds gives the same final model, bit for bit, as running 40 rounds without stopping.

## Reading

- **A continuation does not forget SWELL in these runs; it improves it.** The SWELL score rises from the parent's 0.464 to 0.599 (+0.135 ± 0.118) while WESAD is added, ending close to 40 rounds from scratch (0.608).
- **WESAD is learnt much faster from a trained trunk.**
  - Five rounds after it joins, WESAD reaches 0.721 ± 0.068, against 0.326 ± 0.141 from scratch.
  - Its final score (0.751) matches the baselines'.
- **Freezing the SWELL adapter does not help here.** SWELL ends lower (0.561) and WESAD about the same. Its intervals overlap the full continuation's.
- **The parent alone is weaker on SWELL (0.464) than SWELL + WESAD together (0.549 at 20 rounds),** so in these runs the WESAD data helps the shared trunk.
- **Continuing is exact** on synchronous, lossless runs: the bundle holds everything the next round needs (spec §6 gives the scope).
- **Next step.** Three seeds leave the SWELL intervals wide. Streams (C3) will make the continuation incremental in time instead of one block of new data.

## Files

- `<run_id>/`: `run.json`, `summary.json` and `model.npz` of the 18 runs (3 parents, 12 children, 3 exactness checks). The bundles stay in the original run folders.
- `run_metrics.csv`: per run, the scenario, seed, parent, rounds, global macro-F1 per dataset, WESAD after 5 rounds and the SWELL change from the parent.
- `exactness.csv`: per seed, the 40-round run, the 20-round parent, its continuation and whether the final models are identical.
- `report.html`: macro-F1 per round by scenario (parent and warm-start runs).

`events.jsonl` and `bundle/` are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
