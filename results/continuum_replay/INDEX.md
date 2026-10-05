# Replay against the recent rows only, on the SWELL and WESAD streams (new framework, real data)

`experiments/stream_replay_swell_wesad.yaml` was run on 2026-10-05 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #159 (continuum C4).

```bash
onion_fl run experiments/stream_replay_swell_wesad.yaml --workers 3   # run twice: the second is the determinism check
```

This sets up the comparison between fine-tuning on the recent rows only and fine-tuning plus replay, for the benchmark (C8). Neither the memory's capacity nor its share of training is tuned.

## Setup

- **Stream, labels and training.** As in [continuum_stream/](../continuum_stream/INDEX.md) with every row labelled:
  - SWELL + WESAD replayed in time order after a 20-minute bootstrap;
  - a round every 10 minutes of data, 18 in all, until the last label arrives;
  - labels 10 minutes late;
  - FedAvg with SGD at lr 0.1 and 10 local epochs, on the segregated lossless topology `four_fogs_swell_wesad_lossless`.
- **Two scenarios:**
  - **Recent only** (`continual.memory: none`): each round trains on the rows it has not used yet.
  - **Replay** (`reservoir`, 512 rows, `replay_ratio: 0.5`): each round also trains on a sample of rows the edge already trained on, as many as its new rows. A row enters the memory only after its first successful training, and replay never makes an idle edge train.
- **Scores** are those of the stream comparison:
  - `prequential` scores every arrival with the model the edge was serving;
  - the test macro-F1 is pooled from the summed confusion matrices of the test subjects.
- **Seeds** 0–2.

| Identity | Value |
|---|---|
| `topology_id` | `ff55fe4d2ce1a807f764bcc5205a8d2d4d43ab3d533163a1d13e173a9a6725fa` |
| `data_id` | `f79c1afad836591c82f72c11c6409118eeeb9e60c814b60214f166762d92b2c0` |
| code | commit `0ca028e`, clean tree |
| `config_id`, recent only | `c77b4c88a0f89687b6c29e59f702d8bee83eb47a51de026375031c7fcb2a60a9` |
| `config_id`, replay | `127bb8bba6c1eaf1a1924bc64e5a3664784b840136a82c4e35d13c871d566ec3` |

## Results (macro-F1, mean ± 95% CI over 3 seeds; `run_metrics.csv`)

| Scenario | Prequential SWELL | Prequential WESAD | Test SWELL | Test WESAD | Failed trainings (per seed) |
|---|---|---|---|---|---|
| Recent only | 0.783 ± 0.045 | 0.000 ± 0.000 | 0.404 ± 0.000 | 0.262 ± 0.000 | 0, 0, 0 |
| Replay | 0.633 ± 0.031 | 0.000 ± 0.000 | 0.354 ± 0.237 | 0.387 ± 0.304 | 1, 73, 3 |

**Test predictions** (`summary.json`, confusion matrices):

| Scenario | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Recent only | stress for every row | stress for every row | stress for every row |
| Replay | no stress for every row | almost all stress; 29 of 111 WESAD baseline windows right | stress for every row |

**Memory** (seed 0, end of the run):
- SWELL edges keep 2222 rows in all. Each edge consumes about 125 rows, so the capacity of 512 never binds: the reservoir holds every row it was offered.
- WESAD edges keep all 572 of their rows.

**Checks:**
- **Determinism** (`determinism.csv`): each scenario and seed, run twice, gives the same final model and the same events.
- **Recent only reproduces C3:** its final models equal those of the full-labels stream runs in [continuum_stream/](../continuum_stream/INDEX.md), bit for bit.

## Reading

- **Replay does not help in this setup; it lowers the prequential score.**
  - On SWELL, the score falls from 0.783 to 0.633. After a condition change, the remembered rows of the old condition hold the old prediction for longer: per round, recent only recovers by round 9 and replay by round 12 (`prequential.csv`).
  - Since SWELL's prequential score mostly measures persistence, slower switching costs it directly. This is the stability–plasticity trade-off, not a broken memory.
- **WESAD stays at 0 either way.** After the bootstrap every WESAD window is stress, and no served model predicts it before the sessions end.
- **The final models still collapse.** With replay they flip between seeds, from all stress to no stress. Two of the three seeds remain constant predictors, so the test intervals are wide and say nothing yet.
- **Replay makes some SWELL edges diverge.** At lr 0.1, training on twice the rows sometimes leaves non-finite adapter weights: 1, 73 and 3 failed trainings in the three seeds, none without replay.
  - The edge rolls back, and its rows stay unconsumed until a training succeeds.
  - In seed 1, one SWELL edge diverges on every retry, so it never contributes again and its rows never reach the memory.
  - The learning rate was set for FedAvg on whole datasets, not for replay; C8 chooses it on validation.
- **What C8 needs from this:**
  - learning rates and replay settings chosen on the validation stream;
  - a capacity that binds, since here the memory is all the history;
  - more than three seeds.

## Files

- `<run_id>/`: `run.json`, `summary.json` and `model.npz` of the 6 result runs. The 6 determinism reruns stay in `runs/`.
- `run_metrics.csv`: per run, the scenario, memory, seed, prequential and labelled-only macro-F1 per dataset with their sample counts, the test macro-F1 per dataset, memory rows and replayed rows per dataset at the end, failed trainings, rows arrived and late labels.
- `prequential.csv`, `volume.csv` and `determinism.csv`: as in the stream comparison.

`events.jsonl` and `bundle/` are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
