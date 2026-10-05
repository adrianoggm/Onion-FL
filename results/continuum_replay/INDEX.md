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
  - The fogs and the cloud need every child (quorum 1.0), so one failed edge fails its fog's round, and with it the cloud's.
- **Two scenarios:**
  - **Recent only** (`continual.memory: none`): each round trains on the rows it has not used yet.
  - **Replay** (`reservoir`, 512 rows, `replay_ratio: 0.5`): each round also trains on a sample of rows the edge already trained on, as many as its new rows. A row enters the memory only after its first successful training, and replay never makes an idle edge train.
- **Scores.**
  - `prequential` scores every arrival with the model the edge was serving.
  - **The test scores are the final model's**: `model.npz`, scored offline on the test subjects and pooled from the summed confusion matrices. The root's last in-run evaluation would not always do: when the last rounds fail, it is from an earlier round. Where the run's last round did evaluate, the offline scores equal the in-run ones (`inrun_test_swell_macro_f1`).
- **Seeds** 0–2.

| Identity | Value |
|---|---|
| `topology_id` | `ff55fe4d2ce1a807f764bcc5205a8d2d4d43ab3d533163a1d13e173a9a6725fa` |
| `data_id` | `f79c1afad836591c82f72c11c6409118eeeb9e60c814b60214f166762d92b2c0` |
| code | commit `0ca028e`, clean tree |
| `config_id`, recent only | `c77b4c88a0f89687b6c29e59f702d8bee83eb47a51de026375031c7fcb2a60a9` |
| `config_id`, replay | `127bb8bba6c1eaf1a1924bc64e5a3664784b840136a82c4e35d13c871d566ec3` |

## Results (macro-F1, mean ± 95% CI over 3 seeds; `run_metrics.csv`)

| Scenario | Prequential SWELL | Prequential WESAD | Final model, test SWELL | Final model, test WESAD |
|---|---|---|---|---|
| Recent only | 0.783 ± 0.045 | 0.000 ± 0.000 | 0.404 ± 0.000 | 0.262 ± 0.000 |
| Replay | 0.633 ± 0.031 | 0.000 ± 0.000 | 0.351 ± 0.229 | 0.305 ± 0.187 |

**Per seed, with replay** (recent only: no failed training, no failed round, and a final model that predicts stress for every test row in all three seeds):

| Seed | Failed trainings | Cloud rounds failed | Final model on the test rows | Final test loss (SWELL, WESAD) |
|---|---|---|---|---|
| 0 | 1 (round 17) | 17 | no stress for every row | 4.9·10⁷, 8.2·10⁵ |
| 1 | 73: one in round 13, then all 18 SWELL edges in each of rounds 15–18 | 13, 15, 16, 17, 18 | stress for every row (the round-14 model) | not finite, 2.1·10³³ |
| 2 | 3 (rounds 13, 14, 16) | 13, 14, 16 | stress for every row | 0.63, 0.81 |

**Memory at the end of the run** (from each run's bundle):
- With replay, SWELL edges keep 2255 rows (seeds 0 and 2) and 1820 (seed 1); WESAD edges keep all 572.
- Each SWELL edge consumes about 125 rows, so the capacity of 512 never binds: the reservoir holds every row it was offered.
- In seed 1, the 435 rows labelled in rounds 15–18 never enter the memory, since no training that used them succeeded.

**Checks:**
- **Determinism** (`determinism.csv`): each scenario and seed, run twice, gives the same final model and the same events.
- **Recent only reproduces C3:** its final models equal those of the full-labels stream runs in [continuum_stream/](../continuum_stream/INDEX.md), bit for bit.

## Reading

- **Replay does not help in this setup; it destabilises training.**
  - **Divergence.** At lr 0.1, training on twice the rows makes SWELL edges diverge (non-finite adapter weights).
  - **Lost rounds.** With quorum 1.0, each failed edge fails its fog's round and the cloud's: 1, 5 and 3 cloud rounds are lost in the three seeds.
  - **Seed 1.** From round 15 every SWELL edge diverges on every retry. Their rows stay unconsumed and out of the memory, and the global model stays at round 14's.
  - **Diverged global models.** The global models of seeds 0 and 1 have diverged too: test losses of 10⁵ and more, or not finite.
- **The prequential score drops.**
  - On SWELL it falls from 0.783 to 0.633. After a condition change, the remembered rows of the old condition hold the old prediction for longer: per round, recent only recovers by round 9 and replay by round 12 (`prequential.csv`, seed 0).
  - Since SWELL's prequential score mostly measures persistence, slower switching costs it directly.
- **WESAD stays at 0 either way.** After the bootstrap every WESAD window is stress, and no served model predicts it before the sessions end.
- **Every final model is a constant predictor.** Recent only ends at "stress" for every test row in all seeds. Replay ends at "stress" or "no stress", depending on the seed, which is all the wide test intervals reflect.
- **What C8 needs from this:**
  - learning rates and replay settings chosen on the validation stream, since lr 0.1 was set for FedAvg on whole datasets;
  - a capacity that binds;
  - a quorum below 1.0, or deadlines, so that one diverged edge does not cost the round;
  - the final model's test score (as here), not the last in-run evaluation;
  - more than three seeds.

## Files

- `<run_id>/`: `run.json`, `summary.json` and `model.npz` of the 6 result runs. The 6 determinism reruns stay in `runs/`.
- `run_metrics.csv`, per run:
  - the scenario, memory and seed;
  - the final model's test macro-F1, loss and confusion matrix per dataset (offline), the round of the root's last in-run evaluation and its SWELL macro-F1, and the cloud rounds that failed;
  - prequential and labelled-only macro-F1 per dataset with their sample counts;
  - memory rows per dataset at the end (from the bundle), and rows replayed per dataset over the run (failed attempts included);
  - failed trainings, rows arrived and late labels.
- `prequential.csv`, `volume.csv` and `determinism.csv`: as in the stream comparison.

`events.jsonl` and `bundle/` are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
