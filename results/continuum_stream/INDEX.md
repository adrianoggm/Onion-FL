# SWELL and WESAD as streams with delayed labels (new framework, real data)

`experiments/stream_swell_wesad.yaml` was run on 2026-10-05 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #158 (continuum C3).

```bash
onion_fl run experiments/stream_swell_wesad.yaml --workers 3   # run twice: the second is the determinism check
```

This run checks that the stream works on the real recordings. It is not a benchmark of continual learning: that is C8, once replay memory (C4) and triggers (C6) exist.

## Setup

- **Topology and placement.** `four_fogs_swell_wesad_lossless` (no message loss) with the segregated placement: two fogs hold only SWELL edges and two only WESAD edges. Each edge holds one training subject; 18 SWELL and 10 WESAD edges.
- **The stream.**
  - Each training subject replays its rows in time order, one row at a time (`batch_size: 1`), each when it ends: a SWELL minute, or a WESAD 60 s window (two per minute, half overlapping).
  - Data time runs 60 times faster than the simulation's clock (`speed: 60`).
  - A round opens every 10 minutes of data. The run lasts until the last row has arrived: 17 rounds, set by SWELL's longest session (159 minutes after the bootstrap).
- **Bootstrap.**
  - The first 20 minutes of each subject are history: they fit the preprocessing.
  - In round 1, v0 trains on the history whose labels are already due: half of it, given the 10-minute delay. The rest joins in round 2.
  - SWELL's first 20 minutes hold both classes. WESAD's are all baseline, because its protocol starts with about 20 minutes of baseline, so v0 has never seen WESAD stress.
- **Labels.**
  - Two scenarios: every row labelled, or a seeded fifth of each edge's rows.
  - Either way, each label arrives 10 minutes after its row. A history label already due before the stream started is there at the start.
- **Training.**
  - Each round, each edge trains on what became trainable since its previous round, with nothing older: there is no replay memory before C4.
  - The model and the FedAvg settings are those of the drift comparison (SGD at lr 0.1, 10 local epochs), not tuned for streams.
- **Evaluation.**
  - **Prequential.** Every row that arrives after round 1 is scored by the model the edge was serving when the row came, before the row can train. The few rows that arrive with round 1, before any model is served, are not scored. `prequential` scores every arrival against its true label, which only a simulation can know; `prequential_labelled` scores only the predictions whose label has arrived, which is what a deployment could measure.
  - **Validation subjects** stream for evaluation only.
  - **Test subjects** never reach an edge. The global model is scored on them every 5 rounds and at the end.
- **Seeds** 0–2.

| Identity | Value |
|---|---|
| `topology_id` | `ff55fe4d2ce1a807f764bcc5205a8d2d4d43ab3d533163a1d13e173a9a6725fa` |
| `data_id` | `f79c1afad836591c82f72c11c6409118eeeb9e60c814b60214f166762d92b2c0` |
| code | commit `2da1542`, clean tree |
| `config_id`, every row labelled | `82b775953ad9bd6093420ea89e6de76b781264d4034aa2bbd189bab42c6b553f` |
| `config_id`, a fifth labelled | `64dffbbe3fe898ad7920c27ae54c67b31e05f3630d743b77117f3cf8d8a045a7` |

The `data_id` differs from earlier results because the SWELL and WESAD descriptors gained a `time` step. Features and labels are unchanged; the descriptor parity tests still pass.

## Results (macro-F1, mean ± 95% CI over 3 seeds; `run_metrics.csv`)

The prequential scores pool every edge's predictions per dataset over the whole stream. `prequential.csv` has them per round.

| Labels | Prequential SWELL | Prequential WESAD | Labelled-only SWELL | Labelled-only WESAD | Test SWELL | Test WESAD |
|---|---|---|---|---|---|---|
| Every row | 0.783 ± 0.045 | 0.000 ± 0.000 | 0.781 ± 0.045 | 0.000 ± 0.000 | 0.403 ± 0.000 | 0.262 ± 0.000 |
| A fifth | 0.800 ± 0.037 | 0.000 ± 0.000 | 0.779 ± 0.029 | 0.000 ± 0.000 | 0.403 ± 0.000 | 0.262 ± 0.000 |

**Volume** (`volume.csv`, seed 0):

| Labels | Dataset | Rows | History | Labelled on arrival | Late labels | Last round with arrivals |
|---|---|---|---|---|---|---|
| Every row | SWELL | 2255 | 278 | 122 | 2110 | 17 |
| Every row | WESAD | 572 | 371 | 210 | 362 | 6 |
| A fifth | SWELL | 2255 | 278 | 28 | 419 | 17 |
| A fifth | WESAD | 572 | 371 | 33 | 82 | 6 |

- "Labelled on arrival" counts history rows whose label was already due. Every other label is late.
- A few labels are still due when the run ends.

**Determinism** (`determinism.csv`): running each scenario and seed twice gives the same final model, bit for bit, and the same events, in all 6 pairs.

**The review's buffer fix.**
- These runs are from after the fix. The first version of the buffer took the last 10 minutes before each round, which can lose or repeat rows when rounds run late; the fix takes what became trainable since the previous round.
- In these evenly spaced runs, the final models are bit-identical to the first runs, made at commit `c3e7868`.

## Reading

- **The stream mechanics hold on the real recordings.**
  - Rows arrive in time order and every one is counted.
  - Labels arrive late and are never trained on before they arrive.
  - The runs are deterministic.
- **WESAD's prequential score is 0 because of label drift that the setup cannot follow.**
  - After the bootstrap, every WESAD window is stress, and the served model predicts baseline for all of them, in every seed and both scenarios.
  - WESAD edges do train on the stress windows as their labels arrive (rounds 3–7). But each round brings only a few windows, and the head is shared by task with 18 SWELL edges, so the served model never flips before WESAD's sessions end, by round 6.
- **SWELL's high prequential score mostly measures persistence.**
  - A SWELL condition lasts many minutes, and the model trained on the latest rows predicts the condition the next minute is in. Per round, most windows hold a single class, so the per-round score is 0.5 (all right) or 0 (all wrong), and the misses come at the block changes.
  - On held-out subjects, the same models score 0.403: the always-stress score.
- **Training on the latest rows alone forgets.**
  - At the end, in every run, the global model predicts stress for every test row of both datasets (the confusion matrices are in `summary.json`). A constant predictor scores the same whatever the seed, which is why the test scores have no spread.
  - This is the failure that replay memory (C4) is meant to address.
- **A fifth of the labels changes little in these runs.** Its intervals overlap those of full labelling on every measure.

## Files

- `<run_id>/`: `run.json`, `summary.json` and `model.npz` of the 6 result runs. The 6 determinism reruns stay in `runs/`.
- `run_metrics.csv`: per run, the scenario, seed, rounds, prequential and labelled-only macro-F1 per dataset with their sample counts, final test macro-F1 per dataset, rows arrived and late labels.
- `prequential.csv`: per run, model (`prequential` or `prequential_labelled`), dataset and round, the pooled samples and macro-F1.
- `volume.csv`: per run, dataset and round, rows arrived, history rows, rows labelled on arrival and late labels.
- `determinism.csv`: per scenario and seed, the result run, its rerun, and whether their models and events are identical.

`events.jsonl` and `bundle/` are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
