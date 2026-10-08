# Triggers on the SWELL and WESAD streams (new framework, real data)

`experiments/stream_triggers_swell_wesad.yaml` was run on 2026-10-07 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #161 (continuum C6).

```bash
onion_fl run experiments/stream_triggers_swell_wesad.yaml --workers 3   # run twice: the second is the determinism check
```

These runs show the triggers on the real recordings. The volume and the detector's settings are not tuned; the benchmark is C8.

**How these runs came to be.** This is the third run of the experiment; only this one is kept.

1. **The first runs (commit `c32e81a`, not kept)** measured prior drift as the distance from each edge's history.
   - On SWELL that distance stays near 0.5 in every block, and on WESAD it is 1 from the first window.
   - Page-Hinkley looks for a rise, so it never fired.
   - The statistic was then redesigned, after seeing this data: it now compares the labels' classes with those of a reference window (below).
2. **The second runs (`27728fe`, committed and then replaced)** gave a drift trigger close to the schedule.
   - That was partly an artifact. Statuses still in flight when a round opened counted toward the next one, which added a round at minute 70.
   - The review of the branch found this and four other faults, all fixed before these runs (see the PR of #161).
3. **The volume of 300 rows** replaces the plan's 500: it is about two C3 rounds' worth of SWELL + WESAD rows, so the trigger fires within the run.

## Setup

- **Stream, labels and training.** As in [continuum_stream/](../continuum_stream/INDEX.md) with every row labelled:
  - SWELL + WESAD replayed in time order after a 20-minute bootstrap;
  - labels 10 minutes late;
  - FedAvg with SGD at lr 0.1 and 10 local epochs, on the segregated lossless topology `four_fogs_swell_wesad_lossless`;
  - no replay memory.
- **Three ways to open the rounds** (`continuum.trigger`). Each also opens round 1 at registration and a final round after the last label:
  - **schedule**: every 10 minutes of data, with a status every 10 minutes, C3's pace;
  - **volume**: once the edges report 300 rows that became trainable after the current round reached them, with a status every 5 minutes;
  - **drift**: when a prior shift is detected at an edge, a fog or the cloud, with a status every 5 minutes.
    - The statistic is the total variation between the classes of the labels that arrived in a status window and those of the reference window.
    - The reference is the edge's first window with labels, held until a shift is detected, when the window that showed it replaces it.
    - Each node watches its statistic with Page-Hinkley (`delta` 0.005, `threshold` 0.5, at least 3 windows). Fogs and the cloud watch the sample-weighted mean of what they hear.
- **Scores.** `prequential` scores every arrival with the model its edge was serving. The test scores are the final `model.npz`'s, scored offline on the test subjects.
- **Seeds** 0–2.

| Identity | Value |
|---|---|
| `topology_id` | `ff55fe4d2ce1a807f764bcc5205a8d2d4d43ab3d533163a1d13e173a9a6725fa` |
| `data_id` | `9fd573754157d68b111f037ecd891bdb62abcbff1d5e5e818da789dde470257c` |
| code | commit `29bf949`, clean tree |
| `config_id`, schedule | `a13e9d5fdc991b9802592ee83e922d57235a1ee3cedb79cef32796f44bc1cafa` |
| `config_id`, volume | `d392aa5255f770024928c2df5c18a66b2b949955bbc05a0574bfb9c7aea39fa1` |
| `config_id`, drift | `c756445833acb983dbf3cc5d692c2d4fbfb9208cf05f3845467ccb1a78e5e3c8` |

## Results (mean ± 95% CI over 3 seeds; `run_metrics.csv`)

| Trigger | Rounds | Bytes sent | Prequential macro-F1, SWELL | Prequential macro-F1, WESAD | Final model, test SWELL | Final model, test WESAD |
|---|---|---|---|---|---|---|
| schedule (10 min) | 18 | 186 MB | 0.783 ± 0.045 | 0.000 ± 0.000 | 0.404 ± 0.000 | 0.262 ± 0.000 |
| volume (300 rows) | 8 | 86 MB | 0.514 ± 0.023 | 0.003 ± 0.014 | 0.404 ± 0.000 | 0.262 ± 0.000 |
| drift (prior) | 4 | 45 MB | 0.203 ± 0.032 | 0.056 ± 0.122 | 0.404 ± 0.000 | 0.262 ± 0.000 |

**When the rounds opened** (`triggers.csv`, data minutes after t₀; the same in every seed):
- **schedule:** every 10 minutes from 0 to 160, then the final round at 170.
- **volume:**
  - rounds at 0, 15, 35, 70, 95, 135 and 165, then 170;
  - each volume round opened on 328 to 377 reported rows.
- **drift:**
  - 0;
  - 60, on the edges' and fogs' detections, with 1100 rows reported since round 1;
  - 65, on the cloud's own detection;
  - then the final round at 170.

**What the detectors saw** (events `drift.detected`, the same in every seed):
- **SWELL edges.** Each of the 18 detects exactly one shift, between minutes 55 and 65. It is each subject's switch from the no-stress block to the stress blocks, visible once its labels arrive 10 minutes late.
- **Fogs and cloud.** Both SWELL fogs detect it on their pooled statistic at minute 60, and the cloud at 65.
- **WESAD.** Nothing is detected for WESAD. Every WESAD subject's history is baseline and every row after t₀ is stress, so its first window is its reference, and nothing changes afterwards.

**Checks:**
- **Determinism** (`determinism.csv`): each scenario and seed, run twice, gives the same final model and the same events.
- **The schedule reproduces C3:** its final models equal those of the full-labels stream runs in [continuum_stream/](../continuum_stream/INDEX.md), bit for bit, in all three seeds.
- **No failed training** and no failed round in any run.

## Reading

- **A prior-drift trigger alone sees the switch, not what follows.**
  - It opens rounds at the SWELL switch (60 and 65 minutes) and then none until the end, at a quarter of the schedule's bytes.
  - The model served from minute 65 predicts no stress for the whole stress block, and nothing fires again, because the classes of the arriving labels no longer change. Seed 0's SWELL arrivals from 65 to 170 minutes score 0.0.
  - Hence the prequential score of 0.203, against the schedule's 0.783.
  - A trigger that also watches the error (`performance`), or a schedule next to the drift (`any`), would reopen rounds while the served model is wrong.
- **The volume trigger** opens rounds when rows pile up, not at the switches. With fewer than half the schedule's rounds, the served models are staler at the switches, and its prequential score is 0.514.
- **WESAD stays near 0** whatever the trigger. After the bootstrap every window is stress, and no served model predicts it before the sessions end, as in the stream and replay runs.
- **Every final model is the same constant predictor** ("stress" for every test row), so the test scores cannot tell the triggers apart. That is the stream setup's limit: lr 0.1 was set for FedAvg on whole datasets, and C8 chooses it on validation.
- **What C8 needs from this:**
  - triggers combined with `any`, among them a `performance` drift or a schedule that keeps the served model from going stale;
  - detector settings chosen on the validation stream;
  - the cost (rounds, bytes) next to the prequential score, as here;
  - more than three seeds.

## Files

- `<run_id>/`: `run.json`, `summary.json` and `model.npz` of the 9 result runs. The 9 determinism reruns stay in `runs/`.
- `run_metrics.csv`, per run:
  - the trigger, seed, rounds and triggers fired by name;
  - drift detections by level (edge, fog, global) and status messages sent;
  - prequential macro-F1 per dataset;
  - the final model's test macro-F1 and confusion matrix per dataset (offline);
  - cloud rounds failed, failed trainings and bytes sent;
  - whether the schedule's model equals C3's.
- `triggers.csv`: every round's trigger, data time and the volume reported.
- `prequential.csv` and `determinism.csv`: as in the stream comparison.

`events.jsonl` and `bundle/` are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
