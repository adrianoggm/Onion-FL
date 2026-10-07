# Triggers on the SWELL and WESAD streams (new framework, real data)

`experiments/stream_triggers_swell_wesad.yaml` was run on 2026-10-07 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #161 (continuum C6).

```bash
onion_fl run experiments/stream_triggers_swell_wesad.yaml --workers 3   # run twice: the second is the determinism check
```

These runs show the triggers on the real recordings. Neither the volume nor the detector is tuned; the benchmark is C8.

## Setup

- **Stream, labels and training.** As in [continuum_stream/](../continuum_stream/INDEX.md) with every row labelled:
  - SWELL + WESAD replayed in time order after a 20-minute bootstrap;
  - labels 10 minutes late;
  - FedAvg with SGD at lr 0.1 and 10 local epochs, on the segregated lossless topology `four_fogs_swell_wesad_lossless`;
  - no replay memory.
- **Three ways to open the rounds** (`continuum.trigger`). Each also opens round 1 at registration and a final round after the last label:
  - **schedule**: every 10 minutes of data, with a status every 10 minutes, C3's pace;
  - **volume**: once 300 new rows have become trainable across the federation, with a status every 5 minutes;
  - **drift**: when a prior shift is detected at an edge, a fog or the cloud, with a status every 5 minutes. The statistic is the total variation between the classes of the labels that arrived in a status window and those of the window before. Each node watches it with Page-Hinkley (`delta` 0.005, `threshold` 0.5, at least 3 windows).
- **Scores.** `prequential` scores every arrival with the model its edge was serving. The test scores are the final `model.npz`'s, scored offline on the test subjects.
- **Seeds** 0–2.

| Identity | Value |
|---|---|
| `topology_id` | `ff55fe4d2ce1a807f764bcc5205a8d2d4d43ab3d533163a1d13e173a9a6725fa` |
| `data_id` | `9fd573754157d68b111f037ecd891bdb62abcbff1d5e5e818da789dde470257c` |
| code | commit `27728fe`, clean tree |
| `config_id`, schedule | `a13e9d5fdc991b9802592ee83e922d57235a1ee3cedb79cef32796f44bc1cafa` |
| `config_id`, volume | `d392aa5255f770024928c2df5c18a66b2b949955bbc05a0574bfb9c7aea39fa1` |
| `config_id`, drift | `c756445833acb983dbf3cc5d692c2d4fbfb9208cf05f3845467ccb1a78e5e3c8` |

## Results (mean ± 95% CI over 3 seeds; `run_metrics.csv`)

| Trigger | Rounds | Bytes sent | Prequential macro-F1, SWELL | Prequential macro-F1, WESAD | Final model, test SWELL | Final model, test WESAD |
|---|---|---|---|---|---|---|
| schedule (10 min) | 18 | 186 MB | 0.783 ± 0.045 | 0.000 ± 0.000 | 0.404 ± 0.000 | 0.262 ± 0.000 |
| volume (300 rows) | 9 | 100 MB | 0.625 ± 0.055 | 0.003 ± 0.014 | 0.404 ± 0.000 | 0.262 ± 0.000 |
| drift (prior) | 5, 6, 6 | 55–67 MB | 0.769 ± 0.143 | 0.056 ± 0.122 | 0.404 ± 0.000 | 0.262 ± 0.000 |

**When the rounds opened** (`triggers.csv`, data minutes after t₀):
- **schedule:** every 10 minutes from 0 to 160, then the final round at 170.
- **volume:** 0, 15, 30, 45, 75, 95, 130, 150, then 170, in every seed.
- **drift:**
  - seed 0: 0, then 60, 65 and 70, then 170;
  - seeds 1 and 2: 0, then 45, 60, 65 and 70, then 170.

**What the detectors saw** (`run_metrics.csv`; events `drift.detected`):
- **SWELL edges.** Each of the 18 detects exactly one shift, between minutes 55 and 65, in every seed. It is each subject's switch from the no-stress block to the stress blocks, visible once its labels arrive 10 minutes late.
- **Fogs.**
  - Both SWELL fogs detect it on their pooled statistic at minute 60, in every seed.
  - In seeds 1 and 2, a WESAD fog (fog_b2 and fog_b1) also detects a shift at minute 45 on its pooled statistic, which none of its edges detects. That detection opens the round at minute 45.
- **The cloud** detects nothing. Its pooled statistic also averages in the WESAD windows, which never change, so the SWELL spike stays below the threshold.
- **WESAD edges** detect nothing on their own. Every WESAD subject's history is baseline and every row after t₀ is stress. The only change is in the first window, before Page-Hinkley's minimum of three windows, and none comes after.

**Checks:**
- **Determinism** (`determinism.csv`): each scenario and seed, run twice, gives the same final model and the same events.
- **The schedule reproduces C3:** its final models equal those of the full-labels stream runs in [continuum_stream/](../continuum_stream/INDEX.md), bit for bit, in all three seeds.
- **No failed training** and no failed round in any run.

## Reading

- **Triggers trade rounds for freshness.**
  - The drift trigger opens 5 or 6 rounds instead of 18 and sends about a third of the bytes, and its prequential SWELL score (0.769) is close to the schedule's (0.783). The rounds fall where SWELL's conditions change, which is when a fresh model matters.
  - Its interval is wide: the seeds score 0.736, 0.736 and 0.836.
- **The volume trigger** opens rounds when rows pile up, not when the conditions change. With half the schedule's rounds, the served models are staler at the switches, and its prequential score drops to 0.625.
- **WESAD stays near 0** whatever the trigger. After the bootstrap every window is stress, and no served model predicts it before the sessions end, as in the stream and replay runs.
- **Every final model is the same constant predictor** ("stress" for every test row), so the test scores cannot tell the triggers apart. That is the stream setup's limit: lr 0.1 was set for FedAvg on whole datasets, and C8 chooses it on validation.
- **What C8 needs from this:**
  - a detector tuned on the validation stream, and one that can see a change in the first windows;
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
- `triggers.csv`: every round's trigger, data time and the volume heard.
- `prequential.csv` and `determinism.csv`: as in the stream comparison.

`events.jsonl` and `bundle/` are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
