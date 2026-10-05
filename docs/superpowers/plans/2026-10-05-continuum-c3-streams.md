# Continuum C3: streams and delayed labels

**Goal:** issue #158. Each training subject replays its real rows to its edge in time order, a seeded fraction of them labelled, with each label arriving after a delay. Edges are scored test-then-train on what arrives, and the volume of data and labels is in the events.

**Spec:** `docs/superpowers/specs/2026-10-04-onion-fl-continuum-design.md` §4.1–4.3 and §7 (with §3 and §14).

**Architecture.**
- **Per-row time.** `SubjectData.t` holds seconds since the subject's recording started. It is metadata, never a feature, and comes from a `time` ingest step.
- **The schedule.** `EdgeStream` (`data/stream.py`) turns a subject's rows plus `stream`/`labels` into a schedule in continuum time (data time since the stream started). For each row it gives the arrival time, the label time (none for unlabelled rows) and the time it becomes trainable. It is a pure function of the data, the config and the seed.
- **The edge works lazily.** It reads `ctx.now()` when a global model arrives and turns it into continuum time (virtual seconds × `speed`). Then it:
  1. predicts what arrived since the last round with the model it was serving;
  2. scores those stored predictions;
  3. serves the new model;
  4. trains on its buffer of recently trainable rows.

  The served model only changes when a global model arrives, so lazy processing is exact and needs no timers.
- **Pacing.** The coordinator spaces its rounds by `stream.round_every`, in data time. C6 replaces this with triggers.

## Rulings (where the spec leaves a choice)

- **R1 Pacing.** `stream.round_every` (data time) is required with `stream`. Round r opens at the first round's virtual time plus (r − 1)·`round_every`/`speed`. Without it, rounds run back to back and almost no data arrives.
- **R2 End.** The run lasts `min(rounds, rounds that reach the horizon)`. `horizon: session` is the latest end of a training subject's stream. C6 lets the coordinator run until the horizon.
- **R3 Labels.** `labels.fraction` and `labels.delay` apply to every row, history included, so `fraction: 0` means no label ever reaches training (the issue's acceptance criterion).
  - Each subject labels exactly round(fraction·n) of its rows, drawn with `node_rng(seed, "labels/<dataset>/<subject>")`.
  - The mask therefore does not depend on placement, and draws nothing from the edge's stream.
- **R4 Bootstrap.**
  - `bootstrap: <duration>` puts t₀ at that data time for every subject; `{samples: N}` puts each subject's t₀ at its N-th row.
  - Rows before t₀ are history: they are at the edge when the stream starts, and are never predicted.
  - The preprocessing is fitted on the history rows of the training subjects only.
- **R5 Served model.** It is the last global model the edge received, with its own local groups. A prediction is stored with the round of that model (`predicted_by_version`).
- **R6 Two prequential scores, both on the predictions stored at arrival:**
  - `prequential`: every row that arrived in the window, against its true label. Simulation only, since unlabelled truth is hidden from the learner.
  - `prequential_labelled`: the predictions whose label arrived in the window. This is what a deployment could measure.
- **R7 Buffer.** Rows that became trainable since the edge's previous round; `stream.window`, when set, caps how old they may be. Older ones are C4's memory. (The first version anchored the buffer to the last window before now, which lost or repeated rows when rounds ran late; the review's fix pass changed it.)
- **R8 Idle edges.**
  - An edge with an empty buffer sends an idle update, with no state but with its scores.
  - Collectors leave idle children out of the quorum.
  - A round where every child is idle closes as `round.idle`, without changing the model, and a fog passes the idle answer and the scores up.
- **R9 Batches.** Rows arrive in batches of `batch_size` in time order; a batch is available at the time of its last row. `stream.latency` is left out (the spec's default, 0).
- **R10 Time of a window.** WESAD windows are timed at their end, since a window is only observable once complete. SWELL rows use their timestamp.
- **R11 Evaluation streams.**
  - Zone and `gval` evaluators with a stream score only the rows that arrived since their previous request.
  - Test evaluators keep scoring their whole subjects, so the final test score stays comparable with `results/`.
- **R12 Refused with `stream`:**
  - real mode, and `init: run`, until the bundle holds the stream state;
  - `local_val > 0` (a positional tail conflicts with time order);
  - `scaler: local`;
  - `staggered` with `subjects_per_client > 1`;
  - a dataset without per-row time.

## Global constraints

- **Plugin and config texts.** Titles and descriptions in Spanish; code and docs in English.
- **No synthetic data for ML.**
  - Schedule tests use hand-written arrays.
  - Protocol tests use the stub trainer, or a recording stub.
  - Ingest tests use small format fixtures.
  - Anything that trains for real uses the real extracts and skips without them.
- **Old runs keep their `config_id`.** `stream` and `labels` are left out of the identity when unset, and a run without `stream` behaves as before (the existing suite is the check).
- **Determinism.** The same seed gives the same arrivals, labels, predictions and model.

## Tasks

### Task 1: per-row time
- `SubjectData.t: np.ndarray | None` (float64, one per row) is validated, and carried by `_like`, `_join` and the cache: saved, loaded and covered by the digest.
- **Ingest.** A `time` step either parses a column (`{column, format}`) or derives seconds from the row position (`{rate}`). It registers the result as a meta column. `window` reduces it to the window's end, and `ingest()` makes it relative per subject and never a feature.
- **Descriptors.** `swell.yaml` and `swell_physiology.yaml` parse `timestamp` after the joins; `wesad.yaml` uses `{rate: 700}` before `window`.
- **Tests:**
  - parsing and windowing on format fixtures;
  - `t` is kept out of the features;
  - the cache round trip;
  - on real data (skipped without it): `t` is non-decreasing per subject and spans the session.

### Task 2: the split keeps time and fits on the bootstrap
- `split_subjects(..., fit_rows=None)` takes a callable that gives the rows usable for fitting each subject (the bootstrap). The preprocessing bag and `drop_constant` use only those rows.
- Clients' training data keeps `t`; with several subjects per client, rows are ordered by time.
- **Tests:**
  - the fit ignores rows after t₀ (a value change after t₀ leaves mean and std unchanged);
  - `t` survives into clients and evaluators.

### Task 3: the schedule
- `StreamConfig` and `LabelsConfig` in the experiment config:
  - `order: timestamp`, `batch_size`, `speed`, `start: aligned | {staggered}`, `horizon: session`, `bootstrap`, `round_every`, `window`;
  - `fraction` and `delay`, all durations in data time.
- `EdgeStream(data, stream, labels, seed, offset)` gives:
  - per row: `available_at`, `label_at` (∞ when unlabelled), `trainable_at` and `history`;
  - windows as masks: `arrived(lo, hi)`, `trainable(lo, hi)` and `labelled(lo, hi)`;
  - `horizon`.
- **Tests (hand-written arrays):**
  - batches arrive at their last row;
  - the bootstrap rows are history;
  - exactly round(f·n) rows are labelled, the same for the same seed and different for another;
  - with `fraction: 0` nothing is trainable;
  - nothing is trainable before `label_at`;
  - staggered offsets shift arrivals.

### Task 4: streaming edges, idle rounds and pacing
- **Edge** (`stream=`). On each global model:
  1. it predicts the rows that arrived with the model it was serving;
  2. it emits `data.arrived` (rows, labelled and unlabelled) and `data.labelled` (labels that arrived);
  3. it scores `prequential` and `prequential_labelled` into the update metrics, so `_children_scores` reduces them per fog and globally;
  4. it serves the new model;
  5. it trains on its buffer, or sends an idle update when the buffer is empty.
- **Collectors.** Idle children are left out of the quorum; an all-idle round closes as `round.idle`, and the aggregator forwards idle answers and scores up.
- **Coordinator.** `round_every` (virtual seconds) arms a timer before opening the next round.
- **Tests (stub trainer, plus a recording stub that keeps the rows it was given):**
  - with `fraction: 0` the trainer is never called and every round is idle;
  - with `delay`, every row given to training has `label_at ≤` the edge's time at that round;
  - the volume of each round is in `data.arrived`;
  - the same seed gives the same events and model;
  - rounds are spaced by `round_every`;
  - a non-stream run is unchanged.

### Task 5: evaluation streams
- An evaluator with a stream scores the rows arrived since its previous request; without one, it scores its whole subject as before.
- **Test:** successive zone evaluations cover disjoint windows.

### Task 6: experiment wiring
- `ExperimentConfig.stream` and `.labels` (with `exclude_if None`), with the refusals of R12.
- The runner:
  - fits the split on the bootstrap;
  - builds an `EdgeStream` per client and per stream evaluator;
  - sets the coordinator's pacing and the number of rounds from the horizon;
  - records `data.stream` (t₀, horizon, rounds, rows per edge);
  - adds a stream summary to `plan`.
- **Tests on a format fixture with timestamps:**
  - a stream run finishes, and its events carry the volume;
  - the same seed gives an identical run;
  - each refusal;
  - `config_id` is unchanged without `stream`.

### Task 7: a real-data stream run
- `experiments/stream_swell_wesad.yaml`: SWELL + WESAD over lossless links, `labels.fraction` 0.2 and 1.0, 3 seeds.
- **Committed results** (`results/continuum_stream/`), at a clean commit:
  - prequential macro-F1 per round;
  - data and label volume;
  - the final test score;
  - the determinism check (two runs of the same seed are identical).
- It shows that the stream works on the real recordings. The benchmark is C8.

### Task 8: docs, review and PR
- `docs/architecture.md` §7c, the spec rulings, and the README status.
- A fresh review of the whole branch, then one fix pass, test-first.
