# Continuum C4: replay memory at the edge

**Goal:** issue #159. A streaming edge keeps a bounded memory of rows it has already trained on, and replays a sample of them with its recent rows. This is the basic defence against forgetting, and it leaves aggregation untouched.

**Spec:** `docs/superpowers/specs/2026-10-04-onion-fl-continuum-design.md` §8 (with §4 and §7). It builds on C3's per-row consumption record (`first_consumed_by_version`).

**Architecture.**
- **Plugins.** A `memory` registry (`onion_fl.continuum.memory`) holds `none`, `fifo`, `reservoir` and `class_balanced`.
- **What a memory keeps.** Indices of its edge's own stream rows: the data never leaves the edge and is never copied.
- **When a row enters.** After the training that first consumed it ends with finite weights. A failed training therefore never puts a row in memory.
- **Replay.** Each training mixes the recent rows with a sample drawn from the memory. `continual: {memory, replay_ratio}` in the experiment config sets it up.

## Rulings (where the spec leaves a choice)

- **R1 Eligibility.** A row is offered to the memory, in time order, only after its first successful consumption. A row that is merely labelled, or that a failed round trained on, is not eligible (the user's design point in the #171 review).
- **R2 Replay share.** A training with n recent rows adds k = round(r·n / (1 − r)) rows sampled from the memory without replacement, capped by its size, where r is `replay_ratio`, in [0, 1). So on average a fraction r of every batch comes from memory.
  - Replay never makes an idle edge train: with no recent rows, the edge stays idle.
  - Replayed rows are not consumed again; `first_consumed_by_version` keeps its first round.
- **R3 Randomness.** Each memory has its own stream, `node_rng(seed, "memory/<edge>")`, for reservoir decisions and replay samples. The edge's own stream and the label masks are untouched.
- **R4 Plugins:**
  - `none` keeps nothing; it is the "recent only" baseline.
  - `fifo` keeps the last `capacity` rows.
  - `reservoir` is Algorithm R: a uniform sample of every row offered so far.
  - `class_balanced` keeps at most `capacity` rows. When it overflows, it drops the oldest row of the class with the most rows (the lowest class on a tie).
- **R5 Bundle.** Each run's bundle saves the memory: its rows, the plugin's state and its random stream. Restoring it waits for a stream run to be continuable in time; `init: run` with a stream stays refused.
- **R6 Diagnostics.** `diagnostic.memory` per edge and round, whose value is the number of rows kept. Its tags give:
  - `capacity`;
  - `age`: the mean data-time age of what is kept;
  - `classes`: rows kept per class;
  - `replayed`: rows replayed in the round.
- **R7 Privacy.** Unchanged: DP applies to the released update, which replay is part of, and stored rows get no noise.
- **R8 Config.** `continual` needs `stream`; unset, it stays out of the `config_id`.

## Global constraints

- **Plugin texts.** Titles and descriptions in Spanish; code and docs in English.
- **No synthetic data for ML.** Memory tests use hand-written indices and labels; protocol tests use the stub trainer or a recording stub. The real-data comparison uses SWELL and WESAD.
- **Unchanged without replay.** A stream run with no `continual`, or with `memory: none`, behaves exactly as in C3, bit for bit.
- **Determinism.** The same seed gives the same memory, the same samples and the same model.

## Tasks

### Task 1: memory plugins
- `onion_fl.continuum.memory`: the `memories` registry and `none`, `fifo`, `reservoir` and `class_balanced`. Each has:
  - `add(rows, y)`, `rows()` and `sample(k)`;
  - `state()` and `load_state()`. (The edge reports a memory's occupancy, age and classes itself, so no `describe()` was needed.)
- **Tests:**
  - the capacity is never exceeded;
  - `fifo` keeps the latest rows;
  - `reservoir` keeps a uniform sample (each row's frequency over many seeds is close to capacity / offered) and is the same for the same seed;
  - `class_balanced` evicts from the largest class;
  - `sample(k)` draws without replacement;
  - a state round trip leaves the next decision identical.

### Task 2: replay at the edge
- The `Edge` takes `memory` and `replay_ratio`. Training gets the recent rows plus k replayed rows; after a successful training, the newly consumed rows are offered to the memory. The edge emits `diagnostic.memory`.
- The snapshot saves `replay/rows` and the memory's state.
- **Tests (stub or recording trainer):**
  - rows enter the memory only after a successful training (a raised or non-finite round adds nothing);
  - each training holds about r of replayed rows;
  - an edge with no recent rows stays idle;
  - same seed, same run;
  - without memory, nothing changes.

### Task 3: config and runner
- `continual: {memory: <plugin>, replay_ratio}`, with `exclude_if None`, refused without `stream`.
- The runner gives each training edge its memory, seeded per edge. The Studio gets the axis.
- **Tests:**
  - a stream run with replay records `diagnostic.memory`;
  - the refusal without `stream`;
  - the `config_id` is unchanged without `continual`;
  - the Studio axis.

### Task 4: real-data comparison
- `experiments/stream_replay_swell_wesad.yaml`: C3's stream replay with full labels, comparing recent only (`memory: none`) with recent plus replay (`reservoir`, 512 rows, `replay_ratio` 0.5), on 3 seeds.
- **Committed results** in `results/continuum_replay/`: prequential and test macro-F1 (pooled from the confusion matrices), and memory occupancy.
- The comparison is ready for C8.

### Task 5: docs, review and PR
- `docs/architecture.md`, the spec rulings and the README.
- A fresh review of the whole branch, then one fix pass, test-first.
