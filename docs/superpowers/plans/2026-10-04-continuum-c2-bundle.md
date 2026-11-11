# Continuum C2: model bundle, checkpoint and lineage

**Goal:** issue #157. A run can stop and another can continue it exactly, or extend it with new data. The continuation is traceable through its lineage and keeps the preprocessing of its parent.

**Spec:** `docs/superpowers/specs/2026-10-04-onion-fl-continuum-design.md` §6 (with §3 and §14).

**Architecture.**
- Every node can export and import its state:
  - the coordinator: global model, round and server optimizer;
  - aggregators: zone groups;
  - edges: model, trainer memory and DP release count;
  - every node: its RNG stream.
- `snapshot_federation`/`restore_federation` move that state in memory. The `ModelBundle` writes it to `runs/<run_id>/bundle/`, together with the preprocessing, schema and lineage.
- `init: {name: run}` reads a parent bundle, verifies the parent, and restores what `restore` asks for.

## Global constraints

- **Plugin texts.** Titles and descriptions in Spanish; code and docs in English.
- **No synthetic data for ML.** Protocol tests use the `stub` trainer, with seeded noise when randomness matters. Real-data tests skip without `data/`.
- **Bundles.** They hold no pickles (`.npz` and JSON only). `run_hash` covers them.
- **Reproducibility.** A bundle-less run keeps its `run_hash` and behaviour unchanged.

## Tasks

### Task 1: RNG state
- `rng_state(rng)` and `restore_rng(rng, state)` in `core/context.py`. They capture the bit generator's state plus the number of children spawned from its seed sequence, so `child_rng` keeps giving the same children after a restore.
- **Test:** after restoring, draws and spawned children equal those of an uninterrupted generator.

### Task 2: component state
- **Server optimizers.** `state()` and `load_state()` for FedAvgM (velocity), the adaptive base (m, v) and FedDyn (h). The others are stateless.
- **Trainers.** `export_memory()` and `import_memory(arrays, model)`, built on `_memory`. A module is exported as its state dict, a dict of tensors or arrays as arrays, a scalar as a 0-d array, and None as absent.
- **Tests:** a round trip per stateful optimizer and trainer leaves the next step or train identical.

### Task 3: federation snapshot and restore
- `snapshot_federation(federation)` and `restore_federation(federation, snapshot)` in `roles`:
  - coordinator: state, round, server optimizer;
  - aggregators: zone groups and previous aggregate;
  - edges: model arrays, trainer memory, `_released`;
  - every node: its RNG.
- `build_federation(start_round=...)` continues the round numbering.
- **Rules for edges that change:** an edge without a snapshot starts fresh, and a snapshot without an edge is kept but unused.
- **Test (the acceptance test):** 4 rounds straight equal 2 rounds, a snapshot, and 2 more rounds in a new federation, with the stub trainer and noise.

### Task 4: preprocessing as an artifact
- `DataSplit.preprocessing`, per dataset: kept feature names, fill, mean, std and scaler mode, serialisable to JSON.
- `split_subjects(..., frozen=...)` applies a frozen preprocessing to its datasets instead of fitting, and fits the others (schema evolution).
- **Tests:**
  - a frozen preprocessing reproduces the parent's arrays exactly, even when the training subjects change;
  - a new dataset is fitted.

### Task 5: the bundle on disk
- `onion_fl.continuum.bundle`: `save_bundle(path, snapshot, split, config, lineage)` and `load_bundle(path)`. It writes `model.npz`, `server.npz`, `nodes/<id>.npz` (with RNG state as JSON), `preprocessing.json`, `schema.json`, `lineage.json` and `config.yaml`.
- Every run writes `bundle/`, and `compute_run_hash` covers it when present.
- **Test:** the round trip through disk equals the in-memory snapshot.

### Task 6: `init: run`
- `learning.init: {name: run, run: <run_id or path>, restore: {model, preprocessing, server_state, edge_state}}`.
  - It verifies the parent with `verify_run`.
  - It restores what is asked, continues the round numbering, and records the lineage in `run.json`: parent `run_id`, `run_hash` and version (the parent's last round).
  - The new datasets get their preprocessing fitted, their adapter initialised and, for a new task, their head.
- **Tests:**
  - a continued run equals an uninterrupted one;
  - a tampered parent is refused;
  - the lineage is in `run.json`;
  - with `restore.model: false`, the start is random.

### Task 7: per-group learning rate
- `group_lr: {pattern: factor}` on the standard trainer (and its subclasses), next to the existing `frozen`.
- **Test:** the factor scales the step of the matching groups only.

### Task 8: real-data experiment and results
- `experiments/continuum_warm_start.yaml`:
  - a SWELL-only parent run;
  - then SWELL + WESAD continued from it (adapters for WESAD new, trunk and SWELL kept), against SWELL + WESAD from scratch.
- **Measures:** before and after on the same test subjects, and forgetting on SWELL.
- **Hyperparameters** chosen on validation.
- **Results** in `results/continuum_warm_start/`.

### Task 9: review and PR
- A fresh review of the whole branch, then one fix pass, test-first.
- The full suite.
- A PR into `develop`, left for the user's review.
