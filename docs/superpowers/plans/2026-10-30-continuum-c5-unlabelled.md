# Continuum C5: unlabelled data — implementation plan

**Goal:** an edge with a stream also learns from the rows whose label has not arrived (issue #160), through a pluggable `unlabelled` axis with `pseudo_label` and `consistency`, and reports what it did in the events.

**Spec:** `docs/superpowers/specs/2026-10-04-onion-fl-continuum-design.md` §9 (with §7 and §8 for the stream and the memory).

**Architecture:** the edge picks the unlabelled pool, hands the objective its features only (never `y`), and the objective adds a weighted term to each training batch. The pseudo-labels come from the model as received, before training. The edge, which can see the hidden label in simulation, scores the pseudo-labels afterwards.

## Decisions

- **Pool.** Rows that have arrived (history included), whose label is not there yet (`label_at > now`, never labelled included) and that no successful training has used as unlabelled. `stream.window` caps their age, as for the labelled buffer.
  - A row counts as used (`_unlabelled_by`, the round) only after a training with finite weights, as `_consumed_by` does in C3.
  - A row whose label arrives later still trains as labelled.
- **No labelled rows, no training.** An edge without new labelled rows stays idle, as in C3. Its unlabelled rows wait for the next training.
- **What the trainer gets.** `objective.fit(model, X)` runs before the training, with the received model; then, on each batch, `term(model, idx)` over a chunk of the pool. Each epoch splits a fresh permutation of the pool into as many chunks as there are labelled batches. The loss reported stays the supervised one.
- **Trainers.** The batch hook lives in `Standard.train`, so FedProx, FedBABU, SCAFFOLD, FedNova, FedDyn and MOON inherit it. Ditto, APFL and FedRep run their own loops: they declare `takes_unlabelled = False` and the edge refuses them when it is built.
- **Weight.** The aggregation weight is still the new labelled rows. Pseudo-labelled rows add no weight, as replayed rows add none in C4.
- **Memory.** Pseudo-labelled rows do not enter the replay memory. The memory replays with `stream.data.y`, which for those rows is the hidden label: that would be a leak. This is left for later, with a stored label.
- **Audit (spec §9).** For each pseudo-labelled row, the edge keeps `pseudo_label`, `confidence` and `generated_by_version` (the round of the model that generated it), in the snapshot under `unlabelled/*`, which goes with `edge_state`.
- **Config.** A top-level `unlabelled:` key (a `PluginRef`, as in the spec) that needs `stream`; `none` (the default) means C3/C4 unchanged and keeps every existing `config_id`.

## Plugins (`src/onion_fl/continuum/unlabelled.py`, registry `unlabelled`)

| Name | Params | `fit` | `term` |
|---|---|---|---|
| `none` | — | — | (the edge treats it as absent) |
| `pseudo_label` | `threshold` 0.9 (0–1), `weight` 0.5 (>0) | softmax of the received model; accepted = max prob ≥ threshold | `weight · CE(model(x_acc), label_acc)` over the accepted rows of the chunk; 0 if none |
| `consistency` | `noise` 0.1 (>0, std on the scaled features), `weight` 0.5 | — | `weight · KL(p(x) ‖ p(x + ε))`, with p(x) detached and ε ~ N(0, noise²) drawn by torch inside the seeded training fork |

`fit` returns `{"seen", "accepted", "confidence", "labels", "mask"}` (`labels`/`mask` are None for `consistency`; `confidence` is the mean max probability over the rows seen).

## Diagnostics

`diagnostic.unlabelled`, per edge and round, after a training that worked. Its value is `pseudo_labels_seen`, with these tags:
- `pseudo_labels_accepted` and `acceptance_rate`;
- `confidence`;
- `pseudo_label_accuracy`, on the accepted rows against the hidden label, when there are any (streams run only in simulation).

## Tasks (TDD; one commit each)

1. **Plugins.** `continuum/unlabelled.py` and its registry in `config.REGISTRIES` (so `onion_fl schema` exports it).
   - Tests in `tests/test_continuum_unlabelled.py`, with hand-picked logits:
     - the acceptance follows `threshold`;
     - `confidence` is the mean max probability;
     - `term` is 0 without accepted rows;
     - params are validated;
     - `consistency` gives 0 with `noise` → tiny on a fixed model.
2. **Trainer hook.**
   - `Standard.train` adds the term per batch;
   - `takes_unlabelled = False` on Ditto, APFL and FedRep.
   - Tests on the real SWELL extract (skipped without it):
     - the same seed with `none` equals today's result bit for bit;
     - `pseudo_label` with an empty pool equals no objective;
     - with an accepted pool, the weights differ.
3. **Edge.**
   - `EdgeSpec.unlabelled` and its checks: it needs a stream and a trainer that takes it;
   - the pool, `_unlabelled_by`, the event, the audit arrays and the snapshot.
   - Tests in `tests/test_roles_stream.py`, with a recording stub trainer (`takes_unlabelled = True`):
     - the objective never sees `y` (it gets an `X` only, and changing the hidden `y` of the pool leaves the training identical);
     - the pool holds exactly arrived ∧ not-yet-labelled ∧ unused rows;
     - a failed round keeps its pool;
     - an idle edge consumes nothing;
     - the event's tags, `pseudo_label_accuracy` included;
     - the snapshot keeps `unlabelled/*` under `edge_state` only;
     - Ditto is refused;
     - a pool without a stream is refused.
4. **Config and runner.**
   - `ExperimentConfig.unlabelled`, refused without `stream`;
   - `_unlabelled(scenario, edge)` creates one objective per edge.
   - Tests in `tests/test_experiment_stream.py`:
     - the `config_id` is unchanged when the key is unset;
     - the key is refused without a stream;
     - `plan` accepts it.
5. **Real-data experiments (hyperparameters chosen on validation).**
   - `experiments/stream_unlabelled_select.yaml`: the C4 streams with `labels.fraction: 0.1`, `evaluation.global.subjects: val`, and a sweep over `none`, `pseudo_label` (threshold 0.8/0.9/0.95 × weight 0.5/1.0) and `consistency` (noise 0.05/0.1), with seeds 0–2.
   - `experiments/stream_unlabelled.yaml`: `none` against the chosen configuration of each plugin, on test.
   - `results/continuum_unlabelled/` holds an INDEX with the identity of each run, the selection made on validation, the test numbers, and the pseudo-label acceptance and accuracy.
6. **Docs.**
   - `docs/architecture.md` (the continuum section);
   - the spec's C5 decisions, as in C3/C4;
   - the README status and test count.

## Review focus

- A pool row whose label arrives in the same window must train as labelled and not also as unlabelled with its own label (it leaves the pool, since its `label_at ≤ now`).
- `weight`/`threshold` at the bounds: `threshold: 1.0` accepts only certainty, and above 1 is refused.
- A pool much larger than the labelled set: the chunks stay aligned with the labelled batches, so one epoch covers the pool once.
- Determinism: the same seed gives the same pseudo-labels and the same final model (`test_runtime_equivalence` does not cover streams; a dedicated test does).
- Restoring from a bundle: `init: run` with a stream is still refused, so the audit arrays are saved but not yet restored. This is stated in the docs.
