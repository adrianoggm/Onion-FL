# Onion-FL

A framework to experiment with hierarchical federated learning (edge → fog → … → cloud): trees of any depth, fogs that mix datasets, pluggable learning techniques, a virtual-clock simulator and real runs over MQTT, with every level instrumented. It grew out of a stress-detection project on wearable and workplace data (SWELL, SWEET, WESAD) and is a master's thesis (TFM) prototype.

[![CI](https://github.com/adrianoggm/Onion-FL/actions/workflows/ci.yml/badge.svg)](https://github.com/adrianoggm/Onion-FL/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

> **About this README.** It describes v0.2.0, the redesigned framework (October 2026). Every number in it comes from a file in the repository, cited next to it. The datasets are not in git, so **no experiment of the new framework has been run on real data in this repository yet**; [§1](#1-status) says exactly what is verified and how.

## Contents

1. [Status](#1-status)
2. [Quick start](#2-quick-start)
3. [How it works](#3-how-it-works)
4. [Experiments](#4-experiments)
5. [Datasets](#5-datasets)
6. [Observability](#6-observability)
7. [Results from before the redesign](#7-results-from-before-the-redesign)
8. [Repository map](#8-repository-map)
9. [Development and CI](#9-development-and-ci)
10. [Known limitations](#10-known-limitations)
11. [Roadmap](#11-roadmap)
12. [Project history](#12-project-history)
13. [License and acknowledgments](#13-license-and-acknowledgments)

---

## 1. Status

### At a glance

| Area | Status | Summary |
|---|---|---|
| Topologies | ✅ | Trees of any depth in YAML (compact or general form), each with a `topology_id` (SHA-256 of its structure and links) and a JSON/Mermaid graph |
| Data | ✅ / ⚠️ | Declarative ingestion (readers and steps), a signed cache, subject roles fixed across scenarios, and placement plugins. The SWELL, SWEET and WESAD descriptors are **not yet checked against the raw files** ([§5](#5-datasets)) |
| Learning | ✅ | Modular model (adapter per dataset, shared or per-dataset trunk, head per task or dataset). Sharing scopes: global, per level, local. Per-key aggregators and server optimizers; `standard` and `fedprox` trainers; random or checkpoint init |
| Round protocol | ✅ | Coordinator, aggregators at any level, edges and evaluators. Registration with acknowledged `hello`, quorum and deadline, staleness and participation plugins, per-key weights, evaluation at edge, zone and global level |
| Simulation | ✅ | Virtual clock; links with latency, jitter, bandwidth and loss; compute and availability models; deterministic for a seed |
| Real runs | ✅ | One process per aggregator over MQTT; a manual `onion_fl node` start for several machines. Verified locally against Mosquitto: 3 processes, every message delivered, latencies measured, and the same final model as the simulation |
| Observability | ✅ | `runs/<run_id>/` with signed events, summary and final model. Diagnostics at every aggregator; analysis API with mean ± CI over seeds; HTML report; Prometheus and OpenTelemetry sinks; Grafana dashboard |
| CLI | ✅ | `onion_fl data · topology · plan · run · node · report · baseline · schema` |
| Front (Onion-FL Studio) | ❌ | Planned for v0.3.0 ([#103](https://github.com/adrianoggm/Onion-FL/issues/103)) |
| gRPC and Flower transports, distributed deployment | ❌ | Planned (E5 [#104](https://github.com/adrianoggm/Onion-FL/issues/104), E6 [#105](https://github.com/adrianoggm/Onion-FL/issues/105)) |
| Tests | ✅ | 657 pass and 20 skip locally: the skips need `data/` or an MQTT broker. The CI starts a broker, so only the data-dependent ones skip there |

### What the results can and can't support today

- **No result of the new framework on real data is committed.** The datasets are not in git, and the framework was developed without them. Everything that trains or evaluates learning has a real-data test that skips without `data/` ([docs/RULES.md](docs/RULES.md)). Protocol tests use a stub trainer that learns nothing.
- **The reference SWELL setup is reproduced, not re-run.** `experiments/swell_reference.yaml` has the same subjects per fog, the same held-out test subjects and the same training schedule as the old reference runs. Running it needs `data/SWELL`.
- **Legacy numbers stay legacy.** [§7](#7-results-from-before-the-redesign) lists the baselines from before the redesign with their caveats. Some of them predate the `blok` leak fix.
- **The real-data-only rule holds.** Two old preprocessing scripts once wrote `np.random` values under the real SWELL file names. Both are deleted; if either ran on your machine, restore `data/SWELL/` from the original download. The dataset card (`onion_fl data inspect`) flags any feature whose correlation with the label is above 0.95.

---

## 2. Quick start

### Install

Python 3.11. The repository uses a `.venv` created with [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv --python 3.11
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -e ".[dev]"        # add ",analysis" for XGBoost
```

Or run `just install-dev`. The command line is `onion_fl` (also `python -m onion_fl`).

### Put the data in place

The data is not in git. Download each dataset from its source and place it where its descriptor in `datasets/` expects it:

| Dataset | Path | Descriptor |
|---|---|---|
| SWELL-KW | `data/SWELL/3 - Feature dataset/per sensor/*.csv` | `datasets/swell.yaml` (computer file; facial, posture and physiology as options), `datasets/swell_physiology.yaml` |
| SWEET | `data/SWEET/{sample_subjects,selection1/users,selection2/users}/<user>/` | `datasets/sweet.yaml` |
| WESAD | `data/WESAD/S<n>/S<n>.pkl` | `datasets/wesad.yaml` |

### Run

```bash
onion_fl data inspect swell                   # dataset card: subjects, classes, missing values, leak warnings
onion_fl topology show topologies/four_fogs.yaml
onion_fl plan experiments/mix_ab.yaml         # composition per fog and groups per link, no training
onion_fl run experiments/mix_ab.yaml --workers 3
onion_fl report experiments/mix_ab.yaml --by topology_id --by scenario
onion_fl baseline experiments/mix_ab.yaml     # LR, RF, XGBoost on the same test subjects
```

For a real run over MQTT, start the stack (`just docker-up`, [docker/README.md](docker/README.md)), point the links at the broker and run `onion_fl run <experiment> --mode real`.

---

## 3. How it works

The full design is in [docs/architecture.md](docs/architecture.md). In short:

```mermaid
flowchart TD
    cloud["cloud<br/>global · coordinator"]
    fog_a1["fog_a1<br/>fog · aggregator"]
    fog_a2["fog_a2<br/>fog · aggregator"]
    fog_b1["fog_b1<br/>fog · aggregator"]
    fog_b2["fog_b2<br/>fog · aggregator"]
    fog_a1 -->|mqtt/json/wifi| cloud
    fog_a2 -->|mqtt/json/wifi| cloud
    fog_b1 -->|mqtt/json/wifi| cloud
    fog_b2 -->|mqtt/json/wifi| cloud
    edges_fog_a1(["edges (edge)"]) -->|mqtt/json/4g| fog_a1
    edges_fog_a2(["edges (edge)"]) -->|mqtt/json/4g| fog_a2
    edges_fog_b1(["edges (edge)"]) -->|mqtt/json/4g| fog_b1
    edges_fog_b2(["edges (edge)"]) -->|mqtt/json/4g| fog_b2
```

*`topologies/four_fogs.yaml`, drawn with `onion_fl topology show --mermaid`; the graphs are in [docs/diagrams/](docs/diagrams/).*

| Layer | What it does |
|---|---|
| `core` | `Message` and codecs (`json`, `npz`), `Node` (a state machine) and `Context` (its only way out), topologies and identifiers, plugin registries |
| `data` | Descriptors → cache → `SubjectData` per subject → roles (test, val, train, local_val, clients, scaling fitted on train only) → placement under the leaf aggregators |
| `learning` | Modular model with namespaced keys (`adapter.<ds>`, `trunk`, `head.<task>`), sharing scopes, aggregators, server optimizers, trainers, inits, metrics |
| `roles` | Coordinator, aggregators and edges. Every round sends each link only the groups its scope lets through; weights travel per key, so a tree of FedAvg equals a flat FedAvg |
| `runtime` | `SimRuntime` (virtual clock) and `RealRuntime` (wall clock); both drive the same nodes |
| `transports` | `memory` and `mqtt` (one inbox per node, QoS and broker per link) |
| `observability` | Events, run identity and signature, diagnostics, analysis, report, sinks |
| `experiment` | Config, sweeps, the dry-run plan, the runner and the CLI |

Every experimental axis is a plugin chosen by name in the config, and new ones are registered without touching the core: placement, transport, codec, aggregator, trainer, server optimizer, sharing, model, init, metric, diagnostic, participation, staleness and sink. `onion_fl schema` exports the config's JSON Schema with the catalogue of every plugin and its parameters.

---

## 4. Experiments

An experiment names a topology, the data, the learning setup, the rounds, the evaluation and the runtime, plus seeds and a sweep. `experiments/mix_ab.yaml` takes SWELL and SWEET over four fogs from segregated (α = 0) to fully mixed (α = 1):

```yaml
name: mix_ab
topology: four_fogs
data:
  datasets: {swell: {}, sweet: {options: {label: binary}}}
  roles: {test: 0.2, val: 0.1, local_val: 0.2, scaler: global, seed: 0}
  placement: {name: mixing, alpha: 0.0}
learning:
  model: {name: modular_mlp, adapter_width: 64, trunk_hidden: [64, 32]}
  sharing: fedavg
  trainer: {name: standard, local_epochs: 1, lr: 0.001}
rounds: 20
evaluation: {metrics: [loss, accuracy, macro_f1], edge: {every: 5}, aggregators: {every: 5}, global: {every: 1}}
seeds: [0, 1, 2]
sweep: {data.placement.alpha: [0.0, 0.5, 1.0]}
```

- **Scenarios.** Each combination of the swept values, times each seed, is one run. Its `config_id` hashes the validated config without the seed, so the seeds of a scenario group together.
- **Subject roles.** They use their own seed, so the test subjects are the same in every scenario and seed.
- **Validation.** Errors name their exact path, for example `learning.trainer: plugin 'standard': invalid parameters: lr …`.

Each run writes `runs/<run_id>/`:

| File | Content |
|---|---|
| `run.json` | Identifiers (`topology_id`, `config_id`, `data_id`, `code_version`), resolved config, host, start and end, status (`running`, `finished`, `incomplete`, `failed`) and `run_hash` |
| `events.jsonl` | Every event: messages, rounds, training, evaluation, diagnostics, data composition |
| `summary.json` | Rounds, traffic, failures and the last scores at the root |
| `model.npz` | The final global model, loadable as a `checkpoint` init |

`run_id` is `<UTC date>-<sha12>`, unique even when the same config runs in parallel. `run_hash` covers `run.json`, the events, the summary and the model, so any later change is detected (`verify_run`). A result is cited with `topology_id` + `config_id` + `run_id` + `run_hash`.

---

## 5. Datasets

| Dataset | Label strategies | Unit | Notes |
|---|---|---|---|
| **SWELL-KW** | `binary` (N vs T, I, R), `binary_no_r` | One row per minute and subject | The computer file by default; facial, posture and physiology are optional joins. `swell_physiology.yaml` is physiology only |
| **SWEET** | `binary` (stress ≥ 2), `ordinal` (1–5), `three_class` | One self-report matched to the feature minute | `selection` option: `sample_subjects`, `selection1/users`, `selection2/users`; subjects with fewer than 5 samples are dropped |
| **WESAD** | `binary` (baseline vs stress), `three_class` (+ amusement) | 60 s windows, 50% overlap, five statistics per channel | `location`: wrist (default) or chest; `signals` option |

Missing values stay missing in the cache. Imputation, scaling and the removal of constant features are fitted on the training subjects only. The descriptors list their unverified assumptions in their comments, for example which keys join the SWELL modalities and how wrist signals are resampled. `tests/test_datasets_descriptors.py` compares each descriptor with the loader from before the redesign, and it runs as soon as `data/` is present.

---

## 6. Observability

Every runtime event is enriched into one schema and written to `events.jsonl`, the source of truth:

```
{t_virtual, t_wall, run_id, topology_id, scenario, seed, round, level, node, role, kind, name, value, tags}
```

- **Evaluation.**
  - Edges score the received and the trained model on their `local_val`.
  - Aggregators combine their children's scores by samples, and score the zone model on their `val` evaluators.
  - The coordinator scores the global model on the `test` evaluators, per dataset.
- **Diagnostics at every aggregator.**
  - Divergence: the cosine and L2 dispersion of the children's updates per parameter group.
  - Conflict between datasets: the cosine of their mean updates in the shared groups.
  - Drift across rounds, participation (with late updates and time to quorum) and fairness of the edge scores per dataset.
  - Traffic per link and round, which the run recorder adds.
- **Analysis.** `load_runs("runs/").compare(level="fog", metric="accuracy", by=["topology_id"])` gives the mean ± 95% CI over seeds per round, plus the spread across the nodes of the level. `onion_fl report` builds an HTML page from it.
- **Live.**
  - The `prometheus` sink serves `onionfl_*` series on port 9464 for the Grafana dashboard.
  - The `otel` sink creates one span per send and per receive, linked by the message id. It exports them to the collector of the Docker stack.

---

## 7. Results from before the redesign

**Only results backed by a committed file are listed.** They are centralised baselines from the old scripts (removed in F7.2), and they live in `results/legacy/`, whose [INDEX.md](results/legacy/INDEX.md) gives each file's origin and caveats. Use `onion_fl baseline` to produce new ones on the current data layer.

### WESAD: subject-disjoint holdout

Source: `results/legacy/advanced_ml_results/wesad_baseline_results.json`. The subjects are split 7 train / 3 val / 5 test, and the 22 wrist features give 1,057 test windows.

| Model | Test accuracy | Macro-F1 | F1 (stress class) |
|---|---|---|---|
| Random Forest | 0.828 | 0.647 | 0.393 |
| SVM | 0.780 | 0.544 | 0.215 |
| Logistic Regression | 0.770 | 0.577 | 0.292 |

### Subject-level 5-fold cross-validation

Source: `results/legacy/subject_cv_results/subject_cv_summary.json` (2025-09-28). Values are mean ± std.

| Dataset | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| WESAD | Logistic Regression | 0.865 ± 0.079 | 0.854 ± 0.087 |
| WESAD | Random Forest | 0.768 ± 0.083 | 0.738 ± 0.081 |
| SWELL (computer modality) ⚠️ | Logistic Regression | 0.951 ± 0.009 | 0.946 ± 0.009 |
| SWELL (computer modality) ⚠️ | Random Forest | 0.989 ± 0.006 | 0.987 ± 0.008 |

⚠️ These SWELL rows predate the `blok` leak fix (commit `002246f`, 2025-10-25), when an experimental-block id was used as a feature. The last report after the fix (commit `791a397`, 2025-11-27; no longer in the tree) gave SWELL 0.669 ± 0.014 (LR) and 0.670 ± 0.014 (RF) accuracy.

### SWELL: four-modality holdout

Source: `results/legacy/advanced_ml_results/swell_baseline_results.json` (2025-12-04): a random 50,000-row sample, 163 features, a subject-disjoint 50/20/30 split.

| Model | Test accuracy | Macro-F1 |
|---|---|---|
| Random Forest | 0.573 | 0.572 |
| Logistic Regression | 0.544 | 0.544 |
| Linear SVM | 0.515 | 0.514 |

### SWEET selection1: three classes

102 subjects and 3,927 samples. The largest class holds **0.552** of them, and every model stays at that rate: the best XGBoost reaches 0.551 ± 0.026 over subject 5-fold (`results/legacy/baseline_models/sweet/training_report.json`), and macro-F1 stays between 0.24 and 0.34. On those 14 features no model learns more than the class prior.

---

## 8. Repository map

```
src/onion_fl/
├── core/            message, codec, node, context, topology, registry, ids
├── data/            contract, ingest (readers, steps), cache, roles, placement
├── learning/        model, sharing, aggregators (+ server optimizers), trainers (+ inits), metrics
├── roles/           coordinator, aggregator, edge; round policies; federation builder
├── runtime/         sim (virtual clock), real (wall clock), network, devices
├── transports/      memory, mqtt
├── observability/   events, run, diagnostics, analysis (+ report), sinks
├── experiment/      config, sweep, runner, real (processes), cli
├── baselines.py     LR, RF, XGBoost and subject cross-validation
└── datasets/        loaders from before the redesign, kept as the parity reference
datasets/            dataset descriptors (YAML)
topologies/          topologies (YAML)
experiments/         experiments (YAML): mix_ab, swell_reference
docker/              MQTT broker, OTEL collector, Jaeger, Prometheus, Grafana
docs/                architecture, rules, diagrams, release notes, the design spec
results/legacy/      results from before the redesign, with INDEX.md
scripts/             data extraction (SWEET ZIPs) and real-sample creation
tests/               pytest suite
```

---

## 9. Development and CI

```bash
just check              # ruff check, ruff format --check, pytest: what every PR needs
just test tests/test_roles_protocol.py
just test-cov
```

- **Style.** ruff, 88 columns, `from __future__ import annotations`. The ruff version is pinned in `pyproject.toml` and `.pre-commit-config.yaml`.
- **Rules.** [docs/RULES.md](docs/RULES.md): real data only for anything that trains or evaluates; subject-disjoint evaluation; meta columns are never features; results cite a committed artifact.
- **Workflow.** Each GitHub issue gets a `task/#N` branch from `develop` and a PR back into `develop`; `main` only receives tagged releases. Commits follow `type(scope): Imperative summary in English`. The procedure and a GitHub API helper are in [.claude/skills/tarea-github/SKILL.md](.claude/skills/tarea-github/SKILL.md).

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | PRs into `main` and manual dispatch | `ruff check`, `ruff format --check` and `pytest` on Python 3.11, with an MQTT broker for the real-runtime tests |
| `pr-review.yml` | PRs into `main` | Trivy filesystem scan |
| `codeql.yml` | PRs into `main` and manual dispatch | CodeQL analysis |
| Dependabot | Weekly | Updates opened against `develop` |

The workflows run only on release PRs into `main`, to save CI minutes; task PRs are gated by `just check` locally.

---

## 10. Known limitations

- **Unverified descriptors.** The descriptors were written without the raw files. Run `onion_fl data inspect` and the parity tests before citing results. Specific assumptions:
  - The SWELL modality joins assume shared `pp`, `blok`, `condition` and `timestamp` columns.
  - WESAD wrist signals are held at the label rate (700 Hz), so their statistics match the previous loader only approximately; chest signals match exactly.
  - The SWEET minute deduplication can keep a report that has no stress value.
- **Global evaluation with personal or zone heads.** With `fedper` or `zone` sharing, the global model has no trained heads. Use the edge and zone scores.
- **Security.** MQTT runs without TLS or authentication, the broker sees every update, and there is no secure aggregation or differential privacy.
- **Real-run latency** across machines needs NTP-synchronised clocks.
- **Manual deployments** across machines need the same data and cache on each machine, because every process rebuilds the scenario.
- **Compute in simulation** is modelled as samples per second (or the measured wall time), not as a device profile.

---

## 11. Roadmap

| Milestone | Content |
|---|---|
| v0.3.0 | Onion-FL Studio: topology library and editor, run monitor, comparisons between topologies and scenarios, tutorial with dry-run previews, `onion_fl serve` ([#103](https://github.com/adrianoggm/Onion-FL/issues/103)) |
| Later | gRPC and Flower transports and richer network emulation (E5 [#104](https://github.com/adrianoggm/Onion-FL/issues/104)); real distributed deployment (E6 [#105](https://github.com/adrianoggm/Onion-FL/issues/105)); secure aggregation, TLS and differential privacy |

---

## 12. Project history

| Period | Milestone |
|---|---|
| 2025-07 | First Flower FedAvg + Mosquitto prototype; fog-node aggregation; ECG5000 demo |
| 2025-09 | Real WESAD and SWELL loaders; real-data-only policy; ECG5000 dropped |
| 2025-10 | Meta columns (`blok`, …) excluded from features after the leak was found |
| 2025-11 | First working federated SWELL run; `clients/`, `brokers/`, `servers/` structure |
| 2025-12 | OpenTelemetry, Jaeger, Prometheus and Grafana; SWEET baselines |
| 2026-03 / 04 | `justfile`; `accept`/`strict` stale-update policy; pure-function refactor |
| 2026-10 | v0.2.0: redesign into a framework (spec in `docs/superpowers/specs/`), old runtime removed |

---

## 13. License and acknowledgments

Licensed under the Apache License 2.0; see [LICENSE](LICENSE).

- **Datasets:** WESAD (Schmidt et al.), SWELL-KW (Koldijk et al.) and SWEET.
- **Software:** [PyTorch](https://pytorch.org), [Eclipse Mosquitto](https://mosquitto.org), [paho-mqtt](https://eclipse.dev/paho/), [OpenTelemetry](https://opentelemetry.io), [Prometheus](https://prometheus.io), [Grafana](https://grafana.com), [Jaeger](https://www.jaegertracing.io) and [pydantic](https://docs.pydantic.dev).
