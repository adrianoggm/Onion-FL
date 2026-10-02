# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Hierarchical (edge → fog → cloud) federated learning for stress detection, built on Flower + MQTT. Package name is `flower_basic` (`src/` layout; renaming to `onion_fl` is backlog item F1.1). Active datasets: **SWELL** (main workflow), **SWEET**, **WESAD**.

**A redesign is in progress.** Read the spec before structural changes: `docs/superpowers/specs/2026-10-02-onion-fl-framework-design.md`. It turns this into a transport-agnostic framework: state-machine nodes, a virtual-clock simulator plus MQTT, a modular model with per-key aggregation, declarative dataset ingestion, and pluggable placement, sharing and aggregation. Work is tracked as GitHub issues #73–#105 (milestones v0.2.0–v0.5.0; issue titles carry the phase, e.g. `[F2.3]`). Phase 0 (#73–#76) covers clean-up and tooling; check `gh.py issues --milestone v0.2.0` for what is integrated. The architecture below is the current, pre-redesign code.

`results/legacy/` holds committed experiment outputs from before the redesign. Its `INDEX.md` lists their caveats.

## Commands

```bash
pip install -e ".[dev]"               # add ,analysis for scripts/ (matplotlib, seaborn, xgboost)

python -m pytest                      # full suite; no MQTT broker needed (MQTT is mocked)
python -m pytest tests/test_runtime_protocol.py::test_name -q   # single test
python -m pytest -m "not slow"        # markers: slow, integration

ruff check .                          # CI gate
ruff format --check .                 # CI gate
just format                           # ruff check --fix + ruff format
```

- pytest runs with `filterwarnings = error` (only `UserWarning`/`DeprecationWarning` are ignored), so any new warning class fails the suite.
- Tests that need real data (`data/SWELL`, `data/WESAD`, `data/samples/*.pkl`) skip when it's missing. `data/` and `federated_runs/` are gitignored.
- ruff is pinned to 0.16.x in `pyproject.toml` and `.pre-commit-config.yaml`; keep them in sync. Markdown files are excluded from ruff.
- The `justfile` is the task runner. Its recipes use bash (`source .venv/bin/activate`, `pgrep`), so on Windows run them from Git Bash or WSL.

### Running the federated stack

```bash
just docker-up        # MQTT :1883, OTEL collector :4320, Jaeger :16686, Prometheus :9090, Pushgateway :9091, Grafana :3000
just swell-prepare-physio   # scripts/prepare_swell_federated.py -> federated_runs/swell/<run>/manifest.json + NPZ splits
just swell-launch-physio    # scripts/run_architecture_from_config.py --config ... --manifest ... --launch
just stop-all
```

`run_architecture_from_config.py --plan-only` prints the exact per-process commands without launching anything. It's the quickest way to see how a YAML in `configs/` turns into processes. It supports SWELL only. SWEET has its own launcher: `scripts/run_sweet_architecture.py --config configs/sweet_architecture_5nodes.yaml --dispatch-config --launch`.

## Architecture (current code)

### Topology and message flow

Each role runs as its own OS process (`python -m flower_basic.<pkg>.<module>`). Each process is configured through CLI flags plus env vars (`MQTT_BROKER`, `MQTT_PORT`, `MQTT_TOPIC_*`, `MQTT_REGION`, `FOG_K_MAP`):

```
clients/<ds>.py  --MQTT fl/updates-->  brokers/fog.py (buffers K updates per region, weighted avg)
                 --MQTT fl/partial-->  clients/fog_bridge_<ds>.py (Flower NumPyClient, one per fog)
                 --Flower gRPC----->   servers/<ds>.py (FedAvg strategy)
                 --MQTT fl/global_model--> clients (next round) + broker (round tracking)
```

The fog bridge is what plugs the MQTT world into Flower. `fit()` waits for a partial aggregate from its region and returns it to the server as if it had trained it.

### Code layout: shared base + dataset-specific leaf

| Role | Shared logic | Leaves |
|---|---|---|
| Client | `clients/federated_base.py` `FederatedMQTTClientBase` (round loop, wait for global, publish) | `clients/swell.py`, `clients/sweet.py` |
| Fog bridge | `clients/fog_bridge_base.py` `BaseFogBridgeClient` | `fog_bridge_swell.py`, `fog_bridge_sweet.py` |
| Server | `servers/federated_base.py` `FederatedMQTTStrategyBase(FedAvg)` | `servers/swell.py`, `servers/sweet.py` |
| Broker | `brokers/federated_base.py` `handle_client_update(...)` + `BrokerConfig`/`BrokerCallbacks` | `brokers/fog.py`, `brokers/sweet_fog.py` |

Brokers are module-level functions with module state, not classes. Each broker module builds a `BrokerConfig` and callbacks, then delegates to `handle_client_update`.

Keep pure functions with I/O at the edges:
- `runtime_protocol.py` owns every MQTT payload build/decode (`build_*_payload`, `decode_*_message`, envelopes). Change wire formats only there.
- `training/local.py` contains pure train/eval loops. `datasets/federated_common.py` contains split/manifest loading.
- `federated_architecture.py` pipeline: `parse_architecture_config` → `apply_manifest_paths` / `plan_manifest_application` → `resolve_runtime_architecture` → `plan_runtime_commands`. `plan_*` functions do no I/O. `materialize_*` functions write files.

### Things that span multiple files

- **Manifest drives spawning.** For SWELL, clients are rebuilt from `manifest.json` (`fog_<id>/subject_<id>/{train,val,test}.npz`). Subjects with no train data are skipped. A fog node's `k` is clamped to its number of spawned clients. `model.input_dim` comes from the manifest or the first `train.npz`.
- **Parameter names must match** across client, bridge and server. Weights travel as name→array dicts and are ordered by name, which is why each dataset has a single model class (`SwellMLP`, `SweetMLP`) that all three roles import.
- **Stale-update policy** (`orchestrator.stale_update_policy`: `accept` | `strict`, see `configs/federated_architecture_{accept,strict}.yaml`). The broker learns the latest round from `fl/global_model` and expects `latest + 1`. `strict` drops stale and future updates. `accept` buffers them and records staleness metrics.
- **Observability.** OTEL trace context is injected into MQTT payloads, and spans are linked across processes (`telemetry.start_linked_*_span`). Each process serves Prometheus `/metrics` (`METRICS_PORT_<COMPONENT>` / `METRICS_PORT`). Clients and server also push to Pushgateway, but brokers deliberately don't, so live `/metrics` stays the single source of broker metrics. `tests/test_grafana_dashboard.py` validates `docker/grafana/.../flower-fl.json`, so update both together.
- **Multi-spawn guard.** Several fixes target duplicate child processes and orphaned processes. Components register `atexit`/signal cleanup. `run_architecture_from_config.py` stops everything once the server exits; `run_sweet_architecture.py` doesn't. Keep that cleanup intact when touching entrypoints. It has little test coverage: `tests/test_demo_launchers.py` only checks the module plans the launchers build.

## Project rules (`docs/RULES.md`)

- **No synthetic data for ML.** Don't use `np.random`-generated datasets or fake features for training or evaluation. Protocol/runtime tests may use a stub trainer that learns nothing. Tests that train or evaluate use real extracts (`data/samples/`) and skip when absent. Mocking MQTT or Flower is fine.
- **Splits must be subject-disjoint** for global test. Only the `global` split strategy with `test_assignments` (`configs/swell_federated_10runs.yaml`) guarantees this. The code default, `per_subject`, splits each subject's own samples, so it doesn't.
- **Meta columns are never features** (`blok`, `timestamp`, subject IDs…). `blok` inflated SWELL baselines to ~0.99 before commit `002246f`.
- **README results must cite a committed artifact.** `federated_runs/` is gitignored, so no federated result is currently backed by a file. Keep the README "Known issues" section in sync when fixing them.
- **Workflow:**
  - Each GitHub issue gets a `task/#N` branch from `develop`, merged back by PR. `main` only gets tagged working versions.
  - Commit messages follow `type(scope): Imperative summary in English`. Commits are atomic and carry no `Co-Authored-By`.
  - For any issue, branch, PR, merge, release or backlog work, use the project skill `tarea-github` (`.claude/skills/tarea-github/`). Its `scripts/gh.py` talks to the GitHub API with the git credential, so no `gh` CLI is needed.
- **Style:** ruff, 88 columns. Modules start with `from __future__ import annotations`, often above the docstring (E402 is ignored for this). Log, error and CLI help strings are mostly in Spanish, so match the surrounding file.
- **Python ≥ 3.11**, which CI uses.
