# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Onion-FL is a framework for hierarchical federated learning experiments (edge → fog → … → cloud) with a virtual-clock simulator and real runs over MQTT. Package `onion_fl` (`src/` layout), CLI `onion_fl`. Datasets: SWELL, SWEET, WESAD (in `data/`, not in git).

- **Design.** `docs/architecture.md` describes what is built; the spec it implements, with the decisions, is `docs/superpowers/specs/2026-10-02-onion-fl-framework-design.md` (Spanish). Read the relevant part before structural changes.
- **Tracking.** Work is tracked as GitHub issues; next come E5 (gRPC/Flower transports, #104) and E6 (deployment, #105). The Studio has its own spec, `docs/superpowers/specs/2026-10-03-onion-fl-studio-design.md`.
- **Legacy.** `results/legacy/` holds results from before the redesign, with caveats in its `INDEX.md`.

## Commands

The repo `.venv` (Python 3.11, created with uv) is the environment. The global Python lacks `onion_fl`.

```bash
just install-dev                      # uv venv + CPU torch + -e ".[dev]"
just check                            # ruff check, ruff format --check, pytest: what every PR needs
.venv/Scripts/python -m pytest tests/test_roles_protocol.py -q -k quorum    # one test (bin/ on Linux)
.venv/Scripts/python -m onion_fl plan experiments/mix_ab.yaml               # dry run, no training
```

- pytest runs with `filterwarnings = error` (only `UserWarning`/`DeprecationWarning` are ignored), so any new warning class fails the suite.
- **Skipped tests.** Some tests need `data/` (`data/SWELL`, `data/samples/*.pkl`…) or an MQTT broker (`ONIONFL_MQTT=host:port`, default `localhost:1883`), and skip without them.
  - README §2 says where to download SWELL and WESAD. `data/samples/*.pkl` is built from them by `scripts/create_real_samples.py` (on Windows, with `PYTHONIOENCODING=utf-8`: it prints emoji).
  - Local broker: `docker run -d -p 1883:1883 eclipse-mosquitto:2 mosquitto -c /mosquitto-no-auth.conf`.
  - In Git Bash, prefix it with `MSYS_NO_PATHCONV=1`.
- ruff is pinned to 0.16.x in `pyproject.toml` and `.pre-commit-config.yaml`; keep them in sync. Markdown is excluded from ruff.
- **CI** runs only on PRs into `main`, with a Mosquitto broker. Task PRs into `develop` are gated by `just check` locally.

## Architecture

| Package | What lives there |
|---|---|
| `core` | `Message`/`Payload` and codecs, `Node` + `Context` (the only way a node acts), `Topology` (`topology_id`, `to_graph`), `Registry`, `ids` |
| `data` | `SubjectData` contract, declarative `ingest` (readers, steps, `{option}` placeholders, `when`), signed `cache`, subject `roles`, `placement` plugins |
| `learning` | `modular_mlp` with namespaced keys, sharing scopes, per-key aggregators and server optimizers, trainers and inits, metrics |
| `roles` | `Coordinator`, `Aggregator`, `Edge` state machines; round policies; `build_federation` (topology + edges → nodes and links on a runtime) |
| `runtime` | `SimRuntime` (virtual clock, link channels, compute and availability models), `RealRuntime` (wall clock, hosted subset) |
| `transports` | `memory`, `mqtt` |
| `observability` | Event schema and JSONL, `Run` (`runs/<run_id>/`, `run_hash`), diagnostics, `load_runs`/`compare`/report, Prometheus and OTEL sinks |
| `experiment` | Config (pydantic), sweeps, `plan`/`run_scenario`, real launcher (one process per aggregator), CLI |
| `studio` | `onion_fl serve`: FastAPI over the repository's files, dry-run previews, a dependency-free SPA in `static/` (DOM built with `h()`, never `innerHTML`) |

Things that span several files:

- **Every axis is a plugin.** Registries map a name, or `pkg.mod:Name`, to a factory with a pydantic params model. The config validates through them, and `onion_fl schema` exports them for the front. A new plugin is one decorated class.
- **Keys decide everything.** Parameter keys are namespaced (`adapter.<ds>`, `trunk`, `head.<task>`).
  - `group_of` maps a key to its group, and `SharingPolicy.scope_of` gives the group's scope.
  - `keys_crossing` says what travels on each link; `keys_held_at` says what an aggregator keeps.
  - Aggregators combine per key, sorted by sender; weights are samples per key.
- **Round 1 bootstraps.** The coordinator's first `global_model` carries the full initial state (`meta.bootstrap`), so aggregators can seed the zone groups they keep without building a model.
- **Determinism.**
  - Each node keeps one RNG stream for the whole run (`node_rng(seed, id)`).
  - Roles have their own seed, so test subjects are identical across scenarios.
  - The placement uses the run seed.
  - `tests/test_runtime_equivalence.py` checks that sim, the in-process real runtime and multi-process MQTT give the same final model.
- **Lifecycle.**
  - Children repeat `hello` until acknowledged.
  - When the coordinator finishes, a `control` stop travels down the tree.
  - The real launcher merges the per-process event files and signs the run.
  - A run whose coordinator never finished is `incomplete`.
- **Observability contract.** `docker/grafana/.../onion-fl.json` may only query series and labels that `PrometheusSink` exports; `tests/test_grafana_dashboard.py` checks it, so update both together.
- **Old loaders.** `onion_fl.datasets` keeps the loaders from before the redesign only as the parity reference for the descriptors in `datasets/`.

## Project rules (`docs/RULES.md`)

- **No synthetic data for ML.** Anything that trains or evaluates learning uses real extracts and skips without them. Protocol tests use the `stub` trainer, optionally with seeded `noise`, plus stub scorers. Format fixtures (small CSVs) are fine for parsing tests.
- **Subject-disjoint evaluation.** Test subjects are reserved per dataset. Imputation, scaling and constant-feature removal are fitted on the training portions only.
- **Meta columns are never features.** `blok`, timestamps and subject IDs must not become features: descriptors exclude them explicitly, and the subject and label source columns are always excluded. `blok` inflated SWELL baselines to ~0.99 before commit `002246f`.
- **Results cite a committed artifact,** signed with `topology_id`, `config_id`, `run_id` and `run_hash`. Keep the README status honest about what has and hasn't been run on real data.
- **Workflow:**
  - Each issue gets a `task/#N` branch from `develop` and a PR back into `develop`; `main` only gets tagged releases.
  - Commits follow `type(scope): Imperative summary in English`, are atomic and carry no `Co-Authored-By`.
  - Use the project skill `tarea-github` (`.claude/skills/tarea-github/scripts/gh.py`, GitHub API via the git credential, no `gh` CLI) for issues, PRs, merges and releases.
  - `git rm` stages at once, so check `git diff --cached --name-only` before each commit.
- **Style:** ruff, 88 columns. Modules start with `from __future__ import annotations`, often above the docstring (E402 is ignored for this). Plugin titles and descriptions are in Spanish (the front's language); code, comments and docs are in English.
- **Python ≥ 3.11.**
