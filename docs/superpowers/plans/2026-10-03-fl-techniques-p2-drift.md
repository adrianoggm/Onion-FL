# FL techniques P2 (non-IID drift) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SCAFFOLD, FedNova, FedDyn and MOON can be chosen by name, paired with the server optimizer they need, swept, and compared on SWELL + WESAD (issue #148).

**Architecture:**
- **Auxiliary state (spec §3.1).**
  - Trainers return auxiliary arrays named `<algo>/<parameter key>`, which ride in the update `Payload` and follow their parameter's sharing scope.
  - Aggregators combine them per key like any array, and server optimizers replace them instead of stepping them. Diagnostics ignore them.
- **Round statistics (spec §3.2).**
  - Edges report `train_steps` (their local optimizer steps) and `train_edges`.
  - The collectors reduce every `train_*` metric up the tree.
  - The coordinator hands the round's statistics, plus `edges_total`, to the server optimizer.
- **Pairing.** `learning.server_optimizer` lets an experiment pick the root's optimizer. FedNova and FedDyn are trainer + optimizer pairs that `plan` checks in both directions.
- **MOON.** It needs `ModularMLP.features` (spec §3.5).

**Tech Stack:** Python 3.11, PyTorch, NumPy, pydantic 2.13, pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-fl-techniques-design.md` (§3.1, §3.2, §3.5 and §4 P2).

**Scope:** the `staleness` statistic of §3.2 has no consumer until FedAsync (#150) and moves there.

## Global Constraints

- **Environment.** Python ≥ 3.11; run everything with `.venv/Scripts/python` and `.venv/Scripts/ruff`.
- **Lint and warnings.** ruff with 88 columns. Modules start with `from __future__ import annotations`. pytest runs with `filterwarnings = error`.
- **Language.** Plugin `title`, `description`, `explain` and parameter descriptions are in Spanish. Code, comments and docs are in English.
- **No synthetic data for ML (docs/RULES.md).** Tests that run an optimiser use `data/samples/swell_real_sample.pkl` and skip without it. Arithmetic on hand-written arrays and protocol tests with stub trainers are fine.
- **Stable `config_id`s.** A new config field must not change the `config_id` of existing experiments: when unset it stays out of the dump (`exclude_if`).
- **Evaluation never alters training.** Scoring must not change the training trajectory.
- **Commits and workflow.** Commit subjects are `type(scope): Imperative summary in English`, with no `Co-Authored-By` lines. Work happens on branch `task/#148`.
- **Results.** Edge tables are computed from the edges' own events and backed by a committed CSV, never from the cloud's combined scores, which a lossy round can lose.
- **Full check:** `.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check . && .venv/Scripts/python -m pytest -q`.

## Review Focus

1. **Server optimizers must not step auxiliary arrays.** FedAdam's momentum on a control variate would corrupt it. Pinned in Task 1.
2. **Auxiliary arrays must not reach diagnostics.** Divergence and drift per group must not mix in control variates. Pinned in Task 1.
3. **SCAFFOLD under `fedper`.** The heads' control variates never arrive (treated as zero) and never leave. Training still works. Pinned in Task 4.
4. **A wrong pairing fails at `plan`, in both directions,** with a message naming both sides: a FedNova trainer with `replace`, or a `fednova` optimizer with `standard`. Pinned in Task 3.
5. **FedNova without `train_steps`** fails with a clear error naming it, not a `KeyError`. Pinned in Task 5.

---

### Task 1: Auxiliary arrays travel with the update

**Files:**
- Modify: `src/onion_fl/learning/model.py` (`group_of`, new `is_aux`)
- Modify: `src/onion_fl/learning/trainers.py` (`TrainResult.aux`)
- Modify: `src/onion_fl/learning/aggregators.py` (`FedAvgM.apply`, `FedAdam.apply` pass aux keys through)
- Modify: `src/onion_fl/roles/nodes.py` (`Edge._train` adds `result.aux`; `_Collector._diagnose` strips aux keys)
- Test: `tests/test_learning_model.py`, `tests/test_learning_aggregators.py`, `tests/test_roles_protocol.py`, `tests/test_observability_diagnostics.py`

**Interfaces:**
- Produces:
  - `is_aux(key: str) -> bool`: true when `"/" in key`;
  - `group_of("scaffold/trunk.0.weight") == "trunk"`;
  - `TrainResult.aux: Mapping[str, np.ndarray]`, defaulting to `{}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_learning_model.py` (import `group_of`, `is_aux` from `onion_fl.learning.model`):

```python
def test_auxiliary_keys_belong_to_their_parameter_group() -> None:
    assert is_aux("scaffold/trunk.0.weight") and not is_aux("trunk.0.weight")
    assert group_of("scaffold/trunk.0.weight") == "trunk"
    assert group_of("fednova/adapter.swell.0.weight") == "adapter.swell"
```

`tests/test_learning_aggregators.py` (import `server_optimizers` if absent):

```python
@pytest.mark.parametrize("name", ["fedavgm", "fedadam"])
def test_server_optimizers_replace_auxiliary_arrays(name: str) -> None:
    optimizer = server_optimizers.create(name)
    global_state = {"w": np.zeros(2), "scaffold/w": np.zeros(2)}
    aggregated = {"w": np.ones(2), "scaffold/w": np.full(2, 7.0)}

    out = optimizer.apply(global_state, aggregated)

    np.testing.assert_array_equal(out["scaffold/w"], [7.0, 7.0])
```

`tests/test_roles_protocol.py` (module level, next to the other helper trainers):

```python
class AuxStub:
    """The stub trainer plus one auxiliary array per round: what it last received + 1."""

    def __init__(self) -> None:
        self.stub = trainers.create("stub", {"shift": 1.0})
        self.seen: list[np.ndarray | None] = []

    def train(self, model, data=None, received=None, ctx=None):
        result = self.stub.train(model, data, received, ctx)
        last = (received or {}).get("algo/trunk.0.weight")
        self.seen.append(None if last is None else np.asarray(last).copy())
        base = np.zeros_like(received["trunk.0.weight"]) if last is None else last
        return dataclasses.replace(result, aux={"algo/trunk.0.weight": base + 1})


def test_auxiliary_arrays_go_up_and_come_back_down() -> None:
    trainer = AuxStub()
    edges = {"fog_0": [EdgeSpec("e1", model(A), trainer=trainer)]}

    federation = run(tree(1), edges, rounds=3)

    assert trainer.seen[0] is None
    assert float(trainer.seen[1].mean()) == pytest.approx(1.0)
    assert float(trainer.seen[2].mean()) == pytest.approx(2.0)
    assert float(federation.coordinator.state["algo/trunk.0.weight"].mean()) == pytest.approx(3.0)
```

(add `import dataclasses`; `model`, `A`, `tree`, `run` are this file's existing helpers)

`tests/test_observability_diagnostics.py`: add a test that a `RoundView` built by a collector excludes aux keys. Use the AuxStub federation above with diagnostics on, and assert that no `diagnostic.*` event carries a `group` tag for an aux key. The groups are parameter groups, so the check is that every value in the diagnostics run equals what the same run gives without aux:

```python
def test_auxiliary_arrays_do_not_reach_the_diagnostics() -> None:
    from tests.test_roles_protocol import AuxStub, A, model, run, tree
    from onion_fl.roles import EdgeSpec
    from onion_fl.learning.trainers import trainers

    def diagnostics(trainer):
        edges = {"fog_0": [EdgeSpec("e1", model(A), trainer=trainer), EdgeSpec("e2", model(A), trainer=trainers.create("stub", {"shift": 2.0}))]}
        federation = run(tree(1), edges, rounds=2)
        return sorted(
            (e["node"], e["name"], e["round"], round(float(e["value"]), 9), tuple(sorted((e["tags"] or {}).items())))
            for e in federation.runtime.events
            if e["name"].startswith("diagnostic.divergence")
        )

    assert diagnostics(AuxStub()) == diagnostics(trainers.create("stub", {"shift": 1.0}))
```

If importing from another test module is awkward under `pytest`'s rootdir, put this test in `tests/test_roles_protocol.py` instead, next to the AuxStub. It still exercises the collector's `_diagnose`.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_learning_model.py tests/test_learning_aggregators.py tests/test_roles_protocol.py -q -k "auxiliary"`
Expected: `ImportError` for `is_aux`; then, once it imports:
- `group_of` raises for `scaffold/...`;
- FedAdam returns a stepped array, not 7;
- `TrainResult` has no `aux`.

- [ ] **Step 3: Implement**

`model.py`:

```python
AUX = "/"


def is_aux(key: str) -> bool:
    """An auxiliary array ``<algorithm>/<parameter key>`` (control variates, …)."""
    return AUX in key


def group_of(key: str) -> str:
    """Parameter group of a key: ``adapter.<dataset>``, ``trunk``, ``trunk.<dataset>``, ``head.<task>``.

    An auxiliary key belongs to the group of the parameter it names.
    """
    key = key.split(AUX, 1)[-1]
    ...  # existing body unchanged
```

`trainers.py`:
- add `from dataclasses import dataclass, field`;
- in `TrainResult`, add `aux: Mapping[str, np.ndarray] = field(default_factory=dict)`, an auxiliary state sent with the update.

`aggregators.py`:
- import `is_aux` from `onion_fl.learning.model`;
- in `FedAvgM.apply` and `FedAdam.apply`, make the loop's first lines:

```python
        for key, value in aggregated.items():
            if is_aux(key):  # control variates and the like are replaced, never stepped
                new[key] = value
                continue
```

`nodes.py`:
- import `is_aux` from `onion_fl.learning.model`.
- In `Edge._train`, replace `arrays = state_arrays(self.model)` with `arrays = state_arrays(self.model) | dict(result.aux)`.
- In `_Collector._diagnose`, build the view from model keys only:

```python
        def model_part(state: State) -> State:
            return {k: v for k, v in state.items() if not is_aux(k)}

        fresh = {
            child: Contribution(c.source, model_part(c.state), c.weights)
            for child, c in fresh.items()
        }
        aggregated = model_part(aggregated)
```

and pass `previous=model_part(self.previous)`, `sent=model_part(self.sent)`, `received=model_part(getattr(self, "received", {}))` in the `RoundView(...)` call.

- [ ] **Step 4: Run the affected suites**

Run: `.venv/Scripts/python -m pytest tests/test_learning_model.py tests/test_learning_aggregators.py tests/test_roles_protocol.py tests/test_roles_evaluation.py tests/test_observability_diagnostics.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/learning/model.py src/onion_fl/learning/trainers.py src/onion_fl/learning/aggregators.py src/onion_fl/roles/nodes.py tests/
git commit -m "feat(learning): Carry auxiliary arrays with the updates

Arrays named <algorithm>/<parameter key> travel where their parameter
travels, are combined per key, are replaced (never stepped) by server
optimizers and are left out of the diagnostics."
```

---

### Task 2: Round statistics reach the server optimizer

**Files:**
- Modify: `src/onion_fl/roles/nodes.py` (`_train_metrics`, `Edge._train` metrics, `Coordinator._closed`)
- Modify: `src/onion_fl/learning/aggregators.py` (`apply(..., stats=None)` on every optimizer)
- Test: `tests/test_roles_protocol.py`

**Interfaces:**
- Produces:
  - the edge update metrics `train_steps` (= `TrainResult.batches`) and `train_edges` (= 1);
  - `_train_metrics` returns `train_loss` and `train_steps` as means weighted by examples, and `train_examples` and `train_edges` as sums;
  - every server optimizer is called as `apply(global_state, aggregated, stats)`, with `stats = round metrics | {"edges_total": edges_below()}`.

- [ ] **Step 1: Write the failing test** (`tests/test_roles_protocol.py`)

```python
class Recorder:
    """A server optimizer that keeps every stats it is given and replaces the state."""

    def __init__(self) -> None:
        self.stats: list[dict] = []

    def apply(self, global_state, aggregated, stats=None):
        self.stats.append(dict(stats or {}))
        return {**global_state, **aggregated}


def test_the_server_optimizer_gets_the_round_statistics(monkeypatch) -> None:
    recorder = Recorder()
    monkeypatch.setattr(
        "onion_fl.roles.federation.create",
        _wrap_create(recorder),
    )
    edges = {
        "fog_0": [edge("e1", shift=1, examples=1), edge("e2", shift=1, examples=3)],
        "fog_1": [edge("e3", shift=1, examples=4)],
    }

    run(tree(2), edges)

    (stats,) = recorder.stats
    assert stats["train_edges"] == 3
    assert stats["edges_total"] == 3
    assert stats["train_examples"] == 8
    assert stats["train_steps"] == pytest.approx(1.0)  # the stub reports one step
```

Monkeypatching `create` is brittle. Prefer extending `build_federation` with an optional `server_optimizer` argument that overrides the root setting, used only when given. Then the test passes `server_optimizer=recorder` to `run(...)` (the helper forwards `**kw` to `build_federation`), and `_wrap_create` is not needed. In `federation.py`:

```python
def build_federation(..., server_optimizer: Any = None) -> Federation:
    ...
        server_optimizer=server_optimizer
        or create(server_optimizers, root.settings.get("server_optimizer"), "replace"),
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_roles_protocol.py -q -k round_statistics`
Expected: FAIL, because `apply` is called without `stats`, or the stats lack `train_edges`/`train_steps`.

- [ ] **Step 3: Implement**

`nodes.py`, the `_train_metrics` replacement:

```python
def _train_metrics(reports: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """``train_*`` of the children: sums of examples and edges, the rest averaged by examples."""
    reports = [r for r in reports if r.get("train_examples")]
    examples = sum(r["train_examples"] for r in reports)
    if not examples:
        return {}
    out = {
        "train_examples": float(examples),
        "train_edges": float(sum(r.get("train_edges", 1.0) for r in reports)),
    }
    names = sorted(
        {k for r in reports for k in r if k.startswith("train_")} - set(out)
    )
    for name in names:
        holders = [r for r in reports if name in r]
        weight = sum(r["train_examples"] for r in holders)
        out[name] = float(sum(r[name] * r["train_examples"] for r in holders) / weight)
    return out
```

In `Edge._train`'s payload metrics, add `"train_steps": float(result.batches)` and `"train_edges": 1.0`.

`Coordinator._closed`:

```python
        stats = dict(metrics) | {"edges_total": float(self.edges_below())}
        self.state = self.server_optimizer.apply(
            self.state, _subset(aggregated.state, held), stats
        )
```

`aggregators.py`: give every optimizer's `apply` the signature

```python
    def apply(
        self,
        global_state: Mapping[str, np.ndarray],
        aggregated: Mapping[str, np.ndarray],
        stats: Mapping[str, float] | None = None,
    ) -> State:
```

(unused by `replace`, `fedavgm` and `fedadam`).

- [ ] **Step 4: Run the role and aggregator suites**

Run: `.venv/Scripts/python -m pytest tests/test_roles_protocol.py tests/test_roles_evaluation.py tests/test_learning_aggregators.py tests/test_runtime_equivalence.py -q`
Expected: all pass. The metric `round.train_loss` keeps its value, because `train_loss` is still the mean weighted by examples.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/roles/nodes.py src/onion_fl/roles/federation.py src/onion_fl/learning/aggregators.py tests/test_roles_protocol.py
git commit -m "feat(roles): Give the server optimizer the round statistics

Edges report their local steps; collectors reduce every train_* metric
(steps and loss averaged by examples, examples and edges summed) and the
coordinator adds edges_total."
```

---

### Task 3: Pick the server optimizer in the experiment, and check pairs

**Files:**
- Modify: `src/onion_fl/experiment/config.py` (`LearningConfig.server_optimizer`)
- Modify: `src/onion_fl/experiment/runner.py` (`resolve_topology`, `_check_local_groups` → `_check_learning`)
- Test: `tests/test_experiment.py`

**Interfaces:**
- Produces:
  - `LearningConfig.server_optimizer: PluginRef | None`; unset, it is left out of the dump and the topology's root setting applies;
  - a trainer may declare `server_optimizer: str` (the optimizer it needs);
  - an optimizer may define `check_trainer(name: str, trainer) -> None`, which raises `ValueError`;
  - `_check_learning(config, state)` raises `ConfigError` for a bad pair.

- [ ] **Step 1: Write the failing tests** (`tests/test_experiment.py`, validation section)

```python
def test_the_experiment_can_set_the_root_server_optimizer(workspace: Path) -> None:
    from onion_fl.experiment.runner import resolve_topology

    learning = experiment(workspace)["learning"] | {"server_optimizer": "fedadam"}
    config = parse_experiment(experiment(workspace, learning=learning))

    assert resolve_topology(config).root.settings["server_optimizer"] == "fedadam"


def test_an_unset_server_optimizer_stays_out_of_the_config(workspace: Path) -> None:
    config = parse_experiment(experiment(workspace))

    assert "server_optimizer" not in config.dump()["learning"]


@pytest.mark.parametrize(
    "trainer, optimizer, word",
    [("fednova", "replace", "fednova"), ("stub", "fednova", "fednova")],
)
def test_paired_trainers_and_optimizers_are_checked_both_ways(
    workspace: Path, trainer: str, optimizer: str, word: str
) -> None:
    learning = experiment(workspace)["learning"] | {
        "trainer": trainer,
        "server_optimizer": optimizer,
    }

    with pytest.raises(ConfigError, match=word):
        plan(parse_experiment(experiment(workspace, learning=learning)))
```

(The `fednova` trainer and optimizer arrive in Task 5. Write these tests now; they fail with "unknown plugin" until then, and Task 5's Step 4 runs them green. Until Task 5 the pairing logic is checked by Step 4 below with a registered dummy.)

Add to the same file a test that does not depend on Task 5. It uses a test-only trainer, registered through `monkeypatch` so that it disappears after the test:

```python
def test_a_trainer_that_needs_an_optimizer_is_checked(workspace: Path, monkeypatch) -> None:
    from onion_fl.core.registry import PluginSpec
    from onion_fl.learning.trainers import Stub, StubParams, trainers

    class Needy(Stub):
        server_optimizer = "fedadam"

    monkeypatch.setitem(
        trainers._specs, "needy_test", PluginSpec("needy_test", Needy, "t", "d", StubParams)
    )
    learning = experiment(workspace)["learning"] | {"trainer": "needy_test"}

    with pytest.raises(ConfigError, match="fedadam"):
        plan(parse_experiment(experiment(workspace, learning=learning)))
    ok = learning | {"server_optimizer": "fedadam"}
    assert plan(parse_experiment(experiment(workspace, learning=ok)))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_experiment.py -q -k "server_optimizer or needs_an_optimizer"`
Expected: FAIL, because `learning.server_optimizer` is an unknown field.

- [ ] **Step 3: Implement**

`config.py`, in `LearningConfig`:

```python
    server_optimizer: PluginRef | None = Field(
        None,
        description="Optimizador de servidor de la raíz; sin él, el de la topología",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )

    _server_optimizer = field_validator("server_optimizer")(
        lambda v: v if v is None else _plugin(server_optimizers, v)
    )
```

`runner.py`, in `resolve_topology` after the eval defaults loop:

```python
    if config.learning.server_optimizer is not None:
        root = next(n for n in general["nodes"] if n["parent"] is None)
        root["settings"]["server_optimizer"] = config.learning.server_optimizer
```

Rename `_check_local_groups` to `_check_learning` and extend it:

```python
def _name(ref: Any) -> str:
    return ref if isinstance(ref, str) else ref["name"]


def _check_learning(config: ExperimentConfig, state: Mapping[str, Any]) -> None:
    """The trainer fits the sharing (FedRep) and pairs with the root optimizer (FedNova)."""
    trainer = create(trainers, config.learning.trainer)
    name = _name(config.learning.trainer)
    patterns = list(getattr(trainer, "local_groups", ()))
    policy = create(sharing, config.learning.sharing)
    leaving = not_local(policy, param_groups(state), patterns)
    if leaving:
        raise ConfigError(...)  # existing message unchanged
    ref = resolve_topology(config).root.settings.get("server_optimizer") or "replace"
    optimizer_name = _name(ref)
    needed = getattr(trainer, "server_optimizer", None)
    if needed is not None and optimizer_name != needed:
        raise ConfigError(
            f"learning.trainer: {name} needs server_optimizer {needed!r}, "
            f"not {optimizer_name!r}; set learning.server_optimizer"
        )
    check = getattr(create(server_optimizers, ref), "check_trainer", None)
    if check is not None:
        try:
            check(name, trainer)
        except ValueError as exc:
            raise ConfigError(f"learning.server_optimizer: {exc}") from None
```

(import `server_optimizers` from `onion_fl.learning.aggregators`; `_initial_state` calls `_check_learning`.)

- [ ] **Step 4: Run the experiment suite**

Run: `.venv/Scripts/python -m pytest tests/test_experiment.py tests/test_cli.py -q`
Expected: all pass, except the two `fednova` parametrisations of `test_paired_trainers_and_optimizers_are_checked_both_ways`, which fail with "unknown … fednova" until Task 5. Mark them `@pytest.mark.xfail(reason="fednova arrives in Task 5", strict=True)` now, and remove the mark in Task 5.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/experiment/config.py src/onion_fl/experiment/runner.py tests/test_experiment.py
git commit -m "feat(experiments): Pick the root server optimizer and check pairs

learning.server_optimizer overrides the topology's root setting (out of
the dump when unset). Trainers that need an optimizer, and optimizers
that need a trainer, are checked before the first scenario runs."
```

---

### Task 4: The SCAFFOLD trainer

**Files:**
- Modify: `src/onion_fl/learning/trainers.py`
- Test: `tests/test_learning_trainers.py`

**Interfaces:**
- Consumes: `TrainResult.aux` (Task 1), the `_penalty` hook of `Standard`.
- Produces:
  - the trainer `scaffold` with `ScaffoldParams(StandardParams)`, whose `optimizer` defaults to `"sgd"`;
  - the auxiliary keys `scaffold/<parameter>` (the client's new control variate c_i⁺).

- [ ] **Step 1: Write the failing tests**

```python
@real
def test_scaffold_without_control_variates_is_sgd(swell) -> None:
    params = {"local_epochs": 1, "lr": 0.05, "optimizer": "sgd"}
    plain, corrected = build(), build()
    received = state_arrays(plain)

    trainers.create("standard", params).train(plain, swell, received, ctx())
    trainers.create("scaffold", params).train(corrected, swell, received, ctx())

    for key, value in state_arrays(plain).items():
        np.testing.assert_array_equal(value, state_arrays(corrected)[key])


@real
def test_scaffold_sends_its_new_control_variate(swell) -> None:
    model = build()
    received = state_arrays(model)
    trainer = trainers.create("scaffold", {"local_epochs": 1, "lr": 0.05})

    result = trainer.train(model, swell, received, ctx())

    after = state_arrays(model)
    for key, x in received.items():
        expected = (x - after[key]) / (result.batches * 0.05)  # c = c_i = 0
        np.testing.assert_allclose(result.aux[f"scaffold/{key}"], expected, rtol=1e-5)


@real
def test_scaffold_corrects_with_the_received_control_variate(swell) -> None:
    received = state_arrays(build())
    shifted = received | {
        f"scaffold/{k}": np.full_like(v, 0.1) for k, v in received.items()
    }
    plain, corrected = build(), build()
    params = {"local_epochs": 1, "lr": 0.05}

    trainers.create("scaffold", params).train(plain, swell, received, ctx())
    trainers.create("scaffold", params).train(corrected, swell, shifted, ctx())

    assert any(
        not np.array_equal(v, state_arrays(corrected)[k])
        for k, v in state_arrays(plain).items()
    )


@real
def test_scaffold_trains_when_the_heads_stay_on_the_edge(swell) -> None:
    model = build()
    received = {
        k: v for k, v in state_arrays(model).items() if not k.startswith("head.")
    }

    result = trainers.create("scaffold", {"lr": 0.05}).train(
        model, swell, received, ctx()
    )

    assert any(k.startswith("scaffold/head.") for k in result.aux)
```

Also add `"scaffold"` to `test_trainers_registry_lists_the_built_ins`.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_learning_trainers.py -q -k "scaffold or registry"`
Expected: FAIL with `unknown trainer plugin 'scaffold'`.

- [ ] **Step 3: Implement** (after `FedBABU`; `import dataclasses` at the top if absent)

```python
class ScaffoldParams(StandardParams):
    optimizer: Literal["adam", "sgd"] = Field(
        "sgd", description="La actualización de c_i supone SGD"
    )


@trainers.register(
    "scaffold",
    title="SCAFFOLD",
    description="Corrige cada gradiente con las variables de control: g − c_i + c.",
    params=ScaffoldParams,
    explain=(
        "c viaja con el modelo global y c_i se queda en el edge; el edge envía "
        "c_i⁺ = c_i − c + (x − y)/(K·η) y el agregado es el nuevo c (opción II "
        "de Karimireddy et al., 2020). Con participación parcial, c es la media "
        "de los participantes."
    ),
)
class Scaffold(Standard):
    Params = ScaffoldParams
    PREFIX = "scaffold/"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._c_i: dict[str, np.ndarray] = {}
        self._correction: dict[str, torch.Tensor] = {}

    def _penalty(
        self, model: nn.Module, received: Any, names: Sequence[str]
    ) -> torch.Tensor | float:
        # The gradient of <c − c_i, w> is c − c_i: the SCAFFOLD correction.
        params = dict(model.named_parameters())
        total: torch.Tensor | float = 0.0
        for name in names:
            if name in self._correction:
                total = total + (params[name] * self._correction[name]).sum()
        return total

    def train(
        self,
        model: nn.Module,
        data: Samples,
        received: Mapping[str, np.ndarray] | None = None,
        ctx: Any = None,
    ) -> TrainResult:
        received = dict(received or {})
        names = trainable(model, self.params.frozen)
        start = state_arrays(model)
        c = {
            n: np.asarray(received.get(self.PREFIX + n, np.zeros_like(start[n])))
            for n in names
        }
        c_i = {n: self._c_i.get(n, np.zeros_like(start[n])) for n in names}
        params = dict(model.named_parameters())
        self._correction = {
            n: torch.as_tensor(c[n] - c_i[n], dtype=params[n].dtype) for n in names
        }
        result = super().train(model, data, received, ctx)
        after = state_arrays(model)
        scale = result.batches * self.params.lr
        self._c_i = {
            n: c_i[n] - c[n] + (start[n] - after[n]) / scale for n in names
        }
        aux = {self.PREFIX + n: v.astype(start[n].dtype) for n, v in self._c_i.items()}
        return dataclasses.replace(result, aux=aux)
```

- [ ] **Step 4: Run the trainer tests**

Run: `.venv/Scripts/python -m pytest tests/test_learning_trainers.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/learning/trainers.py tests/test_learning_trainers.py
git commit -m "feat(learning): Add the SCAFFOLD trainer"
```

---

### Task 5: FedNova (trainer and server optimizer)

**Files:**
- Modify: `src/onion_fl/learning/trainers.py`, `src/onion_fl/learning/aggregators.py`
- Test: `tests/test_learning_trainers.py`, `tests/test_learning_aggregators.py`, `tests/test_experiment.py` (remove the xfail marks of Task 3)

**Interfaces:**
- Consumes: `TrainResult.aux` (Task 1), `stats["train_steps"]` (Task 2), pairing (Task 3).
- Produces:
  - the trainer `fednova`, with `server_optimizer = "fednova"`, which sends `fednova/<key>` = (y − x)/τ;
  - the server optimizer `fednova`: `x ← x + τ̄·d̄`. It drops the `fednova/*` keys from the state, and its `check_trainer` requires `fednova`.

- [ ] **Step 1: Write the failing tests**

`tests/test_learning_aggregators.py`:

```python
def test_fednova_with_equal_steps_is_fedavg() -> None:
    x = np.zeros(2)
    ys = [np.array([1.0, 2.0]), np.array([3.0, 6.0])]
    d = np.mean([(y - x) / 4 for y in ys], axis=0)

    out = server_optimizers.create("fednova").apply(
        {"w": x}, {"w": np.mean(ys, axis=0), "fednova/w": d}, {"train_steps": 4.0}
    )

    np.testing.assert_allclose(out["w"], np.mean(ys, axis=0))
    assert "fednova/w" not in out


def test_fednova_scales_the_normalised_update_by_the_mean_steps() -> None:
    out = server_optimizers.create("fednova").apply(
        {"w": np.ones(1)}, {"w": np.full(1, 9.0), "fednova/w": np.full(1, 0.5)},
        {"train_steps": 3.0},
    )

    np.testing.assert_allclose(out["w"], [2.5])  # 1 + 3·0.5


def test_fednova_names_the_missing_steps() -> None:
    with pytest.raises(ValueError, match="train_steps"):
        server_optimizers.create("fednova").apply(
            {"w": np.zeros(1)}, {"w": np.zeros(1), "fednova/w": np.zeros(1)}, {}
        )
```

`tests/test_learning_trainers.py`:

```python
@real
def test_fednova_sends_its_normalised_update(swell) -> None:
    model = build()
    received = state_arrays(model)

    result = trainers.create("fednova", {"lr": 0.05}).train(
        model, swell, received, ctx()
    )

    after = state_arrays(model)
    for key, x in received.items():
        np.testing.assert_allclose(
            result.aux[f"fednova/{key}"], (after[key] - x) / result.batches, rtol=1e-5
        )
```

Add `"fednova"` to the registry list test, and remove the `xfail` marks added in Task 3.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_learning_aggregators.py tests/test_learning_trainers.py tests/test_experiment.py -q -k "fednova or registry or paired"`
Expected: FAIL with unknown plugin `fednova`.

- [ ] **Step 3: Implement**

`trainers.py` (after `Scaffold`):

```python
@trainers.register(
    "fednova",
    title="FedNova",
    description="Entrenamiento estándar que envía además su actualización normalizada por pasos.",
    params=StandardParams,
    explain=(
        "Cada edge envía (y − x)/τ_i; el optimizador de servidor fednova aplica "
        "x + τ̄·d̄, así los edges que dan más pasos no arrastran el modelo "
        "(Wang et al., 2020). Exige server_optimizer fednova."
    ),
)
class FedNova(Standard):
    server_optimizer = "fednova"

    def train(
        self,
        model: nn.Module,
        data: Samples,
        received: Mapping[str, np.ndarray] | None = None,
        ctx: Any = None,
    ) -> TrainResult:
        start = state_arrays(model)
        result = super().train(model, data, received, ctx)
        after = state_arrays(model)
        names = trainable(model, self.params.frozen)
        aux = {
            f"fednova/{n}": ((after[n] - start[n]) / result.batches).astype(
                start[n].dtype
            )
            for n in names
        }
        return dataclasses.replace(result, aux=aux)
```

`aggregators.py`:

```python
@server_optimizers.register(
    "fednova",
    title="FedNova",
    description="x + τ̄·d̄: la media de las actualizaciones normalizadas por la media de pasos.",
    explain="Va con el entrenador fednova; con pasos iguales coincide con FedAvg (Wang et al., 2020).",
)
class FedNovaOptimizer:
    PREFIX = "fednova/"

    def check_trainer(self, name: str, trainer: Any) -> None:
        if name != "fednova":
            raise ValueError(f"fednova needs the fednova trainer, not {name!r}")

    def apply(
        self,
        global_state: Mapping[str, np.ndarray],
        aggregated: Mapping[str, np.ndarray],
        stats: Mapping[str, float] | None = None,
    ) -> State:
        tau = (stats or {}).get("train_steps")
        if not tau:
            raise ValueError("fednova needs train_steps in the round statistics")
        new = {k: v for k, v in global_state.items() if not k.startswith(self.PREFIX)}
        for key, value in aggregated.items():
            if key.startswith(self.PREFIX):
                continue
            step = aggregated.get(self.PREFIX + key)
            if step is None:  # no normalised update (frozen or non-trainable): replace
                new[key] = value
                continue
            current = np.asarray(global_state.get(key, value), dtype=np.float64)
            new[key] = (current + tau * np.asarray(step, np.float64)).astype(
                np.asarray(value).dtype
            )
        return new
```

- [ ] **Step 4: Run the suites**

Run: `.venv/Scripts/python -m pytest tests/test_learning_aggregators.py tests/test_learning_trainers.py tests/test_experiment.py tests/test_studio.py -q`
Expected: all pass, including both pairing directions.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/learning/trainers.py src/onion_fl/learning/aggregators.py tests/
git commit -m "feat(learning): Add FedNova, as a trainer and server optimizer pair"
```

---

### Task 6: FedDyn (trainer and server optimizer)

**Files:**
- Modify: `src/onion_fl/learning/trainers.py`, `src/onion_fl/learning/aggregators.py`
- Test: `tests/test_learning_trainers.py`, `tests/test_learning_aggregators.py`, `tests/test_experiment.py`

**Interfaces:**
- Consumes: `stats["train_edges"]` and `stats["edges_total"]` (Task 2), pairing (Task 3).
- Produces:
  - the trainer `feddyn`, with `FedDynParams(StandardParams).alpha = 0.01` and `server_optimizer = "feddyn"`;
  - the server optimizer `feddyn`, with `alpha = 0.01`. Its `check_trainer` requires the `feddyn` trainer with the same `alpha`.

- [ ] **Step 1: Write the failing tests**

`tests/test_learning_aggregators.py`:

```python
def test_feddyn_moves_the_global_model_against_its_drift_term() -> None:
    optimizer = server_optimizers.create("feddyn", {"alpha": 0.5})
    stats = {"train_edges": 2.0, "edges_total": 4.0}

    first = optimizer.apply({"w": np.zeros(1)}, {"w": np.full(1, 2.0)}, stats)
    # h = 0 − 0.5·(2/4)·(2 − 0) = −0.5 ; w = 2 − h/0.5 = 3
    np.testing.assert_allclose(first["w"], [3.0])

    second = optimizer.apply(first, {"w": np.full(1, 3.0)}, stats)
    # h = −0.5 − 0.5·0.5·(3 − 3) = −0.5 ; w = 3 + 1 = 4
    np.testing.assert_allclose(second["w"], [4.0])
```

`tests/test_learning_trainers.py`:

```python
@real
def test_feddyn_without_history_is_proximal(swell) -> None:
    received = state_arrays(build())
    prox, dyn = build(), build()
    params = {"local_epochs": 1, "lr": 0.05}

    trainers.create("fedprox", params | {"mu": 0.3}).train(prox, swell, received, ctx())
    trainers.create("feddyn", params | {"alpha": 0.3}).train(dyn, swell, received, ctx())

    for key, value in state_arrays(prox).items():
        np.testing.assert_allclose(value, state_arrays(dyn)[key], rtol=1e-6)


@real
def test_feddyn_remembers_its_gradient_term(swell) -> None:
    trainer = trainers.create("feddyn", {"lr": 0.05, "alpha": 0.3})
    model = build()
    trainer.train(model, swell, state_arrays(model), ctx())
    second = state_arrays(model)
    trainer.train(model, swell, second, ctx(1))

    fresh, start = trainers.create("feddyn", {"lr": 0.05, "alpha": 0.3}), build()
    load_arrays(start, second)
    fresh.train(start, swell, second, ctx(1))

    assert any(
        not np.allclose(v, state_arrays(start)[k])
        for k, v in state_arrays(model).items()
    )
```

`tests/test_experiment.py`:

```python
def test_feddyn_needs_the_same_alpha_on_both_sides(workspace: Path) -> None:
    learning = experiment(workspace)["learning"] | {
        "trainer": {"name": "feddyn", "alpha": 0.1},
        "server_optimizer": {"name": "feddyn", "alpha": 0.2},
    }

    with pytest.raises(ConfigError, match="alpha"):
        plan(parse_experiment(experiment(workspace, learning=learning)))
```

Add `"feddyn"` to the registry list test.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_learning_aggregators.py tests/test_learning_trainers.py tests/test_experiment.py -q -k "feddyn or registry"`
Expected: FAIL with unknown plugin `feddyn`.

- [ ] **Step 3: Implement**

`trainers.py`:

```python
class FedDynParams(StandardParams):
    alpha: float = Field(0.01, gt=0, description="α del regularizador dinámico")


@trainers.register(
    "feddyn",
    title="FedDyn",
    description="Regularizador dinámico: término proximal más el gradiente que el edge recuerda.",
    params=FedDynParams,
    explain=(
        "Minimiza L_i(θ) − ⟨∇L_i(θ_i^{t−1}), θ⟩ + α/2·‖θ − θ^{t−1}‖² y actualiza "
        "su gradiente recordado; exige server_optimizer feddyn con el mismo α "
        "(Acar et al., 2021)."
    ),
)
class FedDyn(Standard):
    Params = FedDynParams
    server_optimizer = "feddyn"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._grad: dict[str, torch.Tensor] = {}

    def _check(self, received: Mapping[str, np.ndarray] | None) -> None:
        if received is None:
            raise TrainError("feddyn needs the received global state")

    def _penalty(
        self, model: nn.Module, received: Any, names: Sequence[str]
    ) -> torch.Tensor:
        params = dict(model.named_parameters())
        total = proximal_term(model, received, self.params.alpha, names)
        for name in names:
            if name in self._grad:
                total = total - (params[name] * self._grad[name]).sum()
        return total

    def train(
        self,
        model: nn.Module,
        data: Samples,
        received: Mapping[str, np.ndarray] | None = None,
        ctx: Any = None,
    ) -> TrainResult:
        result = super().train(model, data, received, ctx)
        params = dict(model.named_parameters())
        for name in trainable(model, self.params.frozen):
            if name in received:
                anchor = torch.as_tensor(np.asarray(received[name]), dtype=params[name].dtype)
                drift = params[name].detach() - anchor
                self._grad[name] = self._grad.get(name, torch.zeros_like(drift)) - self.params.alpha * drift
        return result
```

`aggregators.py`:

```python
class FedDynOptimizerParams(BaseModel):
    alpha: float = Field(0.01, gt=0, description="El mismo α que el entrenador feddyn")


@server_optimizers.register(
    "feddyn",
    title="FedDyn",
    description="h ← h − α·(|P|/m)·(θ̄ − θ); θ ← θ̄ − h/α.",
    params=FedDynOptimizerParams,
    explain="Va con el entrenador feddyn y su mismo α; |P|/m sale de train_edges/edges_total (Acar et al., 2021).",
)
class FedDynOptimizer:
    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        self._h: dict[str, np.ndarray] = {}

    def check_trainer(self, name: str, trainer: Any) -> None:
        if name != "feddyn":
            raise ValueError(f"feddyn needs the feddyn trainer, not {name!r}")
        if trainer.params.alpha != self.alpha:
            raise ValueError(
                f"alpha {self.alpha} differs from the trainer's {trainer.params.alpha}"
            )

    def apply(
        self,
        global_state: Mapping[str, np.ndarray],
        aggregated: Mapping[str, np.ndarray],
        stats: Mapping[str, float] | None = None,
    ) -> State:
        stats = stats or {}
        total = stats.get("edges_total") or 0.0
        share = stats.get("train_edges", total) / total if total else 1.0
        new = dict(global_state)
        for key, value in aggregated.items():
            if is_aux(key):
                new[key] = value
                continue
            mean = np.asarray(value, dtype=np.float64)
            current = np.asarray(global_state.get(key, value), dtype=np.float64)
            h = self._h.get(key, np.zeros_like(mean)) - self.alpha * share * (mean - current)
            self._h[key] = h
            new[key] = (mean - h / self.alpha).astype(np.asarray(value).dtype)
        return new
```

- [ ] **Step 4: Run the suites**

Run: `.venv/Scripts/python -m pytest tests/test_learning_aggregators.py tests/test_learning_trainers.py tests/test_experiment.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/learning/trainers.py src/onion_fl/learning/aggregators.py tests/
git commit -m "feat(learning): Add FedDyn, as a trainer and server optimizer pair"
```

---

### Task 7: Model features and the MOON trainer

**Files:**
- Modify: `src/onion_fl/learning/model.py` (`ModularMLP.features`), `src/onion_fl/learning/trainers.py` (`Standard._batch_term` hook, `Moon`)
- Test: `tests/test_learning_model.py`, `tests/test_learning_trainers.py`

**Interfaces:**
- Produces:
  - `ModularMLP.features(x, dataset=None) -> Tensor`, the trunk output; `forward` is now `head(features)`;
  - the hook `Standard._batch_term(model, xb) -> Tensor | float`, which defaults to `0.0`;
  - the trainer `moon`, with `MoonParams(StandardParams)`: `mu: float = 1.0` and `temperature: float = 0.5`.

- [ ] **Step 1: Write the failing tests**

`tests/test_learning_model.py`:

```python
def test_the_head_reads_the_features() -> None:
    model = ModularMLP(ModularMLPConfig(adapter_width=4, trunk_hidden=[3], dropout=0.0), [SHAPE], seed=0)
    x = torch.zeros((2, SHAPE.n_features))
    model.eval()

    features = model.features(x)

    assert features.shape == (2, 3)
    torch.testing.assert_close(model(x), model.head[SHAPE.task](features))
```

(use the file's existing single-dataset `DataShape` constant, or define `SHAPE = DataShape(dataset="swell", task="stress_binary", n_features=16, n_classes=2)`; a zeros input is a shape fixture, not training data)

`tests/test_learning_trainers.py`:

```python
@real
def test_moon_in_its_first_round_is_standard(swell) -> None:
    params = {"local_epochs": 1, "lr": 0.05}
    plain, moon = build(), build()
    received = state_arrays(plain)

    trainers.create("standard", params).train(plain, swell, received, ctx())
    trainers.create("moon", params).train(moon, swell, received, ctx())

    for key, value in state_arrays(plain).items():
        np.testing.assert_array_equal(value, state_arrays(moon)[key])


@real
def test_moon_pulls_towards_the_global_representation_from_round_two(swell) -> None:
    params = {"local_epochs": 1, "lr": 0.05}
    plain, moon = build(), build()
    standard, contrastive = trainers.create("standard", params), trainers.create("moon", params | {"mu": 5.0})
    for model, trainer in ((plain, standard), (moon, contrastive)):
        trainer.train(model, swell, state_arrays(model), ctx())
    second_plain, second_moon = state_arrays(plain), state_arrays(moon)

    standard.train(plain, swell, second_plain, ctx(1))
    contrastive.train(moon, swell, second_moon, ctx(1))

    assert any(
        not np.allclose(v, state_arrays(moon)[k]) for k, v in state_arrays(plain).items()
    )
```

Add `"moon"` to the registry list test.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_learning_model.py tests/test_learning_trainers.py -q -k "features or moon or registry"`
Expected: FAIL. `features` does not exist and `moon` is unknown.

- [ ] **Step 3: Implement**

`model.py`, splitting `forward`:

```python
    def features(self, x: torch.Tensor, dataset: str | None = None) -> torch.Tensor:
        """The trunk output: the representation the head reads."""
        dataset = self._dataset(dataset)
        shared = self.config.adapters == "shared"
        hidden = self.adapter(x) if shared else self.adapter[dataset](x)
        return (
            self.trunk(hidden)
            if self.config.trunk == "shared"
            else self.trunk[dataset](hidden)
        )

    def forward(self, x: torch.Tensor, dataset: str | None = None) -> torch.Tensor:
        dataset = self._dataset(dataset)
        return self.head[self._head_key(self.shapes[dataset])](self.features(x, dataset))

    def _dataset(self, dataset: str | None) -> str:
        ...  # the existing checks from forward, returning the resolved name
```

`trainers.py`:
- in `Standard`, add `def _batch_term(self, model, xb) -> torch.Tensor | float: return 0.0`;
- in `Standard.train`, change the backward line to `(loss + self._penalty(model, received, names) + self._batch_term(model, xs[idx])).backward()`;
- then add:

```python
class MoonParams(StandardParams):
    mu: float = Field(1.0, ge=0, description="Peso de la pérdida contrastiva")
    temperature: float = Field(0.5, gt=0, description="Temperatura τ del contraste")


@trainers.register(
    "moon",
    title="MOON",
    description="Contraste de modelo: la representación local se acerca a la del global y se aleja de la anterior.",
    params=MoonParams,
    explain=(
        "Suma μ·ℓ_con con similitud coseno entre las features del modelo local, "
        "las del global recibido y las del modelo local de la ronda anterior "
        "(Li et al., 2021). En la primera ronda no hay modelo anterior."
    ),
)
class Moon(Standard):
    Params = MoonParams

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._global: nn.Module | None = None
        self._previous: nn.Module | None = None

    def _batch_term(self, model: nn.Module, xb: torch.Tensor) -> torch.Tensor | float:
        if self._previous is None or self._global is None:
            return 0.0
        z = model.features(xb)
        with torch.no_grad():
            z_global, z_previous = self._global.features(xb), self._previous.features(xb)
        tau = self.params.temperature
        logits = torch.stack(
            [
                F.cosine_similarity(z, z_global, dim=-1) / tau,
                F.cosine_similarity(z, z_previous, dim=-1) / tau,
            ],
            dim=1,
        )
        target = torch.zeros(len(xb), dtype=torch.long)
        return self.params.mu * F.cross_entropy(logits, target)

    def train(
        self,
        model: nn.Module,
        data: Samples,
        received: Mapping[str, np.ndarray] | None = None,
        ctx: Any = None,
    ) -> TrainResult:
        self._global = copy.deepcopy(model).eval()  # the model as received
        result = super().train(model, data, received, ctx)
        self._previous = copy.deepcopy(model).eval()
        return result
```

(the `.eval()` copies run without dropout, so they draw nothing from torch's RNG.)

- [ ] **Step 4: Run the suites**

Run: `.venv/Scripts/python -m pytest tests/test_learning_model.py tests/test_learning_trainers.py tests/test_roles_evaluation.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/learning/model.py src/onion_fl/learning/trainers.py tests/
git commit -m "feat(learning): Add model features and the MOON trainer"
```

---

### Task 8: Drift comparison, results and docs

**Files:**
- Create: `experiments/techniques_drift.yaml`
- Create: `results/techniques_drift/` (run copies, `report.html`, `edge_scores.csv`, `INDEX.md`)
- Modify: `README.md` (§1 Learning row, test count, §7), `docs/architecture.md` (learning section, plugin table, round protocol stats)

- [ ] **Step 1: Pick the SGD learning rate.** SCAFFOLD and FedNova assume SGD, so every scenario uses SGD.
  - Probe `standard` with SGD at lr 0.03, 0.1 and 0.3: one seed, α = 0, 20 rounds, on a scratchpad copy of the experiment.
  - Keep the lr whose final global macro-F1 is highest, averaged over SWELL and WESAD.
  - Record it as a ledger ruling.

- [ ] **Step 2: Write the experiment** (lr from Step 1)

```yaml
# Non-IID drift techniques on SWELL and WESAD (issue #148): one scenario per
# technique, three seeds, segregated placement (each fog sees one dataset).
# Every technique uses SGD, which SCAFFOLD and FedNova assume.
#
#   onion_fl plan experiments/techniques_drift.yaml
#   onion_fl run experiments/techniques_drift.yaml --workers 3
name: techniques_drift
description: Deriva no IID (FedAvg, FedProx, SCAFFOLD, FedNova, FedDyn, MOON) sobre SWELL y WESAD.
topology: four_fogs_swell_wesad
data:
  datasets:
    swell: {}
    wesad: {}
  roles: {test: 0.2, val: 0.1, local_val: 0.2, local_val_split: class_tail, scaler: global, seed: 0}
  placement: {name: mixing, alpha: 0.0}
learning:
  model: {name: modular_mlp, adapter_width: 64, trunk_hidden: [64, 32], dropout: 0.2}
  sharing: fedavg
  trainer: {name: standard, optimizer: sgd, local_epochs: 10, batch_size: 32, lr: <LR>}
  init: random
rounds: 20
evaluation:
  metrics: [loss, accuracy, macro_f1]
  edge: {every: 5, models: [received, local]}
  aggregators: {every: 5}
  global: {every: 1}
seeds: [0, 1, 2]
sweep:
  learning.trainer.name,learning.server_optimizer:
    - [standard, replace]   # FedAvg
    - [fedprox, replace]
    - [scaffold, replace]
    - [fednova, fednova]
    - [feddyn, feddyn]
    - [moon, replace]
```

- [ ] **Step 3: Plan, commit, run on a clean tree.**
  - `onion_fl plan` must give 18 entries with no warnings.
  - Commit the experiment.
  - With `git status --short` empty, move `runs/` aside and run `onion_fl run experiments/techniques_drift.yaml --workers 3` in the background. Do not touch the tree until it ends.

- [ ] **Step 4: Results.**
  - Copy each run's `run.json` (without `host`/`pid`), `summary.json` and `model.npz` into `results/techniques_drift/<run_id>/`.
  - Write `edge_scores.csv`: the round-20 edge macro-F1 per run and model, from `source=edge` events weighted by samples. The P1 script `edge_scores_p1.py` pattern applies; adapt its experiment name.
  - Add the final global macro-F1 per dataset from `summary.json`, and each run's `quorum_failed`.
  - Generate `report.html` with `--by scenario --by model`.
  - Write `INDEX.md` like `results/techniques_personalisation/INDEX.md`. Report what the numbers show, including if no technique beats FedAvg.

- [ ] **Step 5: Docs, full check, commit.**
  - README §1 Learning row: add `scaffold`, `fednova`, `feddyn`, `moon` and the round statistics.
  - README test count.
  - README §7: a "New framework: drift techniques" subsection citing `results/techniques_drift/`.
  - `docs/architecture.md`:
    - the aggregation paragraph: auxiliary arrays and stats in the optimizer, with `fednova`/`feddyn` among the server optimizers;
    - the trainers bullet;
    - the plugin table.
  - Run the full check, then commit.

- [ ] **Step 6: Final review, PR (left open for the user's review), no merge.** After the whole-branch review and its fix pass:
  - push `task/#148`;
  - run `gh.py pr 148 --note "<counts>"`;
  - report the PR URL. The user reviews it in depth before it is merged.
