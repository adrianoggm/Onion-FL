# FL techniques P1 (personalisation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ditto, APFL, FedRep, FedBABU and LG-FedAvg can be chosen by name, swept and compared. Each edge reports `personal` and `finetuned` scores on a class-stratified local split, so personalisation can be measured (issue #147).

**Architecture:**
- **Plugins.** Trainers are plugins in `learning/trainers.py`. Every edge keeps one trainer object for the whole run, so a trainer can hold a personal model across rounds and expose it through `personal()`.
- **Edge scoring.** The edge scores that personal model, plus a copy of the received model after a short fine-tune, as `personal` and `finetuned`. It uses the existing `eval.<model>.*` metrics, which aggregators already combine at every level.
- **New config options.** A class-stratified `local_val`, joint sweep keys (`a,b`) and a trainer/sharing compatibility check. Each new config field is left out of the dump when unset, so existing `config_id`s do not change.

**Tech Stack:** Python 3.11, PyTorch (`torch.func.functional_call`), pydantic 2.13 (`Field(exclude_if=...)`), NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-fl-techniques-design.md` (§3.4 and §4 P1).

**Scope:**
- §3.1–3.3 of the spec (auxiliary keys, round statistics, aggregator reference) have no consumer in P1. They move to issues #148 and #149, where SCAFFOLD, FedNova and the robust aggregators use and test them.
- P1 adds two prerequisites that the spec did not list:
  - a class-stratified `local_val`, because the time-ordered tail is a single class and makes personal scores meaningless;
  - joint sweep keys, so a trainer and its sharing policy form one scenario.

## Global Constraints

- **Environment.** Python ≥ 3.11; run everything with `.venv/Scripts/python` (Windows) and `.venv/Scripts/ruff`.
- **Lint and warnings.** ruff with 88 columns. Modules start with `from __future__ import annotations`. pytest runs with `filterwarnings = error`.
- **Language.** Plugin `title`, `description`, `explain` and parameter descriptions are in Spanish. Code, comments and docs are in English.
- **No synthetic data for ML (docs/RULES.md).** Tests that run an optimiser use `data/samples/swell_real_sample.pkl` and skip without it. Protocol tests use the `stub` trainer and a stub scorer.
- **Stable `config_id`s.** A new config field must not change the `config_id` of existing experiments: when it has its default it stays out of the dump (`exclude_if`).
- **Evaluation never alters training.** Scoring (including fine-tuning) must not change the training trajectory of a seed.
- **Commits and workflow.**
  - Commit subjects are `type(scope): Imperative summary in English`, with no `Co-Authored-By` lines.
  - Work happens on branch `task/#147`. Before each commit, check `git diff --cached --name-only`.
- **Full check:** `.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check . && .venv/Scripts/python -m pytest -q`.

## Review Focus

1. **Fine-tune scoring must leave training unchanged.** A run with `finetuned` scoring must end with the same global model as the same seed without it. Pinned in Task 3.
2. **Ditto under a policy that keeps heads local.** With `fedper`, `received` has no head keys, and the personal model must still train. Pinned in Task 4.
3. **A subject with one class under `class_tail`.** It holds out only that class's last rows and keeps at least one training row. Pinned in Task 1.
4. **A malformed joint sweep value** (a string, or a list of the wrong length) gives a `ConfigError` naming the key. Pinned in Task 2.
5. **FedRep with a sharing policy that sends heads up** fails at `plan`, before anything runs, with a message naming `fedrep`. Pinned in Task 6.

---

### Task 1: Class-stratified local validation

**Files:**
- Modify: `src/onion_fl/data/roles.py` (`RolesConfig`, `split_subjects` lines 214-263)
- Test: `tests/test_data_roles.py`

**Interfaces:**
- Produces: `RolesConfig.local_val_split: Literal["tail", "class_tail"]` (default `"tail"`, left out of `model_dump()` when default), `_held_out(y, share, split) -> np.ndarray[bool]`.

- [ ] **Step 1: Write the failing tests** (append after `test_no_local_val_by_default`)

```python
def test_class_tail_holds_out_the_last_rows_of_each_class() -> None:
    subjects = [subject("1", [[i] for i in range(1, 11)], [0] * 5 + [1] * 5)]

    (client,) = split_subjects(
        subjects,
        RolesConfig(
            test=0.0, local_val=0.2, local_val_split="class_tail", scaler="none"
        ),
    ).clients

    assert client.local_val.X[:, 0].tolist() == [5, 10]
    assert client.local_val.y.tolist() == [0, 1]
    assert client.train.X[:, 0].tolist() == [1, 2, 3, 4, 6, 7, 8, 9]


def test_class_tail_always_leaves_a_training_row() -> None:
    subjects = [subject("1", [[1], [2]], [0, 1])]

    (client,) = split_subjects(
        subjects,
        RolesConfig(
            test=0.0, local_val=0.9, local_val_split="class_tail", scaler="none"
        ),
    ).clients

    assert client.train.X[:, 0].tolist() == [1]
    assert client.local_val.X[:, 0].tolist() == [2]


def test_class_tail_with_one_class_is_that_class_tail() -> None:
    subjects = [subject("1", [[1], [2], [3], [4]], [1, 1, 1, 1])]

    (client,) = split_subjects(
        subjects,
        RolesConfig(
            test=0.0, local_val=0.25, local_val_split="class_tail", scaler="none"
        ),
    ).clients

    assert client.local_val.X[:, 0].tolist() == [4]


def test_the_default_split_stays_out_of_the_config() -> None:
    assert "local_val_split" not in RolesConfig().model_dump()
    dumped = RolesConfig(local_val_split="class_tail").model_dump()
    assert dumped["local_val_split"] == "class_tail"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_data_roles.py -q -k "class_tail or default_split"`
Expected: FAIL, because `RolesConfig` rejects the extra field `local_val_split`.

- [ ] **Step 3: Implement**

In `RolesConfig`, after `local_val`:

```python
    local_val_split: Literal["tail", "class_tail"] = Field(
        "tail",
        description="tail: últimas filas del sujeto; class_tail: últimas filas de cada clase",
        exclude_if=lambda v: v == "tail",  # unset, it keeps existing config_ids
    )
```

Above `split_subjects`:

```python
def _held_out(y: np.ndarray, share: float, split: str) -> np.ndarray:
    """Rows of one subject kept for its local validation, in time order.

    ``tail`` takes the last rows, which in a recording usually hold a single
    condition; ``class_tail`` takes the last rows of each class instead.
    """
    n = len(y)
    held = np.zeros(n, dtype=bool)
    if split == "tail":
        held[n - min(round(share * n), n - 1) :] = True
        return held
    for label in np.unique(y):
        rows = np.flatnonzero(y == label)
        k = round(share * len(rows))
        if k:
            held[rows[-k:]] = True
    if held.all():
        held[0] = False  # a subject always keeps a training row
    return held
```

In `split_subjects`, replace the `cut` block and its three uses:

```python
        # Rows each training subject keeps for its local validation.
        held = {
            name: _held_out(pool[name].y, config.local_val, config.local_val_split)
            for name in roles[dataset]["train"]
        }
        bag = np.concatenate([pool[s].X[~held[s]] for s in held])
```

```python
        order = [roles[dataset]["train"][i] for i in rng.permutation(len(held))]
```

```python
            train_X = np.concatenate([d.X[~held[d.subject]][:, keep] for d in rows])
            train_y = np.concatenate([d.y[~held[d.subject]] for d in rows])
            tail_X = np.concatenate([d.X[held[d.subject]][:, keep] for d in rows])
            tail_y = np.concatenate([d.y[held[d.subject]] for d in rows])
```

Also update the module docstring's sentence "never on val, test or the ``local_val`` tails" to "never on val, test or the ``local_val`` rows".

- [ ] **Step 4: Run the roles tests**

Run: `.venv/Scripts/python -m pytest tests/test_data_roles.py -q`
Expected: all pass, including the existing `test_local_val_is_the_tail_of_each_subject`.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/data/roles.py tests/test_data_roles.py
git commit -m "feat(data): Hold out the last rows of each class for local validation

The time-ordered tail of a subject is mostly one condition, so edge
scores on it say nothing about personal models. class_tail keeps the
time order within each class; the default stays tail and out of the dump."
```

---

### Task 2: Sweep several paths together

**Files:**
- Modify: `src/onion_fl/experiment/sweep.py` (`scenarios`)
- Test: `tests/test_experiment.py` (sweeps section)

**Interfaces:**
- Produces: sweep keys like `"learning.sharing,learning.trainer.name"`, whose values are lists with one entry per path. The scenario name is `learning.sharing,learning.trainer.name=fedper,fedrep`.

- [ ] **Step 1: Write the failing tests** (after `test_sweeping_a_plugin_given_by_name`)

```python
def test_paths_joined_by_commas_are_swept_together(workspace: Path) -> None:
    config = parse_experiment(
        experiment(
            workspace,
            sweep={"learning.trainer.shift,rounds": [[1.0, 1], [2.0, 3]]},
        )
    )

    out = scenarios(config)

    assert [s.name for s in out] == [
        "learning.trainer.shift,rounds=1.0,1",
        "learning.trainer.shift,rounds=2.0,3",
    ]
    assert [(s.config.learning.trainer["shift"], s.config.rounds) for s in out] == [
        (1.0, 1),
        (2.0, 3),
    ]


@pytest.mark.parametrize("value", [[1.0], "fedavg"])
def test_a_joint_sweep_value_needs_one_entry_per_path(workspace: Path, value) -> None:
    config = parse_experiment(
        experiment(workspace, sweep={"learning.trainer.shift,rounds": [value]})
    )

    with pytest.raises(ConfigError, match="learning.trainer.shift,rounds"):
        scenarios(config)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_experiment.py -q -k "together or joint"`
Expected: FAIL. `_set` treats `"learning.trainer.shift,rounds"` as one path, so `parse_experiment` rejects the unknown field `"trainer.shift,rounds"`, or else the name differs.

- [ ] **Step 3: Implement** — replace the loop body of `scenarios` in `sweep.py`

```python
def _apply(resolved: dict[str, Any], key: str, value: Any) -> str:
    """Set one sweep entry and return its label; ``a,b`` sets several paths at once."""
    paths = key.split(",")
    if len(paths) == 1:
        _set(resolved, key, value)
        return f"{key}={_label(value)}"
    if not isinstance(value, list) or len(value) != len(paths):
        raise ConfigError(
            f"sweep {key}: each value needs {len(paths)} entries, one per path; "
            f"got {value!r}"
        )
    for path, item in zip(paths, value, strict=True):
        _set(resolved, path, item)
    return f"{key}={','.join(_label(item) for item in value)}"


def scenarios(config: ExperimentConfig) -> list[Scenario]:
    base = config.dump()
    base.pop("sweep")
    seeds = base.pop("seeds")
    keys = sorted(config.sweep)
    out = []
    for values in product(*(config.sweep[k] for k in keys)):
        resolved = copy.deepcopy(base)
        labels = [
            _apply(resolved, key, value)
            for key, value in zip(keys, values, strict=True)
        ]
        name = ",".join(labels) or "base"
        try:
            scenario_config = parse_experiment(resolved | {"seeds": seeds})
        except ConfigError as exc:
            raise ConfigError(f"sweep scenario {name}:\n{exc}") from None
        cid = config_id(identity(scenario_config))
        out += [Scenario(name, seed, scenario_config, cid) for seed in seeds]
    return out
```

- [ ] **Step 4: Run the experiment tests**

Run: `.venv/Scripts/python -m pytest tests/test_experiment.py tests/test_studio.py -q`
Expected: all pass. The existing names (`data.placement.alpha=0.0,rounds=1`) are unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/experiment/sweep.py tests/test_experiment.py
git commit -m "feat(experiments): Sweep several paths together

A key a,b takes values [x, y] and sets both paths in one scenario, so a
trainer and the sharing it needs are swept as a pair."
```

---

### Task 3: Personal and fine-tuned scores at the edges

**Files:**
- Modify: `src/onion_fl/experiment/config.py` (`EdgeEval`)
- Modify: `src/onion_fl/roles/federation.py` (`build_federation`, edge loop)
- Modify: `src/onion_fl/roles/nodes.py` (`Edge.__init__`, `Edge._score`, `Edge._train`)
- Test: `tests/test_roles_evaluation.py`, `tests/test_experiment.py`

**Interfaces:**
- Consumes: any trainer may define `personal() -> nn.Module | None`.
- Produces:
  - `EdgeEval.models` accepts `"personal"` and `"finetuned"`, and `EdgeEval.finetune: PluginRef | None` is a trainer.
  - `Edge(..., finetuner=None)`.
  - Events `eval.<metric>` with tag `model="personal"` or `model="finetuned"`, and update metrics `eval.personal.*` / `eval.finetuned.*`.

- [ ] **Step 1: Write the failing protocol tests** (in `tests/test_roles_evaluation.py`, after `test_edge_scores_follow_every`; add `import copy`, `import numpy as np` and `import torch` to the imports)

```python
class PersonalStub:
    """The stub trainer plus a personal model: the trained one shifted by 10."""

    def __init__(self) -> None:
        self.stub = trainers.create("stub", {"shift": 1.0})
        self.model = None

    def train(self, model, data=None, received=None, ctx=None):
        result = self.stub.train(model, data, received, ctx)
        self.model = copy.deepcopy(model)
        with torch.no_grad():
            for tensor in self.model.parameters():
                tensor.add_(10.0)
        return result

    def personal(self):
        return self.model


PERSONAL = {"eval": {"every": 1, "models": ["personal"]}}


def test_an_edge_scores_its_personal_model_when_the_trainer_has_one() -> None:
    spec = EdgeSpec("e1", model(A), trainer=PersonalStub(), val_data=samples(4))

    federation = run(tree(edge=PERSONAL), {"fog_0": [spec]})

    personal = [e["value"] for e in scores(federation, "e1", model="personal")]
    assert personal == pytest.approx([M0 + 11])


def test_trainers_without_a_personal_model_score_nothing_personal() -> None:
    federation = run(tree(edge=PERSONAL), {"fog_0": [trainer_edge("e1", val=4)]})

    assert scores(federation, "e1") == []


def test_the_fog_combines_personal_scores_like_the_others() -> None:
    edges = {
        "fog_0": [
            EdgeSpec(f"e{i}", model(A), trainer=PersonalStub(), val_data=samples(n))
            for i, n in ((1, 1), (2, 3))
        ]
    }

    federation = run(tree(edge=PERSONAL), edges)

    (fog,) = scores(federation, "fog_0", model="personal", source="children")
    assert fog["value"] == pytest.approx(M0 + 11)


FINETUNE = {
    "eval": {
        "every": 1,
        "models": ["finetuned"],
        "finetune": {"name": "stub", "shift": 5.0},
    }
}


def test_finetuned_scores_the_received_model_after_the_finetune_trainer() -> None:
    federation = run(
        tree(edge=FINETUNE),
        {"fog_0": [trainer_edge("e1", shift=1, val=4)]},
        rounds=2,
    )

    finetuned = [e["value"] for e in scores(federation, "e1", model="finetuned")]
    assert finetuned == pytest.approx([M0 + 5, M0 + 1 + 5])


def test_finetune_scoring_does_not_change_training() -> None:
    def final(edge: dict) -> dict:
        spec = EdgeSpec(
            "e1",
            model(A),
            trainer=trainers.create("stub", {"noise": 0.1}),
            val_data=samples(4),
        )
        federation = run(tree(edge=edge), {"fog_0": [spec]}, rounds=3)
        return federation.coordinator.state

    plain = final({"eval": {"every": 1, "models": ["local"]}})
    noisy_finetune = {"name": "stub", "noise": 0.1}
    scored = final(
        {
            "eval": {
                "every": 1,
                "models": ["local", "finetuned"],
                "finetune": noisy_finetune,
            }
        }
    )

    for key, value in plain.items():
        np.testing.assert_array_equal(value, scored[key])
```

In `tests/test_experiment.py`, in the validation section:

```python
def test_finetuned_scores_need_a_finetune_trainer(workspace: Path) -> None:
    raw = experiment(workspace, evaluation={"edge": {"models": ["finetuned"]}})

    with pytest.raises(ConfigError, match="finetune"):
        parse_experiment(raw)


def test_an_unset_finetune_stays_out_of_the_config(workspace: Path) -> None:
    config = parse_experiment(experiment(workspace))

    assert "finetune" not in config.dump()["evaluation"]["edge"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_roles_evaluation.py tests/test_experiment.py -q -k "personal or finetune"`
Expected: the personal tests FAIL because no `personal` events appear. The fine-tune tests FAIL because the `finetune` key is not accepted, or no `finetuned` events appear.

- [ ] **Step 3: Implement the config** (`config.py`)

```python
class EdgeEval(Strict):
    every: PositiveInt | None = None
    models: list[Literal["received", "local", "personal", "finetuned"]] = Field(
        default_factory=lambda: ["received", "local"]
    )
    finetune: PluginRef | None = Field(
        None,
        description="Entrenador del ajuste fino antes de puntuar 'finetuned'",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )

    _finetune = field_validator("finetune")(
        lambda v: v if v is None else _plugin(trainers, v)
    )

    @model_validator(mode="after")
    def _finetuned_needs_a_trainer(self) -> EdgeEval:
        if "finetuned" in self.models and self.finetune is None:
            raise ValueError("models: 'finetuned' needs evaluation.edge.finetune")
        return self
```

`trainers`, `model_validator` and `PluginRef` are already imported in `config.py`. If any is not, add it to the existing import lines.

- [ ] **Step 4: Implement the federation wiring** (`federation.py`)

At the top: `from onion_fl.learning.trainers import trainers as trainer_plugins` (aliased because the function already has a `trainers_at_root` variable and edges carry a `trainer`). Before the edge loop:

```python
    finetune = edge_eval.get("finetune")
```

In the `Edge(...)` call add:

```python
                finetuner=(
                    create(trainer_plugins, finetune)
                    if finetune is not None and spec.train
                    else None
                ),
```

- [ ] **Step 5: Implement the edge** (`nodes.py`)

Imports: add `import copy`, `from types import SimpleNamespace` and `import numpy as np` if absent.

`Edge.__init__`: add the keyword `finetuner: Any = None` and `self.finetuner = finetuner`.

Replace `_score`:

```python
    def _score(
        self, model: str, round: int, ctx: Context, module: Any = None
    ) -> dict[str, float]:
        scores, samples = self.evaluate(
            self.model if module is None else module, self.val_data
        )
        _emit_scores(
            ctx,
            scores,
            model=model,
            source="edge",
            round=round,
            samples=samples,
            **self.tags,
        )
        return {f"eval.{model}.{k}": float(v) for k, v in scores.items()} | {
            f"eval.{model}.samples": float(samples)
        }

    def _finetuned(self, start: Any, received: State, ctx: Context) -> Any:
        """``start`` (the model as received) trained by the finetune trainer.

        It draws from a child stream of the node's generator, so scoring
        never shifts the draws of the edge's own training.
        """
        rng = np.random.default_rng(ctx.rng.bit_generator.seed_seq.spawn(1)[0])
        self.finetuner.train(
            start, self.data, received=received, ctx=SimpleNamespace(rng=rng)
        )
        return start
```

In `_train`, right after `load_arrays(self.model, received)` and the `scoring` assignment, add:

```python
        finetuning = (
            scoring and "finetuned" in self.eval_models and self.finetuner is not None
        )
        start = copy.deepcopy(self.model) if finetuning else None
```

After the existing `local` scoring block, add:

```python
        if scoring and "personal" in self.eval_models:
            personal = getattr(self.trainer, "personal", lambda: None)()
            if personal is not None:
                metrics |= self._score("personal", msg.round, ctx, personal)
        if finetuning:
            metrics |= self._score(
                "finetuned", msg.round, ctx, self._finetuned(start, received, ctx)
            )
```

- [ ] **Step 6: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_roles_evaluation.py tests/test_experiment.py tests/test_roles_protocol.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/onion_fl/experiment/config.py src/onion_fl/roles/federation.py src/onion_fl/roles/nodes.py tests/test_roles_evaluation.py tests/test_experiment.py
git commit -m "feat(roles): Score personal and fine-tuned models at the edges

A trainer with personal() gets its personal model scored as 'personal';
evaluation.edge.finetune trains a copy of the received model and scores
it as 'finetuned' from a child random stream, so training is unchanged.
Aggregators combine both like the other edge scores."
```

---

### Task 4: Shared training scaffold and the Ditto trainer

**Files:**
- Modify: `src/onion_fl/learning/trainers.py`
- Test: `tests/test_learning_trainers.py`

**Interfaces:**
- Produces:
  - `_samples(data) -> tuple[torch.Tensor, torch.Tensor]` and the context manager `_training(models, names, rng)`, used again by Task 5;
  - the trainer `ditto` with `DittoParams(StandardParams)`: `lam: float = 0.1`, `personal_epochs: PositiveInt | None = None`;
  - `Ditto.personal() -> nn.Module | None`.

- [ ] **Step 1: Refactor `Standard.train` onto two helpers (behaviour unchanged)**

Add near `trainable` (imports: `import copy`, `from collections.abc import Iterator`, `from contextlib import contextmanager`):

```python
def _samples(data: Samples) -> tuple[torch.Tensor, torch.Tensor]:
    X, y = np.asarray(data.X, np.float32), np.asarray(data.y, np.int64)
    if len(X) == 0:
        raise TrainError("no samples to train on")
    if len(X) != len(y):
        raise TrainError(f"X has {len(X)} rows but y has {len(y)} labels")
    return torch.from_numpy(X), torch.from_numpy(y)


@contextmanager
def _training(
    models: Sequence[nn.Module], names: Sequence[str], rng: np.random.Generator
) -> Iterator[None]:
    """Train mode with gradients only on ``names``, dropout seeded from ``rng``.

    Dropout draws from torch's global RNG: it is seeded from the node's rng
    inside a fork, so runs are reproducible and the caller's state is kept.
    """
    wanted = set(names)
    saved = [{n: t.requires_grad for n, t in m.named_parameters()} for m in models]
    for module in models:
        module.train()
        for name, tensor in module.named_parameters():
            tensor.requires_grad_(name in wanted)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(rng.integers(2**63)))
            yield
    finally:
        for module, was in zip(models, saved, strict=True):
            for name, tensor in module.named_parameters():
                tensor.requires_grad_(was[name])
```

Replace `Standard.train` with:

```python
    def train(
        self,
        model: nn.Module,
        data: Samples,
        received: Mapping[str, np.ndarray] | None = None,
        ctx: Any = None,
    ) -> TrainResult:
        xs, ys = _samples(data)
        self._check(received)
        p = self.params
        names = trainable(model, p.frozen)
        if not names:
            raise TrainError(f"every parameter is frozen by {p.frozen}")
        params = dict(model.named_parameters())
        optimizer = OPTIMIZERS[p.optimizer](
            [params[n] for n in names], lr=p.lr, weight_decay=p.weight_decay
        )
        rng = ctx.rng if ctx is not None else np.random.default_rng(0)
        total, batches = 0.0, 0
        with _training([model], names, rng):
            for _ in range(p.local_epochs):
                for idx in batches_of(len(xs), p.batch_size, rng):
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(xs[idx]), ys[idx])
                    total += float(loss.item()) * len(idx)
                    (loss + self._penalty(model, received, names)).backward()
                    optimizer.step()
                    batches += 1
        samples = p.local_epochs * len(xs)
        return TrainResult(
            loss=total / samples, samples=samples, examples=len(xs), batches=batches
        )
```

- [ ] **Step 2: Run the existing trainer tests (refactor stays green)**

Run: `.venv/Scripts/python -m pytest tests/test_learning_trainers.py -q`
Expected: all pass, including `test_the_same_rng_gives_the_same_weights` and `test_frozen_groups_do_not_change`.

- [ ] **Step 3: Commit the refactor**

```bash
git add src/onion_fl/learning/trainers.py
git commit -m "refactor(learning): Share sample checks and the training context"
```

- [ ] **Step 4: Write the failing Ditto tests** (real-data section, after the FedProx test)

```python
@real
def test_ditto_keeps_a_personal_model_apart_from_the_global(swell) -> None:
    model = build()
    received = state_arrays(model)
    trainer = trainers.create("ditto", {"local_epochs": 2, "lr": 0.05, "lam": 0.1})

    result = trainer.train(model, swell, received, ctx())

    personal, trained = state_arrays(trainer.personal()), state_arrays(model)
    assert any(not np.array_equal(personal[k], trained[k]) for k in trained)
    assert any(not np.array_equal(personal[k], received[k]) for k in received)
    assert result.samples == 4 * len(swell.y)  # global and personal epochs
    assert result.examples == len(swell.y)


@real
def test_a_larger_lambda_keeps_the_personal_model_near_the_global(swell) -> None:
    received = state_arrays(build())

    def distance(lam: float) -> float:
        trainer = trainers.create(
            "ditto", {"local_epochs": 3, "lr": 0.05, "lam": lam}
        )
        trainer.train(build(), swell, received, ctx())
        return sum(
            float(((v - received[k]) ** 2).sum())
            for k, v in state_arrays(trainer.personal()).items()
        )

    assert distance(10.0) < distance(0.0)


@real
def test_ditto_trains_when_the_heads_stay_on_the_edge(swell) -> None:
    model = build()
    received = {
        k: v for k, v in state_arrays(model).items() if not k.startswith("head.")
    }
    trainer = trainers.create("ditto", {"lr": 0.05})

    trainer.train(model, swell, received, ctx())

    assert trainer.personal() is not None


def test_ditto_needs_the_received_state() -> None:
    data = SimpleNamespace(X=np.zeros((1, 16), np.float32), y=np.zeros(1, np.int64))

    with pytest.raises(TrainError, match="received"):
        trainers.create("ditto").train(build(), data, None, ctx())
```

- [ ] **Step 5: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_learning_trainers.py -q -k ditto`
Expected: FAIL with `PluginError: unknown trainer 'ditto'`. The `@real` tests skip without `data/samples/swell_real_sample.pkl`; build it first with `PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/create_real_samples.py` if `data/SWELL` and `data/WESAD` exist.

- [ ] **Step 6: Implement Ditto** (after the `FedProx` class)

```python
class DittoParams(StandardParams):
    lam: float = Field(
        0.1, ge=0, description="λ: cuánto se ata el modelo personal al global"
    )
    personal_epochs: PositiveInt | None = Field(
        None, description="Épocas del modelo personal; por defecto local_epochs"
    )


@trainers.register(
    "ditto",
    title="Ditto",
    description="Entrena el global como standard y, aparte, un modelo personal atado al global.",
    params=DittoParams,
    explain=(
        "El modelo personal minimiza su pérdida más λ/2·‖v − w‖² hacia el global "
        "recibido; cada edge lo conserva entre rondas y se puntúa como 'personal' "
        "(Li et al., 2021)."
    ),
)
class Ditto(Standard):
    Params = DittoParams

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._personal: nn.Module | None = None

    def train(
        self,
        model: nn.Module,
        data: Samples,
        received: Mapping[str, np.ndarray] | None = None,
        ctx: Any = None,
    ) -> TrainResult:
        if received is None:
            raise TrainError("ditto needs the received global state")
        if self._personal is None:
            self._personal = copy.deepcopy(model)  # the first global model
        result = super().train(model, data, received, ctx)
        p = self.params
        tied = FedProx(
            **p.model_dump(exclude={"lam", "personal_epochs", "local_epochs"}),
            local_epochs=p.personal_epochs or p.local_epochs,
            mu=p.lam,
        )
        own = tied.train(self._personal, data, received, ctx)
        return TrainResult(
            loss=result.loss,
            samples=result.samples + own.samples,
            examples=result.examples,
            batches=result.batches + own.batches,
        )

    def personal(self) -> nn.Module | None:
        return self._personal
```

- [ ] **Step 7: Run the trainer tests**

Run: `.venv/Scripts/python -m pytest tests/test_learning_trainers.py -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/onion_fl/learning/trainers.py tests/test_learning_trainers.py
git commit -m "feat(learning): Add the Ditto trainer"
```

---

### Task 5: The APFL trainer

**Files:**
- Modify: `src/onion_fl/learning/trainers.py`
- Test: `tests/test_learning_trainers.py`

**Interfaces:**
- Consumes: `_samples`, `_training`, `trainable`, `batches_of`, `OPTIMIZERS` (Task 4).
- Produces:
  - the trainer `apfl` with `APFLParams(StandardParams)`: `alpha: float = 0.5`, `adapt_alpha: bool = True`, `alpha_lr: float = 0.01`;
  - `APFL.alpha: float`, the current mixing weight;
  - `APFL.personal() -> nn.Module | None`, which returns `α·v + (1−α)·w`.

- [ ] **Step 1: Write the failing tests**

```python
@real
def test_apfl_with_alpha_zero_is_the_global_model(swell) -> None:
    model = build()
    trainer = trainers.create(
        "apfl", {"lr": 0.05, "alpha": 0.0, "adapt_alpha": False}
    )

    trainer.train(model, swell, state_arrays(model), ctx())

    for key, value in state_arrays(trainer.personal()).items():
        np.testing.assert_allclose(value, state_arrays(model)[key])


@real
def test_apfl_learns_its_alpha_within_bounds(swell) -> None:
    model = build()
    trainer = trainers.create(
        "apfl", {"local_epochs": 2, "lr": 0.05, "alpha": 0.5, "alpha_lr": 0.5}
    )

    result = trainer.train(model, swell, state_arrays(model), ctx())

    assert trainer.alpha != 0.5 and 0.0 <= trainer.alpha <= 1.0
    personal = state_arrays(trainer.personal())
    assert any(not np.allclose(personal[k], v) for k, v in state_arrays(model).items())
    assert result.samples == 2 * 2 * len(swell.y)  # two models per batch
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_learning_trainers.py -q -k apfl`
Expected: FAIL with `unknown trainer 'apfl'` (or SKIP without the sample; build it as in Task 4 Step 5).

- [ ] **Step 3: Implement APFL** (after `Ditto`; import `from torch.func import functional_call`)

```python
class APFLParams(StandardParams):
    alpha: float = Field(
        0.5, ge=0, le=1, description="Peso inicial del modelo personal en la mezcla"
    )
    adapt_alpha: bool = Field(True, description="Cada edge aprende su α")
    alpha_lr: float = Field(0.01, gt=0, description="Paso del descenso sobre α")


@trainers.register(
    "apfl",
    title="APFL",
    description="Mezcla un modelo personal con el global: α·v + (1−α)·w.",
    params=APFLParams,
    explain=(
        "En cada paso entrena el global w con sus datos y el personal v a través "
        "de la mezcla; con adapt_alpha cada edge ajusta su α (Deng et al., 2020)."
    ),
)
class APFL(Standard):
    Params = APFLParams

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.alpha = self.params.alpha
        self._v: nn.Module | None = None
        self._w: dict[str, torch.Tensor] = {}

    def train(
        self,
        model: nn.Module,
        data: Samples,
        received: Mapping[str, np.ndarray] | None = None,
        ctx: Any = None,
    ) -> TrainResult:
        xs, ys = _samples(data)
        p = self.params
        names = trainable(model, p.frozen)
        if not names:
            raise TrainError(f"every parameter is frozen by {p.frozen}")
        if self._v is None:
            self._v = copy.deepcopy(model)
        w, v = dict(model.named_parameters()), dict(self._v.named_parameters())
        make = OPTIMIZERS[p.optimizer]
        opt_w = make([w[n] for n in names], lr=p.lr, weight_decay=p.weight_decay)
        opt_v = make([v[n] for n in names], lr=p.lr, weight_decay=p.weight_decay)
        alpha = torch.tensor(self.alpha, requires_grad=p.adapt_alpha)
        rng = ctx.rng if ctx is not None else np.random.default_rng(0)
        total, batches = 0.0, 0
        with _training([model, self._v], names, rng):
            for _ in range(p.local_epochs):
                for idx in batches_of(len(xs), p.batch_size, rng):
                    opt_w.zero_grad()
                    loss = F.cross_entropy(model(xs[idx]), ys[idx])
                    loss.backward()
                    opt_w.step()
                    opt_v.zero_grad()
                    alpha.grad = None
                    mixed = {n: alpha * v[n] + (1 - alpha) * w[n].detach() for n in v}
                    out = functional_call(self._v, mixed, (xs[idx],))
                    F.cross_entropy(out, ys[idx]).backward()
                    opt_v.step()
                    if p.adapt_alpha:
                        with torch.no_grad():
                            alpha -= p.alpha_lr * alpha.grad
                            alpha.clamp_(0.0, 1.0)
                    total += float(loss.item()) * len(idx)
                    batches += 1
        self.alpha = float(alpha)
        self._w = {n: t.detach().clone() for n, t in w.items()}
        epochs_seen = p.local_epochs * len(xs)
        return TrainResult(
            loss=total / epochs_seen,
            samples=2 * epochs_seen,  # the global and the personal model per batch
            examples=len(xs),
            batches=batches,
        )

    def personal(self) -> nn.Module | None:
        if self._v is None:
            return None
        out = copy.deepcopy(self._v)
        with torch.no_grad():
            for name, tensor in out.named_parameters():
                tensor.copy_(self.alpha * tensor + (1 - self.alpha) * self._w[name])
        return out
```

- [ ] **Step 4: Run the trainer tests**

Run: `.venv/Scripts/python -m pytest tests/test_learning_trainers.py -q`
Expected: all pass. A `UserWarning` from `functional_call` would be ignored. Any other warning class fails the suite and must be fixed, not filtered.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/learning/trainers.py tests/test_learning_trainers.py
git commit -m "feat(learning): Add the APFL trainer"
```

---

### Task 6: FedRep, FedBABU and the trainer/sharing check

**Files:**
- Modify: `src/onion_fl/learning/trainers.py`, `src/onion_fl/learning/sharing.py`, `src/onion_fl/experiment/runner.py` (`_initial_state`)
- Test: `tests/test_learning_trainers.py`, `tests/test_learning_sharing.py`, `tests/test_experiment.py`

**Interfaces:**
- Produces:
  - the trainer `fedrep` with `FedRepParams(StandardParams)`: `head_epochs: PositiveInt = 5`, `head: list[str] = ["head*"]`; `local_epochs` is the number of body epochs;
  - the property `FedRep.local_groups -> list[str]`;
  - the trainer `fedbabu` with `FedBABUParams(StandardParams)`, where `frozen` defaults to `["head*"]`;
  - `not_local(policy, groups, patterns) -> list[str]` in `sharing.py`.

- [ ] **Step 1: Write the failing tests**

`tests/test_learning_trainers.py` (real-data section):

```python
@real
def test_fedrep_trains_the_head_then_the_body(swell) -> None:
    model = build()
    before = state_arrays(model)
    trainer = trainers.create(
        "fedrep", {"head_epochs": 2, "local_epochs": 1, "lr": 0.05}
    )

    result = trainer.train(model, swell, before, ctx())

    after = state_arrays(model)
    assert all(not np.array_equal(after[k], v) for k, v in before.items())
    assert result.samples == 3 * len(swell.y)
    assert trainer.local_groups == ["head*"]


@real
def test_fedbabu_leaves_the_head_as_it_was_initialised(swell) -> None:
    model = build()
    before = state_arrays(model)

    trainers.create("fedbabu", {"lr": 0.05}).train(model, swell, before, ctx())

    after = state_arrays(model)
    for key, value in before.items():
        assert np.array_equal(after[key], value) == key.startswith("head."), key
```

`tests/test_learning_sharing.py` (import `not_local` from `onion_fl.learning.sharing`):

```python
def test_groups_a_trainer_keeps_local_are_checked_against_the_policy() -> None:
    groups = ["adapter.swell", "trunk", "head.stress_binary"]

    assert not_local(sharing.create("fedper"), groups, ["head*"]) == []
    assert not_local(sharing.create("fedavg"), groups, ["head*"]) == [
        "head.stress_binary"
    ]
    assert not_local(sharing.create("fedavg"), groups, []) == []
```

`tests/test_experiment.py` (validation section):

```python
def test_fedrep_needs_sharing_that_keeps_the_heads_local(workspace: Path) -> None:
    learning = experiment(workspace)["learning"] | {"trainer": "fedrep"}

    with pytest.raises(ConfigError, match="fedrep"):
        plan(parse_experiment(experiment(workspace, learning=learning)))
    local = learning | {"sharing": "fedper"}
    assert plan(parse_experiment(experiment(workspace, learning=local)))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_learning_trainers.py tests/test_learning_sharing.py tests/test_experiment.py -q -k "fedrep or fedbabu or keeps_local or not_local"`
Expected: FAIL. The trainers are unknown and `not_local` cannot be imported.

- [ ] **Step 3: Implement `not_local`** (`sharing.py`, after `keys_held_at`)

```python
def not_local(
    policy: SharingPolicy, groups: Iterable[str], patterns: Sequence[str]
) -> list[str]:
    """Groups matching ``patterns`` that ``policy`` lets leave the edge."""
    return sorted(
        g
        for g in groups
        if any(fnmatch.fnmatchcase(g, p) for p in patterns)
        and policy.scope_of(g) != "local"
    )
```

- [ ] **Step 4: Implement FedRep and FedBABU** (`trainers.py`, after `APFL`; `fnmatch` and `group_of` are already imported)

```python
class FedRepParams(StandardParams):
    head_epochs: PositiveInt = Field(
        5, description="Épocas de la cabeza con el cuerpo congelado"
    )
    head: list[str] = Field(
        default_factory=lambda: ["head*"],
        description="Grupos que forman la cabeza, por nombre o patrón",
    )


@trainers.register(
    "fedrep",
    title="FedRep",
    description="Entrena primero la cabeza local y después el cuerpo compartido (local_epochs).",
    params=FedRepParams,
    explain=(
        "La cabeza se queda en el edge: exige una compartición que la mantenga "
        "local, como fedper; solo viaja la representación (Collins et al., 2021)."
    ),
)
class FedRep(Standard):
    Params = FedRepParams

    @property
    def local_groups(self) -> list[str]:
        return list(self.params.head)

    def train(
        self,
        model: nn.Module,
        data: Samples,
        received: Mapping[str, np.ndarray] | None = None,
        ctx: Any = None,
    ) -> TrainResult:
        p = self.params
        base = p.model_dump(exclude={"head_epochs", "head", "local_epochs", "frozen"})
        groups = {group_of(name) for name, _ in model.named_parameters()}
        body = sorted(
            g for g in groups if not any(fnmatch.fnmatchcase(g, h) for h in p.head)
        )
        head = Standard(
            **base, local_epochs=p.head_epochs, frozen=[*p.frozen, *body]
        ).train(model, data, received, ctx)
        rest = Standard(
            **base, local_epochs=p.local_epochs, frozen=[*p.frozen, *p.head]
        ).train(model, data, received, ctx)
        return TrainResult(
            loss=rest.loss,
            samples=head.samples + rest.samples,
            examples=rest.examples,
            batches=head.batches + rest.batches,
        )


class FedBABUParams(StandardParams):
    frozen: list[str] = Field(
        default_factory=lambda: ["head*"],
        description="Grupos congelados; por defecto la cabeza, que no se entrena",
    )


@trainers.register(
    "fedbabu",
    title="FedBABU",
    description="Solo aprende el cuerpo; la cabeza se queda como se inicializó.",
    params=FedBABUParams,
    explain=(
        "Se evalúa ajustando el modelo recibido con evaluation.edge.finetune y "
        "puntuándolo como 'finetuned' (Oh et al., 2022)."
    ),
)
class FedBABU(Standard):
    Params = FedBABUParams
```

- [ ] **Step 5: Check the pair at plan and build time** (`runner.py`)

Add `not_local` to the `onion_fl.learning.sharing` import, and `ConfigError` to the `onion_fl.experiment.config` import if absent. Replace `_initial_state`:

```python
def _initial_state(config: ExperimentConfig, shapes: Sequence[Any], seed: int):
    family = create(models, config.learning.model)
    model = family.build(shapes, seed=seed)
    create(inits, config.learning.init).init(model)
    state = state_arrays(model)
    _check_local_groups(config, state)
    return family, state


def _check_local_groups(config: ExperimentConfig, state: Mapping[str, Any]) -> None:
    """A trainer that keeps groups on the edge (FedRep) needs sharing that keeps them."""
    trainer = create(trainers, config.learning.trainer)
    patterns = list(getattr(trainer, "local_groups", ()))
    policy = create(sharing, config.learning.sharing)
    leaving = not_local(policy, param_groups(state), patterns)
    if leaving:
        ref = config.learning.trainer
        name = ref if isinstance(ref, str) else ref["name"]
        raise ConfigError(
            f"learning.trainer: {name} keeps {patterns} on the edge, but sharing "
            f"{policy.name!r} sends {leaving} up; use fedper or a custom rule "
            "that keeps them local"
        )
```

(`Mapping` comes from `collections.abc`; add it to the existing import.)

- [ ] **Step 6: Run the affected tests**

Run: `.venv/Scripts/python -m pytest tests/test_learning_trainers.py tests/test_learning_sharing.py tests/test_experiment.py tests/test_cli.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/onion_fl/learning/trainers.py src/onion_fl/learning/sharing.py src/onion_fl/experiment/runner.py tests/test_learning_trainers.py tests/test_learning_sharing.py tests/test_experiment.py
git commit -m "feat(learning): Add the FedRep and FedBABU trainers

FedRep declares the groups it keeps on the edge; plan and build refuse a
sharing policy that sends them up, naming the trainer."
```

---

### Task 7: The LG-FedAvg sharing preset

**Files:**
- Modify: `src/onion_fl/learning/sharing.py` (presets)
- Test: `tests/test_learning_sharing.py`

**Interfaces:**
- Produces: the sharing preset `lg_fedavg`, which keeps adapters and trunk local and heads global.

- [ ] **Step 1: Write the failing tests** — update the registry list and add a scope test

```python
def test_every_preset_and_custom_are_registered() -> None:
    assert sharing.names() == [
        "custom",
        "fedavg",
        "fedper",
        "harmonized",
        "independent",
        "lg_fedavg",
        "zone",
    ]


def test_lg_fedavg_keeps_the_representation_local() -> None:
    assert scopes(sharing.create("lg_fedavg")) == {
        "adapter.swell": "local",
        "trunk": "local",
        "head.stress_binary": "global",
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_learning_sharing.py -q`
Expected: FAIL, because `lg_fedavg` is not registered.

- [ ] **Step 3: Implement** (after the `fedper` preset)

```python
@sharing.register(
    "lg_fedavg",
    title="LG-FedAvg",
    description="Adaptadores y tronco locales; las cabezas se agregan globalmente.",
    explain="Cada edge aprende su representación y solo comparte la cabeza (Liang et al., 2020).",
)
def lg_fedavg() -> SharingPolicy:
    return SharingPolicy(
        name="lg_fedavg", rules={"adapter*": "local", "trunk*": "local"}
    )
```

- [ ] **Step 4: Run the sharing tests**

Run: `.venv/Scripts/python -m pytest tests/test_learning_sharing.py tests/test_studio.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/learning/sharing.py tests/test_learning_sharing.py
git commit -m "feat(learning): Add the LG-FedAvg sharing preset"
```

---

### Task 8: Comparison experiment, results and docs

**Files:**
- Create: `experiments/techniques_personalisation.yaml`
- Create: `results/techniques_personalisation/` (`<run_id>/run.json|summary.json|model.npz`, `report.html`, `INDEX.md`)
- Modify: `README.md` (§1 Learning row and test count, §4, §7), `docs/architecture.md` (learning section and plugin table)
- Test: `tests/test_cli.py` (`test_every_shipped_experiment_and_its_topology_are_valid` already covers the new file)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the experiment**

```yaml
# Personalisation techniques on SWELL and WESAD (issue #147): one scenario per
# technique, three seeds. Every edge scores on its own class-stratified rows
# the model it received, the one it trained, its personal model (Ditto, APFL)
# and the received model after two epochs of fine-tuning.
#
#   onion_fl plan experiments/techniques_personalisation.yaml
#   onion_fl run experiments/techniques_personalisation.yaml --workers 3
#   onion_fl report experiments/techniques_personalisation.yaml --by scenario
name: techniques_personalisation
description: Personalización (FedAvg, FedPer, LG-FedAvg, Ditto, APFL, FedRep, FedBABU) sobre SWELL y WESAD.
topology: four_fogs_swell_wesad
data:
  datasets:
    swell: {}
    wesad: {}
  roles: {test: 0.2, val: 0.1, local_val: 0.2, local_val_split: class_tail, scaler: global, seed: 0}
  placement: {name: mixing, alpha: 0.5}
learning:
  model: {name: modular_mlp, adapter_width: 64, trunk_hidden: [64, 32], dropout: 0.2}
  sharing: fedavg
  trainer: {name: standard, local_epochs: 10, batch_size: 32, lr: 0.003}
  init: random
rounds: 20
evaluation:
  metrics: [loss, accuracy, macro_f1]
  edge:
    every: 5
    models: [received, local, personal, finetuned]
    finetune: {name: standard, local_epochs: 2, batch_size: 32, lr: 0.003}
  aggregators: {every: 5}
  global: {every: 1}
seeds: [0, 1, 2]
sweep:
  learning.sharing,learning.trainer.name:
    - [fedavg, standard]      # FedAvg
    - [fedper, standard]      # FedPer
    - [lg_fedavg, standard]   # LG-FedAvg
    - [fedavg, ditto]
    - [fedavg, apfl]
    - [fedper, fedrep]
    - [fedavg, fedbabu]
```

- [ ] **Step 2: Plan it and check the shipped-experiments test**

Run: `.venv/Scripts/python -m onion_fl plan experiments/techniques_personalisation.yaml > $TEMP/plan_p1.json; .venv/Scripts/python -m pytest tests/test_cli.py -q -k shipped`
Expected: the plan exits 0 with 21 entries (7 scenarios × 3 seeds) and no warnings; the test passes.

- [ ] **Step 3: Commit the experiment, then run it on a clean tree**

```bash
git add experiments/techniques_personalisation.yaml
git commit -m "feat(experiments): Add the personalisation comparison"
git status --short   # must be empty: a dirty tree signs the runs dirty
.venv/Scripts/python -m onion_fl run experiments/techniques_personalisation.yaml --workers 3
```

Expected: 21 run folders under `runs/`, each `finished`, `code_version.dirty == false`, and `verify_run` true.

- [ ] **Step 4: Summarise and store the artifact**

Run this script from the repo root (save it in the scratchpad, not in the repo):

```python
import json, shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from onion_fl.observability.run import verify_run

OUT = Path("results/techniques_personalisation")
rows = defaultdict(lambda: defaultdict(list))
for run in sorted(Path("runs").iterdir()):
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if meta["config"]["name"] != "techniques_personalisation":
        continue
    assert verify_run(run) and meta["status"] == "finished", run
    assert not meta["code_version"]["dirty"], run
    final = json.loads((run / "summary.json").read_text(encoding="utf-8"))["final"]
    row = rows[meta["scenario"]]
    for ds in ("swell", "wesad"):
        row[f"global {ds}"].append(final[f"global/{ds}"]["macro_f1"])
    for line in (run / "events.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        tags = e.get("tags") or {}
        if (e["node"], e["name"], tags.get("source"), e.get("round")) == (
            "cloud", "eval.macro_f1", "children", 20
        ):
            row[f"edge {tags['model']}"].append(e["value"])
    dest = OUT / run.name
    dest.mkdir(parents=True, exist_ok=True)
    meta.pop("host", None), meta.pop("pid", None)
    (dest / "run.json").write_text(json.dumps(meta, indent=2) + "
", encoding="utf-8")
    for name in ("summary.json", "model.npz"):
        shutil.copy(run / name, dest / name)


def ci(values):
    v = np.asarray(values)
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():.3f} ± {half:.3f}"


columns = ["global swell", "global wesad", "edge local", "edge personal", "edge finetuned"]
print("| Scenario | " + " | ".join(columns) + " |")
for scenario, row in rows.items():
    cells = [ci(row[c]) if len(row[c]) > 1 else "—" for c in columns]
    print(f"| {scenario.split('=')[1]} | " + " | ".join(cells) + " |")
```

Then run `.venv/Scripts/python -m onion_fl report runs --out results/techniques_personalisation/report.html --metric macro_f1 --by scenario`, and check that `grep -r "$(hostname)" results/techniques_personalisation` finds nothing.

Write `INDEX.md` in the same shape as `results/mix_swell_wesad/INDEX.md`, with the printed table:
- the commands, the setup and the identity table (`topology_id`, one `config_id` per scenario, `data_id`, the commit);
- a reading of the results, reporting what the numbers show, including if personalisation does not help. With `fedper`, `lg_fedavg` and `fedrep`, the global model has untrained local groups, so read their edge scores, not the global ones;
- the files list.

- [ ] **Step 5: Update the docs**

- README §1 "Learning" row: list the trainers `standard`, `fedprox`, `ditto`, `apfl`, `fedrep`, `fedbabu` and the `lg_fedavg` preset, plus personal and fine-tuned scores.
- README §1 "Tests" row: the new counts from the full check.
- README §4: one bullet for joint sweep keys (`a,b: [[x, y]]`) and `local_val_split: class_tail`.
- README §7: a subsection "New framework: personalisation techniques" with the table, citing `results/techniques_personalisation/`.
- `docs/architecture.md`:
  - in the learning section, add the new trainers and preset, and the `personal`/`finetuned` scores;
  - in the plugin table, the `trainer` row and the `sharing` row.

- [ ] **Step 6: Full check and commit**

Run: `.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check . && .venv/Scripts/python -m pytest -q`
Expected: all pass.

```bash
git add results/techniques_personalisation README.md docs/architecture.md
git commit -m "docs(results): Add the personalisation techniques comparison"
```

- [ ] **Step 7: Move the unused core extensions to their consumers**

```python
import sys
sys.path.insert(0, ".claude/skills/tarea-github/scripts")
import gh

R = f"/repos/{gh.REPO}/issues"
moves = {
    148: "- [ ] Núcleo (§3.1–3.2): claves auxiliares `<algoritmo>/<clave>`, `TrainResult.aux` y `steps`, y estadísticos de ronda para el optimizador de servidor.
",
    149: "- [ ] Núcleo (§3.3): referencia para los agregadores (el estado que el nodo envió).
",
}
for number, line in moves.items():
    issue, _ = gh.api("GET", f"{R}/{number}")
    body = issue["body"].replace("
- [ ] ", "
" + line + "- [ ] ", 1)
    gh.api("PATCH", f"{R}/{number}", {"body": body})
issue, _ = gh.api("GET", f"{R}/147")
body = issue["body"].replace(
    "- [ ] Núcleo: claves auxiliares `<algoritmo>/<clave>`, `TrainResult.aux` y `steps`, estadísticos de ronda para el optimizador de servidor, referencia para los agregadores.",
    "- [x] Núcleo: movido a #148 (§3.1–3.2) y #149 (§3.3), donde tienen consumidor.",
).replace("- [ ] ", "- [x] ")
gh.api("PATCH", f"{R}/147", {"body": body})
```

- [ ] **Step 8: Push, PR, merge and close with evidence** (tarea-github skill)

```bash
git push origin "HEAD:refs/heads/task/#147"
.venv/Scripts/python .claude/skills/tarea-github/scripts/gh.py pr 147 --note "<ruff and pytest counts, data and broker used>"
.venv/Scripts/python .claude/skills/tarea-github/scripts/gh.py merge <P>
.venv/Scripts/python .claude/skills/tarea-github/scripts/gh.py close 147 --pr <P> --notes <scratchpad>/cierre.md
```

`cierre.md` has `## Qué se ha hecho`, `## Verificación` (counts, data, `results/techniques_personalisation/` with its ids and commit) and `## Pendiente` (§3.1–3.3 moved to #148/#149).
