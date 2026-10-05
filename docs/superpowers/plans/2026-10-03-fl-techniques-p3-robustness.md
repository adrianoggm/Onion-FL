# FL techniques P3 (robustness, attacks, privacy) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Robust aggregators, malicious edges and differential privacy can be chosen by name and compared: accuracy under attack, detection of the attackers and the privacy budget ε (issue #149).

**Architecture:**
- **Aggregator calls (spec §3.3).** Aggregators now receive the state the node sent down that round (`reference`) and their own random stream (`rng`). They can return report items, which the collector emits as events. For items naming dropped children, the collector adds how many were malicious.
- **New aggregators.** Krum, Multi-Krum, Bulyan and the geometric median score whole update vectors over the model keys every child holds, then combine per key. Norm clipping and central DP clip each child's update against `reference`.
- **Attacks.** An `attack` plugin is attached to a seeded fraction of each dataset's edges, with hooks on the data and on the update.
- **Local DP.** A `privacy` plugin adds clipping and noise at the edge. Both kinds of DP report ε from a closed-form RDP accountant.
- **Experiment override.** `learning.aggregator` sets the aggregator of the leaf aggregators (the ones whose children are edges).

**Tech Stack:** Python 3.11, NumPy, PyTorch (unchanged models), pydantic 2.13, pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-fl-techniques-design.md` (§3.3, §3.6, §3.7, §4 P3).

**Branching:** the work is on `task/#149`, created from `task/#148`. That branch is under review in PR #154, so the PR for this one targets `task/#148`, and GitHub retargets it to `develop` once #154 merges.

## Global Constraints

- **Environment.** Python ≥ 3.11; use `.venv/Scripts/python` and `.venv/Scripts/ruff`.
- **Lint and warnings.** ruff with 88 columns. Modules start with `from __future__ import annotations`. pytest runs with `filterwarnings = error`.
- **Language.** Plugin `title`, `description`, `explain` and parameter descriptions are in Spanish. Code, comments and docs are in English.
- **No synthetic data for ML.**
  - Aggregator and attack maths is tested on hand-written arrays.
  - Protocol tests use stub trainers.
  - `label_flip` corrupts real labels to simulate an attacker. It is not synthetic training data, and docs/RULES.md must say so.
- **Stable `config_id`s.** New config fields stay out of the dump when unset (`exclude_if`).
- **Honest results.**
  - Hyperparameters are chosen on validation subjects or on training statistics, never on test scores.
  - Every cited number is in a committed CSV.
  - Edge and selection numbers come from the events of the edges and aggregators that produced them.
- **Commits and workflow.** Commit subjects are `type(scope): Imperative summary in English`, with no `Co-Authored-By`.

## Review Focus

1. **A robust aggregator with fewer children than its `f` needs** (lossy rounds) must not crash the node. It lowers `f` to the largest feasible value and reports `f_used`. Pinned in Task 2.
2. **Children holding different dataset-specific keys.** Selection scores use the model keys every child holds; each key is then combined over the selected children that hold it, and no key is dropped silently. Pinned in Task 2.
3. **Auxiliary arrays (SCAFFOLD's `c`) pass through robust and DP aggregators unscored and uncorrupted.** They are averaged over the selected children and are neither clipped nor noised. Pinned in Tasks 2 and 3.
4. **Attacks only touch model keys that cross the link.** Local groups and auxiliary arrays are left alone. Pinned in Task 4.
5. **The ε of a run is reported per round and grows with the rounds actually applied,** not the rounds started. Pinned in Tasks 3 and 5.

---

### Task 1: Aggregators get the reference, a stream, and a voice

**Files:**
- Modify: `src/onion_fl/learning/aggregators.py` (signatures of the built-in aggregators)
- Modify: `src/onion_fl/roles/nodes.py` (`_Collector._close`)
- Modify: `src/onion_fl/experiment/config.py` (`LearningConfig.aggregator`), `src/onion_fl/experiment/runner.py` (`resolve_topology`)
- Test: `tests/test_roles_protocol.py`, `tests/test_experiment.py`

**Interfaces:**
- Produces:
  - `aggregate(contributions, source, reference=None, rng=None) -> Contribution` on every aggregator;
  - an optional `report() -> list[tuple[str, float, dict]]`, called right after `aggregate`, whose items the collector emits as `ctx.emit(name, value, round=..., **tags)`;
  - for an item whose tags hold `dropped: list[str]`, the collector adds `malicious_dropped` and `malicious`, counting children whose hello tags say `malicious: true`;
  - `LearningConfig.aggregator: PluginRef | None`, which sets the leaf aggregators' `aggregator` setting and stays out of the dump when unset.

- [ ] **Step 1: Write the failing tests**

`tests/test_roles_protocol.py`:

```python
class Spy:
    """FedAvg that records what it is given and reports one dropped child."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict, object]] = []

    def aggregate(self, contributions, source, reference=None, rng=None):
        self.calls.append((dict(reference or {}), rng))
        return aggregators.create("fedavg").aggregate(contributions, source)

    def report(self):
        return [("aggregation.dropped", 1.0, {"dropped": ["e2"]})]


def test_aggregators_get_the_reference_and_a_stream_and_are_heard(monkeypatch) -> None:
    spy = Spy()
    monkeypatch.setattr(
        "onion_fl.roles.federation._round_settings",
        _with_aggregator(spy),
    )
    edges = {
        "fog_0": [
            edge("e1", shift=1),
            EdgeSpec("e2", model(A), trainer=trainers.create("stub"), tags={"malicious": True}),
        ]
    }

    federation = run(tree(1), edges)

    reference, rng = spy.calls[0]
    assert set(reference) == set(INITIAL) and rng is not None
    (dropped,) = names(federation, "aggregation.dropped", "fog_0")
    assert dropped["tags"]["malicious_dropped"] == 1
    assert dropped["tags"]["malicious"] == 1
```

`_with_aggregator(spy)` wraps the real `_round_settings` so that every node's `aggregator` is `spy`:

```python
def _with_aggregator(spy):
    from onion_fl.roles import federation as module

    original = module._round_settings

    def settings(raw):
        return original(raw) | {"aggregator": spy}

    return settings
```

(the root and the fog both use the spy; the assertions read the fog's events.)

`tests/test_experiment.py`:

```python
def test_the_experiment_can_set_the_leaf_aggregators(workspace: Path) -> None:
    from onion_fl.experiment.runner import resolve_topology

    learning = experiment(workspace)["learning"] | {"aggregator": "median"}
    topology = resolve_topology(parse_experiment(experiment(workspace, learning=learning)))

    leaves = {leaf.id for leaf in topology.leaves()}
    for node in topology.nodes:
        expected = "median" if node.id in leaves else None
        assert node.settings.get("aggregator") == expected, node.id


def test_an_unset_aggregator_stays_out_of_the_config(workspace: Path) -> None:
    assert "aggregator" not in parse_experiment(experiment(workspace)).dump()["learning"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_roles_protocol.py tests/test_experiment.py -q -k "reference_and_a_stream or leaf_aggregators or unset_aggregator"`
Expected: FAIL. The collector calls `aggregate(contributions, source=...)` without `reference`, and `learning.aggregator` is an unknown field.

- [ ] **Step 3: Implement**

`aggregators.py`: give `FedAvg`, `Mean`, `Median` and `TrimmedMean` the signature

```python
    def aggregate(
        self,
        contributions: Sequence[Contribution],
        source: str,
        reference: Mapping[str, np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ) -> Contribution:
```

(they ignore the two new arguments).

`nodes.py`, `_Collector._close`, replacing the `aggregate` call:

```python
        aggregated = self.aggregator.aggregate(
            [*fresh.values(), *stale],
            source=self.id,
            reference=self.sent,
            rng=child_rng(ctx.rng),
        )
        self._report_aggregation(ctx)
```

and add to `_Collector`:

```python
    def _report_aggregation(self, ctx: Context) -> None:
        """Emit what the aggregator reports; dropped children get their malicious count."""
        malicious = {
            child
            for child, meta in self.registered.items()
            if (meta.get("tags") or {}).get("malicious")
        }
        for name, value, tags in getattr(self.aggregator, "report", list)():
            tags = dict(tags)
            if "dropped" in tags:
                tags["malicious_dropped"] = len(set(tags["dropped"]) & malicious)
                tags["malicious"] = len(malicious & set(self.participants))
            ctx.emit(name, value, round=self.round, **tags)
```

`config.py`, in `LearningConfig`:

```python
    aggregator: PluginRef | None = Field(
        None,
        description="Agregador de los nodos cuyos hijos son edges; sin él, el de la topología",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )

    _aggregator = field_validator("aggregator")(
        lambda v: v if v is None else _plugin(aggregators, v)
    )
```

(import `aggregators` from `onion_fl.learning.aggregators` alongside `server_optimizers`.)

`runner.py`, in `resolve_topology` next to the server-optimizer override:

```python
    if config.learning.aggregator is not None:
        parents = {n["parent"] for n in general["nodes"]}
        for node in general["nodes"]:
            if node["id"] not in parents:  # a leaf aggregator: its children are edges
                node["settings"]["aggregator"] = config.learning.aggregator
```

- [ ] **Step 4: Run the suites**

Run: `.venv/Scripts/python -m pytest tests/test_roles_protocol.py tests/test_roles_evaluation.py tests/test_learning_aggregators.py tests/test_experiment.py tests/test_runtime_equivalence.py -q`
Expected: all pass.

- [ ] **Step 5: Commit** `feat(roles): Give aggregators the reference, a stream and a report`

---

### Task 2: Selection-based and median aggregators

**Files:**
- Modify: `src/onion_fl/learning/aggregators.py`
- Test: `tests/test_learning_aggregators.py`

**Interfaces:**
- Produces:
  - the aggregators `krum(f=1)`, `multi_krum(f=1, m=None)`, `geometric_median(iterations=10, eps=1e-6)` and `bulyan(f=1)`;
  - `report()` on the selection aggregators, giving `("aggregation.dropped", n, {"dropped": [...], "f_used": f})`.

- [ ] **Step 1: Write the failing tests** (hand-written arrays; `c(source, w, n=1)` builds a `Contribution` with state `{"w": np.array(w)}` and weights `{"w": n}`; add it to the test file if absent)

```python
def c(source: str, w, n: float = 1.0, **extra) -> Contribution:
    state = {"w": np.asarray(w, dtype=np.float64)} | {k: np.asarray(v, np.float64) for k, v in extra.items()}
    return Contribution(source, state, dict.fromkeys(state, n))


HONEST = [c(f"h{i}", [1.0 + 0.1 * i, 1.0]) for i in range(5)]
BYZANTINE = [c("b0", [50.0, -50.0]), c("b1", [60.0, -40.0])]


def test_krum_picks_an_honest_update() -> None:
    krum = aggregators.create("krum", {"f": 2})

    out = krum.aggregate(HONEST + BYZANTINE, "fog")

    assert out.state["w"][0] < 2.0
    (item,) = krum.report()
    assert {"b0", "b1"} <= set(item[2]["dropped"]) and item[2]["f_used"] == 2


def test_multi_krum_averages_the_m_best() -> None:
    out = aggregators.create("multi_krum", {"f": 2, "m": 5}).aggregate(HONEST + BYZANTINE, "fog")

    np.testing.assert_allclose(out.state["w"], np.mean([h.state["w"] for h in HONEST], axis=0))


def test_too_few_children_lower_f_instead_of_failing() -> None:
    krum = aggregators.create("krum", {"f": 2})

    krum.aggregate(HONEST[:3], "fog")  # n=3 fits f=0 only (n >= 2f + 3)

    assert krum.report()[0][2]["f_used"] == 0


def test_the_geometric_median_resists_an_outlier() -> None:
    points = [c("a", [0.0, 0.0]), c("b", [1.0, 0.0]), c("c", [0.0, 1.0]), c("d", [100.0, 100.0])]

    out = aggregators.create("geometric_median").aggregate(points, "fog")

    assert np.linalg.norm(out.state["w"]) < 1.0


def test_bulyan_bounds_an_extreme_coordinate() -> None:
    children = HONEST + [c("b0", [1000.0, 1.0])]  # n=6 fits f=0 (n >= 4f + 3 needs 7 for f=1)
    out = aggregators.create("bulyan", {"f": 1}).aggregate(children + [c("h5", [1.2, 1.0])], "fog")

    assert out.state["w"][0] < 2.0


def test_selection_scores_common_keys_and_combines_every_key() -> None:
    children = [
        c("s0", [1.0], adapter_s=[1.0]),
        c("s1", [1.1], adapter_s=[3.0]),
        c("t0", [0.9], adapter_t=[5.0]),
        c("bad", [90.0], adapter_t=[7.0]),
    ]

    out = aggregators.create("multi_krum", {"f": 1, "m": 3}).aggregate(children, "fog")

    np.testing.assert_allclose(out.state["adapter_s"], [2.0])  # both holders selected
    np.testing.assert_allclose(out.state["adapter_t"], [5.0])  # only the honest holder


def test_auxiliary_arrays_follow_the_selected_children_unscored() -> None:
    children = HONEST + [c("b0", [50.0, -50.0])]
    for child in children:
        child.state["scaffold/w"] = np.full(2, 1.0 if child.source != "b0" else 500.0)
        child.weights["scaffold/w"] = 1.0

    out = aggregators.create("multi_krum", {"f": 1, "m": 5}).aggregate(children, "fog")

    np.testing.assert_allclose(out.state["scaffold/w"], [1.0, 1.0])
```

(`c` builds plain dicts, so assigning extra keys after creation is fine. If `Contribution` freezes its mappings, build the children with the aux key passed through `**extra` instead, as `scaffold_w` named arguments cannot contain `/`: then construct those four `Contribution`s explicitly.)

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_learning_aggregators.py -q -k "krum or median or bulyan or selection or auxiliary_arrays_follow"`
Expected: FAIL with unknown aggregator plugins.

- [ ] **Step 3: Implement** (in `aggregators.py`, after `TrimmedMean`)

```python
def _common_model_keys(ordered: Sequence[Contribution]) -> list[str]:
    keys = set.intersection(*(set(c.state) for c in ordered)) if ordered else set()
    return sorted(k for k in keys if not is_aux(k))


def _vectors(
    ordered: Sequence[Contribution],
    keys: Sequence[str],
    reference: Mapping[str, np.ndarray] | None,
) -> np.ndarray:
    """One flattened update per child over ``keys``: its delta against ``reference``."""
    rows = []
    for item in ordered:
        parts = []
        for key in keys:
            value = np.asarray(item.state[key], dtype=np.float64)
            if reference is not None and key in reference:
                value = value - np.asarray(reference[key], dtype=np.float64)
            parts.append(np.ravel(value))
        rows.append(np.concatenate(parts) if parts else np.zeros(0))
    return np.stack(rows)


def _krum_scores(vectors: np.ndarray, f: int) -> np.ndarray:
    n = len(vectors)
    nearest = max(n - f - 2, 1)
    distances = ((vectors[:, None, :] - vectors[None, :, :]) ** 2).sum(axis=-1)
    return np.array(
        [np.sort(np.delete(distances[i], i))[:nearest].sum() for i in range(n)]
    )


class _Selection:
    """Keeps the last selection so the collector can report it."""

    dropped: list[str]
    f_used: int

    def report(self) -> list[tuple[str, float, dict[str, Any]]]:
        return [
            (
                "aggregation.dropped",
                float(len(self.dropped)),
                {"dropped": list(self.dropped), "f_used": self.f_used},
            )
        ]

    def _keep(self, ordered, chosen, source) -> Contribution:
        kept = [ordered[i] for i in sorted(chosen)]
        self.dropped = [c.source for i, c in enumerate(ordered) if i not in set(chosen)]
        return _combine_per_key(kept, source, _weighted_mean, needs_weights=True)


class KrumParams(BaseModel):
    f: int = Field(1, ge=0, description="Hijos maliciosos que tolera")


@aggregators.register(
    "krum",
    title="Krum",
    description="Elige la actualización con menor suma de distancias a sus n − f − 2 vecinas.",
    params=KrumParams,
    explain=(
        "Puntúa sobre las claves que tienen todos los hijos y conserva una sola "
        "contribución; con menos de 2f + 3 hijos baja f (Blanchard et al., 2017)."
    ),
)
class Krum(_Selection):
    def __init__(self, f: int = 1) -> None:
        self.f = f

    def _feasible(self, n: int) -> int:
        return max(0, min(self.f, (n - 3) // 2))

    def _ranked(self, ordered, reference) -> tuple[np.ndarray, int]:
        f = self._feasible(len(ordered))
        keys = _common_model_keys(ordered)
        scores = _krum_scores(_vectors(ordered, keys, reference), f)
        return np.argsort(scores, kind="stable"), f

    def aggregate(self, contributions, source, reference=None, rng=None) -> Contribution:
        ordered = sorted(contributions, key=lambda item: item.source)
        ranked, self.f_used = self._ranked(ordered, reference)
        return self._keep(ordered, ranked[:1], source)


class MultiKrumParams(KrumParams):
    m: PositiveInt | None = Field(None, description="Cuántas conserva; por defecto n − f")


@aggregators.register(
    "multi_krum",
    title="Multi-Krum",
    description="Promedia (FedAvg) las m actualizaciones mejor puntuadas por Krum.",
    params=MultiKrumParams,
    explain="Con menos de 2f + 3 hijos baja f (Blanchard et al., 2017).",
)
class MultiKrum(Krum):
    def __init__(self, f: int = 1, m: int | None = None) -> None:
        super().__init__(f)
        self.m = m

    def aggregate(self, contributions, source, reference=None, rng=None) -> Contribution:
        ordered = sorted(contributions, key=lambda item: item.source)
        ranked, self.f_used = self._ranked(ordered, reference)
        keep = min(self.m or len(ordered) - self.f_used, len(ordered))
        return self._keep(ordered, ranked[:keep], source)
```

Geometric median: Weiszfeld over the common-key vectors, weighted by sample weight. Its final weights are then applied to every key over that key's holders:

```python
class GeometricMedianParams(BaseModel):
    iterations: PositiveInt = Field(10, description="Iteraciones de Weiszfeld")
    eps: float = Field(1e-6, gt=0, description="Distancia mínima (evita dividir por cero)")


@aggregators.register(
    "geometric_median",
    title="Mediana geométrica",
    description="Punto que minimiza la suma de distancias a las actualizaciones (Weiszfeld).",
    params=GeometricMedianParams,
    explain=(
        "Pesa cada hijo por muestras / distancia a la mediana y aplica esos pesos a "
        "todas las claves (RFA, Pillutla et al., 2022)."
    ),
)
class GeometricMedian:
    def __init__(self, iterations: int = 10, eps: float = 1e-6) -> None:
        self.iterations, self.eps = iterations, eps

    def aggregate(self, contributions, source, reference=None, rng=None) -> Contribution:
        ordered = sorted(contributions, key=lambda item: item.source)
        keys = _common_model_keys(ordered)
        vectors = _vectors(ordered, keys, reference)
        alpha = np.array(
            [float(c.weights.get(keys[0], 1.0)) if keys else 1.0 for c in ordered]
        )
        beta = alpha / alpha.sum()
        for _ in range(self.iterations):
            z = (beta[:, None] * vectors).sum(axis=0)
            distance = np.maximum(np.linalg.norm(vectors - z, axis=1), self.eps)
            beta = alpha / distance
            beta = beta / beta.sum()
        weighted = [
            Contribution(item.source, item.state, dict.fromkeys(item.state, float(b)))
            for item, b in zip(ordered, beta, strict=True)
        ]
        out = _combine_per_key(weighted, source, _weighted_mean, needs_weights=True)
        totals = _combine_per_key(ordered, source, _weighted_mean, needs_weights=True)
        return Contribution(source, out.state, totals.weights)  # sample weights go up
```

Bulyan: Multi-Krum keeps θ = n − 2f, then each coordinate is the mean of the β = θ − 2f values closest to its median. `f` drops until n ≥ 4f + 3:

```python
@aggregators.register(
    "bulyan",
    title="Bulyan",
    description="Multi-Krum y después media recortada alrededor de la mediana, coordenada a coordenada.",
    params=KrumParams,
    explain="Con menos de 4f + 3 hijos baja f (El Mhamdi et al., 2018).",
)
class Bulyan(Krum):
    def _feasible(self, n: int) -> int:
        return max(0, min(self.f, (n - 3) // 4))

    def aggregate(self, contributions, source, reference=None, rng=None) -> Contribution:
        ordered = sorted(contributions, key=lambda item: item.source)
        ranked, f = self._ranked(ordered, reference)
        self.f_used = f
        theta = len(ordered) - 2 * f
        chosen = sorted(ranked[:theta])
        selected = [ordered[i] for i in chosen]
        self.dropped = [c.source for i, c in enumerate(ordered) if i not in set(chosen)]
        beta = max(theta - 2 * f, 1)

        def around_median(stacked: np.ndarray, _weights: np.ndarray) -> np.ndarray:
            keep = min(beta, stacked.shape[0])
            median = np.median(stacked, axis=0)
            order = np.argsort(np.abs(stacked - median), axis=0, kind="stable")[:keep]
            return np.take_along_axis(stacked, order, axis=0).mean(axis=0)

        out = _combine_per_key(selected, source, around_median, needs_weights=False)
        totals = _combine_per_key(selected, source, _weighted_mean, needs_weights=True)
        return Contribution(source, out.state, totals.weights)
```

(the auxiliary keys follow the selected children: Krum/Multi-Krum combine them with FedAvg, while Bulyan applies the same median trimming to them, which a test does not pin; note it in `explain` if it matters.)

- [ ] **Step 4: Run the aggregator suite** (`.venv/Scripts/python -m pytest tests/test_learning_aggregators.py -q`). Also update `test_registries_list_the_built_ins` with the new names, and make `test_optimizers_keep_the_dtype`'s sibling for aggregators (if any) accept the new ones.

- [ ] **Step 5: Commit** `feat(learning): Add Krum, Multi-Krum, Bulyan and the geometric median`

---

### Task 3: Norm clipping, central DP and the ε accountant

**Files:**
- Create: `src/onion_fl/learning/privacy.py` (`gaussian_epsilon`, the `privacies` registry and `local_dp` for Task 5)
- Modify: `src/onion_fl/learning/aggregators.py` (`norm_clip`, `dp_fedavg`)
- Test: `tests/test_learning_privacy.py` (new), `tests/test_learning_aggregators.py`

**Interfaces:**
- Produces:
  - `gaussian_epsilon(noise_multiplier: float, rounds: int, delta: float) -> float`;
  - the aggregator `norm_clip(bound)`, which reports `("aggregation.clipped", n, {})`;
  - the aggregator `dp_fedavg(clip, sigma, delta)`, which reports `("privacy.epsilon", ε, {"mechanism": "central"})`.

- [ ] **Step 1: Write the failing tests**

`tests/test_learning_privacy.py`:

```python
from __future__ import annotations

import pytest

from onion_fl.learning.privacy import gaussian_epsilon


def test_epsilon_matches_the_closed_form_optimum() -> None:
    # σ=1, T=1, δ=1e-5: min_α α/2 + ln(1e5)/(α−1) at α = 1 + sqrt(2·ln 1e5)
    assert gaussian_epsilon(1.0, 1, 1e-5) == pytest.approx(5.298, abs=0.01)


def test_epsilon_grows_with_rounds_and_shrinks_with_noise() -> None:
    assert gaussian_epsilon(1.0, 10, 1e-5) > gaussian_epsilon(1.0, 1, 1e-5)
    assert gaussian_epsilon(2.0, 10, 1e-5) < gaussian_epsilon(1.0, 10, 1e-5)
    assert gaussian_epsilon(1.0, 0, 1e-5) == 0.0
```

`tests/test_learning_aggregators.py`:

```python
def test_norm_clip_bounds_each_update_against_the_reference() -> None:
    reference = {"w": np.zeros(2)}
    children = [c("a", [3.0, 4.0]), c("b", [0.3, 0.4])]  # norms 5 and 0.5
    clip = aggregators.create("norm_clip", {"bound": 1.0})

    out = clip.aggregate(children, "fog", reference=reference)

    np.testing.assert_allclose(out.state["w"], [(0.6 + 0.3) / 2, (0.8 + 0.4) / 2])
    assert clip.report() == [("aggregation.clipped", 1.0, {})]


def test_dp_fedavg_adds_seeded_noise_and_reports_epsilon() -> None:
    reference = {"w": np.zeros(3)}
    children = [c("a", [1.0, 1.0, 1.0]), c("b", [1.0, 1.0, 1.0])]
    dp = aggregators.create("dp_fedavg", {"clip": 10.0, "sigma": 1.0, "delta": 1e-5})

    first = dp.aggregate(children, "fog", reference=reference, rng=np.random.default_rng(0))
    again = aggregators.create("dp_fedavg", {"clip": 10.0, "sigma": 1.0, "delta": 1e-5})
    second = again.aggregate(children, "fog", reference=reference, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(first.state["w"], second.state["w"])
    assert not np.allclose(first.state["w"], [1.0, 1.0, 1.0])  # noise σ·C/m = 5
    (name, value, tags), = dp.report()
    assert name == "privacy.epsilon" and tags == {"mechanism": "central"}
    assert value == pytest.approx(5.298, abs=0.01)  # one round at σ=1


def test_dp_fedavg_leaves_auxiliary_arrays_unclipped_and_unnoised() -> None:
    reference = {"w": np.zeros(1), "scaffold/w": np.zeros(1)}
    children = [
        Contribution("a", {"w": np.ones(1), "scaffold/w": np.full(1, 9.0)}, {"w": 1.0, "scaffold/w": 1.0}),
        Contribution("b", {"w": np.ones(1), "scaffold/w": np.full(1, 9.0)}, {"w": 1.0, "scaffold/w": 1.0}),
    ]

    out = aggregators.create("dp_fedavg", {"clip": 0.1, "sigma": 1.0}).aggregate(
        children, "fog", reference=reference, rng=np.random.default_rng(0)
    )

    np.testing.assert_allclose(out.state["scaffold/w"], [9.0])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_learning_privacy.py tests/test_learning_aggregators.py -q -k "epsilon or norm_clip or dp_fedavg"`
Expected: FAIL. The module and the plugins don't exist yet.

- [ ] **Step 3: Implement**

`src/onion_fl/learning/privacy.py`:

```python
from __future__ import annotations

"""Differential privacy: the RDP accountant and the edge-side mechanism (spec §3.7)."""

import numpy as np

ORDERS = np.concatenate([np.linspace(1.01, 10.0, 400), np.arange(11.0, 513.0)])


def gaussian_epsilon(noise_multiplier: float, rounds: int, delta: float) -> float:
    """ε after ``rounds`` Gaussian mechanisms of this noise multiplier.

    RDP of order α is α/(2σ²) per round and adds up; ε = min_α T·α/(2σ²) +
    ln(1/δ)/(α − 1). No amplification by subsampling: a conservative bound.
    """
    if rounds <= 0:
        return 0.0
    eps = rounds * ORDERS / (2 * noise_multiplier**2) + np.log(1 / delta) / (ORDERS - 1)
    return float(eps.min())
```

`aggregators.py`:

```python
def _clipped(
    item: Contribution, reference: Mapping[str, np.ndarray], bound: float
) -> tuple[Contribution, bool]:
    """``item`` with its model-key update scaled to norm ≤ ``bound``; aux keys untouched."""
    keys = [k for k in item.state if not is_aux(k) and k in reference]
    delta = {k: np.asarray(item.state[k], np.float64) - np.asarray(reference[k], np.float64) for k in keys}
    norm = float(np.sqrt(sum(float((d**2).sum()) for d in delta.values())))
    scale = min(1.0, bound / norm) if norm > 0 else 1.0
    state = dict(item.state)
    for key in keys:
        state[key] = (np.asarray(reference[key], np.float64) + scale * delta[key]).astype(
            np.asarray(item.state[key]).dtype
        )
    return Contribution(item.source, state, item.weights), scale < 1.0


class NormClipParams(BaseModel):
    bound: float = Field(1.0, gt=0, description="Norma máxima de la actualización de cada hijo")


@aggregators.register(
    "norm_clip",
    title="Recorte de norma",
    description="Recorta la actualización de cada hijo a una norma máxima y aplica FedAvg.",
    params=NormClipParams,
    explain="Acota la influencia de cualquier hijo, malicioso o no (Sun et al., 2019).",
)
class NormClip:
    def __init__(self, bound: float = 1.0) -> None:
        self.bound = bound
        self.clipped = 0

    def report(self) -> list[tuple[str, float, dict[str, Any]]]:
        return [("aggregation.clipped", float(self.clipped), {})]

    def aggregate(self, contributions, source, reference=None, rng=None) -> Contribution:
        reference = reference or {}
        pairs = [_clipped(c, reference, self.bound) for c in contributions]
        self.clipped = sum(1 for _, was in pairs if was)
        return _combine_per_key([c for c, _ in pairs], source, _weighted_mean, needs_weights=True)


class DPFedAvgParams(BaseModel):
    clip: float = Field(1.0, gt=0, description="Norma máxima C de cada actualización")
    sigma: float = Field(1.0, gt=0, description="Multiplicador de ruido σ")
    delta: float = Field(1e-5, gt=0, lt=1, description="δ del presupuesto (ε, δ)")


@aggregators.register(
    "dp_fedavg",
    title="DP-FedAvg (central)",
    description="Recorta cada actualización a C, promedia y suma ruido N(0, (σ·C/m)²).",
    params=DPFedAvgParams,
    explain=(
        "Privacidad diferencial a nivel de hijo en el agregador, que es de confianza; "
        "informa ε por ronda con un contable RDP sin amplificación por submuestreo "
        "(McMahan et al., 2018). Las claves auxiliares no se recortan ni se ruidean."
    ),
)
class DPFedAvg:
    def __init__(self, clip: float = 1.0, sigma: float = 1.0, delta: float = 1e-5) -> None:
        self.clip, self.sigma, self.delta = clip, sigma, delta
        self.rounds = 0

    def report(self) -> list[tuple[str, float, dict[str, Any]]]:
        epsilon = gaussian_epsilon(self.sigma, self.rounds, self.delta)
        return [("privacy.epsilon", epsilon, {"mechanism": "central"})]

    def aggregate(self, contributions, source, reference=None, rng=None) -> Contribution:
        reference = reference or {}
        rng = rng if rng is not None else np.random.default_rng(0)
        clipped = [_clipped(c, reference, self.clip)[0] for c in contributions]
        uniform = [
            Contribution(c.source, c.state, dict.fromkeys(c.state, 1.0)) for c in clipped
        ]
        mean = _combine_per_key(uniform, source, _weighted_mean, needs_weights=True)
        totals = _combine_per_key(clipped, source, _weighted_mean, needs_weights=True)
        state = dict(totals.state)  # aux keys: FedAvg, untouched
        for key, value in mean.state.items():
            if is_aux(key) or key not in reference:
                continue
            holders = sum(1 for c in clipped if key in c.state)
            noise = rng.normal(0.0, self.sigma * self.clip / holders, size=np.shape(value))
            state[key] = (np.asarray(value, np.float64) + noise).astype(np.asarray(value).dtype)
        self.rounds += 1
        return Contribution(source, state, totals.weights)
```

(import `gaussian_epsilon` from `onion_fl.learning.privacy`.)

- [ ] **Step 4: Run the suites** (`tests/test_learning_privacy.py`, `tests/test_learning_aggregators.py`; update the registry list test).

- [ ] **Step 5: Commit** `feat(learning): Add norm clipping, central DP and the epsilon accountant`

---

### Task 4: The attack axis

**Files:**
- Create: `src/onion_fl/learning/attacks.py` (`attacks` registry: `label_flip`, `sign_flip`, `gaussian`, `scale`)
- Modify: `src/onion_fl/experiment/config.py` (`ExperimentConfig.attack`, `REGISTRIES`), `src/onion_fl/experiment/runner.py` (`malicious_edges`, `edge_specs`, `record_data`), `src/onion_fl/roles/federation.py` (`EdgeSpec.attack`), `src/onion_fl/roles/nodes.py` (`Edge` hooks)
- Modify: `docs/RULES.md` (label flipping simulates an attacker)
- Test: `tests/test_learning_attacks.py` (new), `tests/test_roles_protocol.py`, `tests/test_experiment.py`

**Interfaces:**
- Produces:
  - `Attack` objects with `fraction`, `start_round`, `on_data(data) -> data` and `on_update(arrays, received, rng) -> arrays`;
  - `ExperimentConfig.attack: PluginRef | None`, out of the dump when unset;
  - `malicious_edges(config, clients, seed) -> set[str]`, which picks `round(fraction · n)` edges per dataset with `node_rng(seed, f"attack/{dataset}")`;
  - malicious edges carry `tags["malicious"] = True`, and the run records `data.attack`.

- [ ] **Step 1: Write the failing tests**

`tests/test_learning_attacks.py` (hand-written arrays and a tiny label array; nothing is trained):

```python
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from onion_fl.learning.attacks import attacks

RECEIVED = {"w": np.zeros(2), "head.t.weight": np.zeros(1)}
TRAINED = {"w": np.array([1.0, 2.0]), "head.t.weight": np.ones(1), "scaffold/w": np.full(2, 5.0)}


def test_sign_flip_reverses_the_update_of_crossing_keys_only() -> None:
    out = attacks.create("sign_flip", {"scale": 1.0}).on_update(
        TRAINED, {"w": RECEIVED["w"]}, np.random.default_rng(0)
    )

    np.testing.assert_allclose(out["w"], [-1.0, -2.0])
    np.testing.assert_allclose(out["head.t.weight"], [1.0])  # not received: local
    np.testing.assert_allclose(out["scaffold/w"], [5.0, 5.0])  # auxiliary: untouched


def test_scale_boosts_the_update() -> None:
    out = attacks.create("scale", {"factor": 10.0}).on_update(TRAINED, RECEIVED, np.random.default_rng(0))

    np.testing.assert_allclose(out["w"], [10.0, 20.0])


def test_gaussian_replaces_the_update_with_seeded_noise() -> None:
    a = attacks.create("gaussian", {"sigma": 1.0}).on_update(TRAINED, RECEIVED, np.random.default_rng(3))
    b = attacks.create("gaussian", {"sigma": 1.0}).on_update(TRAINED, RECEIVED, np.random.default_rng(3))

    np.testing.assert_array_equal(a["w"], b["w"])
    assert not np.allclose(a["w"], TRAINED["w"])


def test_label_flip_mirrors_the_labels_and_keeps_the_original() -> None:
    data = SimpleNamespace(X=np.zeros((3, 1)), y=np.array([0, 1, 1]), n_classes=2)

    flipped = attacks.create("label_flip").on_data(data)

    assert flipped.y.tolist() == [1, 0, 0] and data.y.tolist() == [0, 1, 1]
```

`tests/test_roles_protocol.py`:

```python
def test_a_malicious_edge_attacks_from_its_start_round() -> None:
    attack = attacks.create("scale", {"factor": 3.0, "start_round": 2})
    edges = {"fog_0": [EdgeSpec("e1", model(A), trainer=trainers.create("stub", {"shift": 1.0}), attack=attack)]}

    federation = run(tree(1), edges, rounds=2)

    assert_global(federation, "trunk.0.weight", 1.0 + 3.0)  # round 1 honest, round 2 ×3
```

(import `attacks` from `onion_fl.learning.attacks`.)

`tests/test_experiment.py`:

```python
def test_malicious_edges_are_a_seeded_fraction_of_each_dataset(workspace: Path) -> None:
    from onion_fl.experiment.runner import _scenario_data, malicious_edges

    config = parse_experiment(experiment(workspace, attack={"name": "sign_flip", "fraction": 0.5}))
    (scenario,) = scenarios(config)
    _, split, _, _ = _scenario_data(scenario)

    chosen = malicious_edges(scenario.config, split.clients, scenario.seed)

    assert len(chosen) == round(0.5 * len(split.clients))
    assert chosen == malicious_edges(scenario.config, split.clients, scenario.seed)


def test_an_unset_attack_stays_out_of_the_config(workspace: Path) -> None:
    assert "attack" not in parse_experiment(experiment(workspace)).dump()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_learning_attacks.py tests/test_roles_protocol.py tests/test_experiment.py -q -k "attack or malicious or flip or scale or gaussian"`
Expected: FAIL. The attacks module is missing, `EdgeSpec` has no `attack`, and `attack` is an unknown config field.

- [ ] **Step 3: Implement**

`src/onion_fl/learning/attacks.py`:

```python
from __future__ import annotations

"""Malicious edges (spec §3.6): hooks on the training data and on the update.

Label flipping corrupts real labels to simulate an attacker; it is not
synthetic training data (docs/RULES.md).
"""

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, PositiveInt

from onion_fl.core.registry import Registry
from onion_fl.learning.model import is_aux

attacks = Registry("attack")
Arrays = Mapping[str, np.ndarray]


class AttackParams(BaseModel):
    fraction: float = Field(0.2, ge=0, le=1, description="Fracción de edges maliciosos de cada dataset")
    start_round: PositiveInt = Field(1, description="Primera ronda en la que atacan")


class Attack:
    def __init__(self, fraction: float = 0.2, start_round: int = 1) -> None:
        self.fraction, self.start_round = fraction, start_round

    def on_data(self, data: Any) -> Any:
        return data

    def on_update(self, arrays: Arrays, received: Arrays, rng: np.random.Generator) -> dict[str, np.ndarray]:
        return dict(arrays)

    @staticmethod
    def _crossing(arrays: Arrays, received: Arrays) -> list[str]:
        return [k for k in arrays if k in received and not is_aux(k)]


@attacks.register("label_flip", title="Inversión de etiquetas", description="Entrena con y → n_clases − 1 − y.", params=AttackParams,
                  explain="Simula a un atacante que envenena sus datos; las etiquetas son reales, no sintéticas.")
class LabelFlip(Attack):
    def on_data(self, data: Any) -> Any:
        flipped = copy.copy(data)
        classes = int(getattr(data, "n_classes", int(np.max(data.y)) + 1))
        object.__setattr__(flipped, "y", classes - 1 - np.asarray(data.y))
        return flipped


class SignFlipParams(AttackParams):
    scale: float = Field(1.0, gt=0, description="Cuánto se invierte la actualización")


@attacks.register("sign_flip", title="Inversión de signo", description="Envía x − s·(y − x): empuja en contra del aprendizaje.", params=SignFlipParams)
class SignFlip(Attack):
    def __init__(self, scale: float = 1.0, **params: Any) -> None:
        super().__init__(**params)
        self.scale = scale

    def on_update(self, arrays, received, rng):
        out = dict(arrays)
        for k in self._crossing(arrays, received):
            x = np.asarray(received[k], np.float64)
            out[k] = (x - self.scale * (np.asarray(arrays[k], np.float64) - x)).astype(np.asarray(arrays[k]).dtype)
        return out
```

(`ScaleParams(AttackParams).factor = 10.0` with `x + factor·(y − x)`; `GaussianParams(AttackParams).sigma = 1.0` with `x + rng.normal(0, σ, shape)`. Same structure: register both.)

`config.py`:
- register `"attack": attacks` in `REGISTRIES`;
- add the field to `ExperimentConfig`:

```python
    attack: PluginRef | None = Field(
        None,
        description="Ataque de una fracción de edges de cada dataset",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )

    _attack = field_validator("attack")(lambda v: v if v is None else _plugin(attacks, v))
```

`federation.py`: add `attack: Any = None` to `EdgeSpec` and pass `attack=spec.attack` to `Edge(...)`.

`nodes.py`, `Edge`:
- add the keyword `attack: Any = None` to `__init__`;
- in `_train`, after the `scoring`/`finetuning` setup:

```python
        attacking = self.attack is not None and msg.round >= self.attack.start_round
        data = self.attack.on_data(self.data) if attacking else self.data
```

- pass `data` to `self.trainer.train(...)`;
- just before `up = keys_crossing(...)`:

```python
        if attacking:
            arrays = self.attack.on_update(arrays, received, child_rng(ctx.rng))
```

`runner.py`:

```python
def malicious_edges(config: ExperimentConfig, clients: Sequence[Any], seed: int) -> set[str]:
    """The seeded ``fraction`` of each dataset's training edges that attack."""
    if config.attack is None:
        return set()
    fraction = create(attacks, config.attack).fraction
    chosen: set[str] = set()
    for dataset in sorted({c.dataset for c in clients}):
        ids = sorted(c.id for c in clients if c.dataset == dataset)
        rng = node_rng(seed, f"attack/{dataset}")
        n = round(fraction * len(ids))
        chosen |= set(rng.choice(ids, size=n, replace=False).tolist()) if n else set()
    return chosen
```

In `edge_specs`:
- compute `bad = malicious_edges(config, split.clients, scenario.seed)`;
- give each training `EdgeSpec` `attack=create(attacks, config.attack) if client.id in bad else None` and `tags={"dataset": client.dataset} | ({"malicious": True} if client.id in bad else {})`.

In `run_scenario`, after `record_data`:

```python
        bad = malicious_edges(config, split.clients, scenario.seed)
        if bad:
            run.record("data.attack", float(len(bad)), edges=sorted(bad))
```

`docs/RULES.md`, in "Sin datos sintéticos": add a line saying that the `label_flip` attack corrupts real labels to simulate an attacker. It is an attack model, not synthetic training data.

- [ ] **Step 4: Run the suites** (attacks, protocol, experiment, evaluation, studio).

- [ ] **Step 5: Commit** `feat(learning): Add the attack axis: label, sign, Gaussian and scaling attacks`

---

### Task 5: Local differential privacy at the edge

**Files:**
- Modify: `src/onion_fl/learning/privacy.py` (`privacies` registry, `local_dp`)
- Modify: `src/onion_fl/experiment/config.py` (`ExperimentConfig.privacy`, `REGISTRIES`), `src/onion_fl/experiment/runner.py` (`edge_specs`), `src/onion_fl/roles/federation.py`, `src/onion_fl/roles/nodes.py`
- Test: `tests/test_learning_privacy.py`, `tests/test_roles_protocol.py`, `tests/test_experiment.py`

**Interfaces:**
- Produces:
  - the `privacies` registry with `local_dp(clip, sigma, delta)` and its methods `on_update(arrays, received, rng)` and `epsilon(rounds)`;
  - `ExperimentConfig.privacy: PluginRef | None`;
  - an edge with privacy emits `privacy.epsilon` (mechanism `local`) after every update it sends.

- [ ] **Step 1: Write the failing tests**

```python
def test_local_dp_clips_and_noises_crossing_keys_only() -> None:
    from onion_fl.learning.privacy import privacies

    dp = privacies.create("local_dp", {"clip": 1.0, "sigma": 0.0001})
    out = dp.on_update(
        {"w": np.array([3.0, 4.0]), "scaffold/w": np.full(2, 5.0)},
        {"w": np.zeros(2)},
        np.random.default_rng(0),
    )

    np.testing.assert_allclose(out["w"], [0.6, 0.8], atol=1e-3)
    np.testing.assert_allclose(out["scaffold/w"], [5.0, 5.0])
    assert dp.epsilon(3) == pytest.approx(gaussian_epsilon(0.0001, 3, 1e-5))
```

(`tests/test_learning_privacy.py`; import numpy.)

```python
def test_an_edge_with_local_dp_reports_its_epsilon_each_round() -> None:
    from onion_fl.learning.privacy import privacies

    spec = EdgeSpec("e1", model(A), trainer=trainers.create("stub"), privacy=privacies.create("local_dp", {"clip": 10.0, "sigma": 1.0}))

    federation = run(tree(1), {"fog_0": [spec]}, rounds=2)

    values = [e["value"] for e in names(federation, "privacy.epsilon", "e1")]
    assert len(values) == 2 and values[0] < values[1]
```

(`tests/test_roles_protocol.py`)

- [ ] **Step 2: Run them to verify they fail.** Expected: unknown `privacies` / `EdgeSpec.privacy`.

- [ ] **Step 3: Implement**

`privacy.py`:

```python
from typing import Any
from collections.abc import Mapping

from pydantic import BaseModel, Field

from onion_fl.core.registry import Registry
from onion_fl.learning.model import is_aux

privacies = Registry("privacy")


class LocalDPParams(BaseModel):
    clip: float = Field(1.0, gt=0, description="Norma máxima C de la actualización del edge")
    sigma: float = Field(1.0, gt=0, description="Multiplicador de ruido σ")
    delta: float = Field(1e-5, gt=0, lt=1, description="δ del presupuesto (ε, δ)")


@privacies.register(
    "local_dp",
    title="DP local",
    description="Cada edge recorta su actualización a C y le suma N(0, (σ·C)²) antes de enviarla.",
    params=LocalDPParams,
    explain="No confía en el agregador; informa ε por ronda (contable RDP, sin amplificación).",
)
class LocalDP:
    def __init__(self, clip: float = 1.0, sigma: float = 1.0, delta: float = 1e-5) -> None:
        self.clip, self.sigma, self.delta = clip, sigma, delta

    def epsilon(self, rounds: int) -> float:
        return gaussian_epsilon(self.sigma, rounds, self.delta)

    def on_update(self, arrays: Mapping[str, Any], received: Mapping[str, Any], rng) -> dict[str, Any]:
        keys = [k for k in arrays if k in received and not is_aux(k)]
        delta = {k: np.asarray(arrays[k], np.float64) - np.asarray(received[k], np.float64) for k in keys}
        norm = float(np.sqrt(sum(float((d**2).sum()) for d in delta.values())))
        scale = min(1.0, self.clip / norm) if norm > 0 else 1.0
        out = dict(arrays)
        for k in keys:
            noisy = scale * delta[k] + rng.normal(0.0, self.sigma * self.clip, size=delta[k].shape)
            out[k] = (np.asarray(received[k], np.float64) + noisy).astype(np.asarray(arrays[k]).dtype)
        return out
```

Wiring follows Task 4's pattern:
- `ExperimentConfig.privacy`, with `exclude_if` None and `"privacy": privacies` in `REGISTRIES`;
- `EdgeSpec.privacy`, passed to `Edge`;
- `edge_specs` gives every training edge its own `create(privacies, config.privacy)`.

`Edge`:
- `privacy: Any = None` and `self._released = 0`;
- after the attack hook:

```python
        if self.privacy is not None:
            arrays = self.privacy.on_update(arrays, received, child_rng(ctx.rng))
            self._released += 1
            ctx.emit(
                "privacy.epsilon",
                self.privacy.epsilon(self._released),
                round=msg.round,
                mechanism="local",
            )
```

- [ ] **Step 4: Run the suites.** Expected: all pass.

- [ ] **Step 5: Commit** `feat(learning): Add local differential privacy at the edges`

---

### Task 6: Robustness and privacy comparison, results and docs

**Files:**
- Create: `experiments/techniques_robustness.yaml`, `experiments/techniques_privacy.yaml`
- Create: `results/techniques_robustness/`, `results/techniques_privacy/` (run copies, CSVs, `report.html`, `INDEX.md`)
- Modify: `README.md`, `docs/architecture.md`

- [ ] **Step 1: Pick the clipping bound on training statistics.**
  - Run FedAvg, seed 0, no attack. Record every child's update norm against `reference` at the fogs: wrap `FedAvg.aggregate` in a scratchpad script, as in the SCAFFOLD debugging.
  - Set `C` to the median norm, rounded.
  - Use it as `norm_clip.bound` and as the DP `clip`, and ledger it as a ruling. It comes from training updates, not test scores.

- [ ] **Step 2: Write the experiments.** Both use the drift experiment's setting: SGD, lr 0.3, chosen on validation for FedAvg. Placement α = 0.5, 20 rounds, seeds 0–2, global evaluation only.
  - `techniques_robustness.yaml`: `sweep: {learning.aggregator: [fedavg, median, trimmed_mean, krum, multi_krum, geometric_median, bulyan, {name: norm_clip, bound: C}], attack: [null, {name: sign_flip, fraction: 0.2}]}`. That is 16 scenarios.
  - `techniques_privacy.yaml`: `sweep: {learning.aggregator,privacy: [[fedavg, null], [{name: dp_fedavg, clip: C, sigma: 0.5}, null], [{name: dp_fedavg, clip: C, sigma: 1.0}, null], [fedavg, {name: local_dp, clip: C, sigma: 0.5}], [fedavg, {name: local_dp, clip: C, sigma: 1.0}]]}`. That is 5 scenarios.
  - Plan both, then commit.

- [ ] **Step 3: Run both on a clean tree, in the background,** one after the other.

- [ ] **Step 4: Results.** For each experiment:
  - copy the runs (no `host`/`pid`);
  - write `run_metrics.csv` with the global macro-F1 per dataset and `quorum_failed`, plus, for robustness, the selection detection (`aggregation.dropped` events at the fogs: malicious dropped / malicious present) and, for privacy, the final ε per mechanism;
  - generate `report.html`;
  - write `INDEX.md` saying what the numbers show.

- [ ] **Step 5: Docs.**
  - README: §1 Learning row, test count, §4 `learning.aggregator`/`attack`/`privacy`, and a §7 subsection per experiment.
  - `docs/architecture.md`: the aggregation paragraph, the plugin table (`attack`, `privacy`) and the aggregator signature.
  - Run the full check, then commit.

- [ ] **Step 6: Final review (fresh reviewer), fix pass, PR targeting `task/#148` and left open.**
