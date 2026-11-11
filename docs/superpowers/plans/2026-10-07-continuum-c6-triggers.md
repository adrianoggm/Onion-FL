# Continuum C6: triggers and continuous federation, implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Issue #161. In a stream run, triggers open the rounds instead of a fixed `round_every`, and edges train only when their local trigger fires. Drift detection runs per edge, zone and federation; it is reported as events and can be a trigger.

**Architecture:**
- **Two new modules**, both pure and unit-tested:
  - `continuum/drift.py`: per-window statistics for three kinds of drift, plus a Page-Hinkley detector registry;
  - `continuum/triggers.py`: the trigger plugins `schedule`, `volume`, `drift` and `any`.
- **Status messages.** Each streaming edge sends one every `continuum.status_every` of data time. It holds the rows that became trainable since the last one and the drift statistics, never data.
  - Each collector pools its children's statuses on its own tick and runs its own detectors.
  - A fog passes the pooled status up.
  - The coordinator adds it to what it has heard since the last round, and opens a round when its trigger fires.
- **Local trigger.** An edge whose local trigger has not fired answers a round idle (C3's idle path).
- **End of run.** The run ends with one final round at the first tick after the last label arrives.

**Tech Stack:** Python 3.11, numpy, pydantic v2, the repo's plugin `Registry`, `SimRuntime`.

**Spec:** `docs/superpowers/specs/2026-10-04-onion-fl-continuum-design.md`, §4.4 (kinds of drift) and §10 (triggers and continuous federation). §7's C3 decisions define the stream this builds on.

## Global Constraints

- Data time for every trigger duration (§10, "Duraciones en tiempo de datos"); `stream.speed` only converts to virtual time (§4.2).
- Status messages carry counts and statistics, never data (§10).
- An edge joins a round only if its local trigger fired (§10).
- Rounds stay synchronous; FedAsync/FedBuff are later consumption policies (§10, §3 "Asincronía").
- Everything runs in `SimRuntime`; streams stay refused in real mode, so `RealRuntime` equivalence is unaffected (§3 "Simulación", issue #161).
- No synthetic data for ML: protocol tests use the stub trainer and hand-written rows (docs/RULES.md).
- Plugin titles and descriptions in Spanish; code, comments and docs in English (CLAUDE.md).
- A run without `continuum` behaves exactly as in C3/C4, and its `config_id` does not change.

## Rulings (decisions this plan makes where the spec is silent)

1. **Which kinds of drift.** C6 detects three kinds: `data` (P(X)), `prior` (P(Y)) and `performance` (prequential error). `concept` and `client` are refused for now.
   - `concept` needs the prior drift removed from the error, so it waits for C8's analysis.
   - `client` is already covered by `diagnostic.divergence_*`.
   - Cost if wrong: one more kind later; the `Literal` grows.
2. **The statistics.**
   - `data`: the mean absolute shift of the window's features from the edge's history, in history standard deviations.
   - `prior`: the total variation between the window's labelled classes and the history's.
   - `performance`: the error rate of the stored predictions whose labels arrived in the window.
   - All three rise with drift, so one rise detector serves them all.
3. **Detector.** `page_hinkley`, for a sustained rise, with defaults (`delta` 0.005, `threshold` 0.5, `min_samples` 3) set for statistics in [0, 1]. It resets after each detection.
4. **Where detection runs.**
   - Each node keeps one detector per kind named by the triggers.
   - Edges feed theirs their window statistic.
   - A collector feeds its own with the n-weighted mean of what its children reported since its last tick: zone drift at a fog, federation drift at the cloud.
   - Detectors run only for the kinds that some drift trigger names, and every detection is a `drift.detected` event. The events are the diagnostic.
5. **What a status message carries.**
   - `fresh`: rows that became trainable since the sender's last status.
   - For each kind: `drift.<kind>.sum`, `.n` and `.detected` (detections at the sender or below).
   - `at`: the sender's data time.
   - Counts are deltas, so the coordinator never counts stale volume: it adds up what it heard since the round opened, and starts over when a round opens.
6. **Ticks.** Every node with a continuum ticks every `status_every`, starting at its start. A collector forwards only on its own tick, so a status reaches the cloud at most two `status_every` after the edge saw the rows. Triggers are checked at each coordinator tick and when a round closes.
7. **Round 1** opens as soon as registration ends (the bootstrap round, v0), with `trigger.fired` `start`.
8. **End of run.** At the first coordinator tick at or after the last label (`until`), one final round opens (`trigger.fired` `horizon`), and the run finishes when it closes. `rounds` stays an upper bound. Operator stop is C7.
9. **`stream.round_every`** becomes optional:
   - a stream needs either `round_every` or `continuum.trigger`, never both;
   - a schedule trigger with `every` equal to the old `round_every` and `status_every` equal to it reproduces C3's rounds, which a test checks bit for bit;
   - configs that give `round_every` keep their `config_id`.
10. **Predictions at a tick.** At each tick the edge predicts its new arrivals with the model it serves. `_arrivals` at round time predicts only rows not predicted yet. The served model changes only at a round, so this is the same model C3 used.
11. **Local trigger view.**
    - `volume`: the edge's unconsumed trainable rows;
    - `since`: the data time of its last successful training (its stream's start before that);
    - `drift`: the edge's own detections since then.
12. **Events.**
    - `trigger.fired` at the coordinator: value is the volume heard; tags `trigger`, `at`, `round`, `drift`.
    - `trigger.fired` at an edge whose local trigger fired: value is its volume.
    - `drift.detected` at any node: value is the detector statistic; tags `kind`, `value`, `at`.
    - In continuum mode `round.started` also carries `at`. These `at` tags are the data-time cursor the acceptance asks for.

## Review Focus

1. **A trigger that never fires** (volume above everything that arrives): the run must still end, with a final round at the last label. Test `test_a_trigger_that_never_fires_still_ends_at_the_last_label` (Task 4).
2. **Ticking nodes after the run ends** would keep the simulation alive until `max_events`. The finished coordinator and stopped nodes must stop ticking. Same test, which checks that `federation.run()` returns and nothing ticks after `run.finished` (Task 4).
3. **Windows with nothing for a detector** (no labels: `labels.fraction: 0`): the detectors must skip them silently. Test `test_a_drift_trigger_without_labels_never_fires` (Task 4).
4. **A trigger that fires while a round is open** must not open a second one, and the next round must open as soon as it closes. Test `test_a_trigger_during_an_open_round_waits_for_it` (Task 4).
5. **An edge whose local trigger never fires** must answer idle every round without blocking the fog's round, and its rows must stay unconsumed. Test `test_an_edge_trains_only_when_its_local_trigger_fires` (Task 3).

---

## File structure

| File | Responsibility |
|---|---|
| `src/onion_fl/continuum/pace.py` (new) | `View` (what a trigger sees) and `Continuum` (how a continuous federation keeps time). No imports beyond the standard library, so `roles` can import it without cycles |
| `src/onion_fl/continuum/drift.py` (new) | `KINDS`, the `detectors` registry with `page_hinkley`, `Reference` and `window()` |
| `src/onion_fl/continuum/triggers.py` (new) | The `triggers` registry (`schedule`, `volume`, `drift`, `any`) and `drift_detectors()` |
| `src/onion_fl/roles/nodes.py` | Edge ticks, statuses and local trigger; collector pooling and ticks; coordinator rounds opened by triggers and the final round |
| `src/onion_fl/roles/federation.py` | `build_federation(..., continuum=None)` hands the pace to every node |
| `src/onion_fl/data/stream.py` | `StreamConfig.round_every` optional |
| `src/onion_fl/experiment/config.py` | `ContinuumConfig`, the `continuum` field and its checks, and the `trigger` and `detector` registries in `REGISTRIES` |
| `src/onion_fl/experiment/runner.py` | `_paced` builds the `Continuum`; `build_scenario` passes it; the plan preview |
| `tests/test_continuum_triggers.py` (new) | Unit tests of triggers, detectors and window statistics |
| `tests/test_roles_stream.py` | Protocol tests of continuous federation with the stub trainer |
| `tests/test_experiment_stream.py` | Config and runner tests |
| `docs/architecture.md`, `README.md`, the spec's §10 | Documentation |
| `experiments/stream_triggers_swell_wesad.yaml`, `results/continuum_triggers/` | A real-data run of the triggers |

---

### Task 1: Drift statistics and the Page-Hinkley detector

**Files:**
- Create: `src/onion_fl/continuum/pace.py`, `src/onion_fl/continuum/drift.py`
- Test: `tests/test_continuum_triggers.py`

**Interfaces:**
- Produces:
  - `View(now: float, since: float, volume: float, drift: Mapping[str, int])`, a frozen dataclass;
  - `Continuum(trigger, edge_trigger, status_every: float, speed: float, until: float, drift: Mapping[str, Any])`, a frozen dataclass;
  - `KINDS = ("data", "prior", "performance")`;
  - `detectors: Registry`, with `page_hinkley` → `PageHinkley.update(value: float) -> float | None` (the statistic when a rise is detected);
  - `Reference(stream)` with `.mean`, `.std`, `.prior`;
  - `window(kind, stream, arrived, labelled, logits, predicted, reference) -> tuple[float, int]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Triggers, drift detectors and window statistics (continuum C6, issue #161).

The rows are hand-written; nothing is trained (docs/RULES.md).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from onion_fl.continuum.drift import Reference, detectors, window


def ph(**params):
    return detectors.create("page_hinkley", params)


def test_page_hinkley_ignores_a_flat_series() -> None:
    detector = ph()
    assert all(detector.update(0.2) is None for _ in range(50))


def test_page_hinkley_detects_a_sustained_rise_then_starts_over() -> None:
    detector = ph(threshold=0.5)
    found = [detector.update(x) for x in [0.1] * 10 + [0.9] * 10]

    first = next(i for i, f in enumerate(found) if f is not None)
    assert first >= 10 and found[first] > 0.5  # after the rise, with its statistic
    assert detector.n < 10  # it started over


def test_page_hinkley_waits_for_its_minimum_windows() -> None:
    detector = ph(threshold=0.1, min_samples=5)
    assert [detector.update(x) for x in (0.0, 1.0, 1.0, 1.0)] == [None] * 4


def stream(X, y, history, label_at=None):
    n = len(y)
    data = SimpleNamespace(X=np.asarray(X, float), y=np.asarray(y), n_classes=2)
    label_at = np.zeros(n) if label_at is None else np.asarray(label_at, float)
    return SimpleNamespace(
        data=data,
        history=np.asarray(history, bool),
        available_at=np.zeros(n),
        label_at=label_at,
    )


def test_data_drift_is_the_feature_shift_in_history_deviations() -> None:
    s = stream([[0.0], [2.0], [5.0], [5.0]], [0, 1, 0, 1], [1, 1, 0, 0])
    arrived = np.array([0, 0, 1, 1], bool)
    reference = Reference(s)  # history mean 1, std 1

    value, n = window("data", s, arrived, ~arrived, None, None, reference)

    assert (value, n) == (pytest.approx(4.0), 2)


def test_prior_drift_is_the_class_shift_of_the_labels_that_arrived() -> None:
    s = stream([[0.0]] * 6, [0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0])
    labelled = np.array([0, 0, 0, 0, 1, 1], bool)

    value, n = window("prior", s, labelled, labelled, None, None, Reference(s))

    assert (value, n) == (pytest.approx(0.5), 2)  # half and half, then all 1


def test_performance_drift_is_the_error_of_the_stored_predictions() -> None:
    s = stream([[0.0]] * 4, [0, 1, 1, 1], [0, 0, 0, 0])
    logits = np.array([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [np.nan, np.nan]])
    predicted = np.array([1, 1, 1, 0], bool)
    labelled = np.ones(4, bool)

    value, n = window("performance", s, labelled, labelled, logits, predicted, None)

    assert (value, n) == (pytest.approx(1 / 3), 3)  # the unpredicted row is left out


def test_a_window_with_nothing_for_its_kind_rests_on_no_rows() -> None:
    s = stream([[0.0]] * 2, [0, 1], [1, 1])
    nothing = np.zeros(2, bool)

    for kind in ("data", "prior", "performance"):
        assert window(kind, s, nothing, nothing, None, nothing, Reference(s))[1] == 0
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_continuum_triggers.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'onion_fl.continuum.drift'`.

- [ ] **Step 3: Write `pace.py` and `drift.py`**

`src/onion_fl/continuum/pace.py`:

```python
from __future__ import annotations

"""How a continuous federation keeps time (spec §10).

Kept free of imports from the rest of the package, so the roles can use it.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class View:
    """What a trigger sees at its level: data time now and at the last round (or
    update), the rows that became trainable since, and the drift detected since,
    by kind."""

    now: float
    since: float
    volume: float
    drift: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Continuum:
    """A continuous federation's pace, in virtual seconds (``speed`` gives data
    seconds per virtual second)."""

    trigger: Any  # federated: when the coordinator opens a round
    edge_trigger: Any | None  # local: whether an edge has an update for a round
    status_every: float  # virtual seconds between status messages
    speed: float
    until: float  # virtual time of the last label: the final round
    drift: Mapping[str, Any]  # kind -> detector spec, for every node
```

`src/onion_fl/continuum/drift.py`:

```python
from __future__ import annotations

"""Drift on a stream (spec §4.4): a statistic per window and a detector.

Each node turns what reached it since its last status into one number per kind,
and a detector watches that number for a sustained rise:

- ``data``, P(X): the mean absolute shift of the window's features from the
  edge's history, in history standard deviations;
- ``prior``, P(Y): the total variation between the classes of the labels that
  arrived and those of the history;
- ``performance``: the error rate of the stored predictions whose labels arrived.

Concept drift (the error net of a prior shift) and client drift are not
detected here; ``diagnostic.divergence_*`` already covers the client.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, PositiveInt

from onion_fl.core.registry import Registry

KINDS = ("data", "prior", "performance")
detectors = Registry("detector")


class PageHinkleyParams(BaseModel):
    delta: float = Field(0.005, ge=0, description="Subida tolerada sobre la media")
    threshold: float = Field(
        0.5, gt=0, description="Subida acumulada que cuenta como deriva"
    )
    min_samples: PositiveInt = Field(3, description="Ventanas antes de poder detectar")


@detectors.register(
    "page_hinkley",
    title="Page-Hinkley",
    description="Detecta una subida sostenida del estadístico respecto a su media.",
    params=PageHinkleyParams,
    explain="Tras cada detección vuelve a empezar (Page, 1954; Hinkley, 1971).",
)
class PageHinkley:
    def __init__(
        self, delta: float = 0.005, threshold: float = 0.5, min_samples: int = 3
    ) -> None:
        self.delta, self.threshold, self.min_samples = delta, threshold, min_samples
        self.reset()

    def reset(self) -> None:
        self.n, self.mean, self.cum, self.low = 0, 0.0, 0.0, 0.0

    def update(self, value: float) -> float | None:
        """One window's value; the statistic when a rise is detected, else None."""
        self.n += 1
        self.mean += (value - self.mean) / self.n
        self.cum += value - self.mean - self.delta
        self.low = min(self.low, self.cum)
        statistic = self.cum - self.low
        if self.n >= self.min_samples and statistic > self.threshold:
            self.reset()
            return float(statistic)
        return None


class Reference:
    """What an edge's windows are compared with: its history, the rows before t₀.

    The prior counts only history rows whose label was there by t₀.
    """

    def __init__(self, stream: Any) -> None:
        d, history = stream.data, stream.history
        X = np.asarray(d.X, float)[history]
        width = np.asarray(d.X).shape[1]
        self.mean = X.mean(axis=0) if len(X) else np.zeros(width)
        std = X.std(axis=0) if len(X) > 1 else np.ones(width)
        self.std = np.where(std > 0, std, 1.0)
        known = history & (stream.label_at <= stream.available_at)
        counts = np.bincount(d.y[known], minlength=d.n_classes).astype(float)
        self.prior = (
            counts / counts.sum()
            if counts.sum()
            else np.full(d.n_classes, 1 / d.n_classes)
        )


def window(
    kind: str,
    stream: Any,
    arrived: np.ndarray,
    labelled: np.ndarray,
    logits: np.ndarray | None,
    predicted: np.ndarray | None,
    reference: Reference | None,
) -> tuple[float, int]:
    """The ``kind`` statistic over one window, and how many rows it rests on.

    ``arrived`` and ``labelled`` mark the rows that arrived and whose labels
    arrived in the window; history rows are never part of one.
    """
    d = stream.data
    if kind == "data":
        rows = arrived & ~stream.history
        if not rows.any():
            return 0.0, 0
        mean = np.asarray(d.X, float)[rows].mean(axis=0)
        return float(np.mean(np.abs(mean - reference.mean) / reference.std)), int(
            rows.sum()
        )
    rows = labelled & ~stream.history
    if kind == "performance":
        rows = rows & predicted
    n = int(rows.sum())
    if not n:
        return 0.0, 0
    if kind == "prior":
        p = np.bincount(d.y[rows], minlength=d.n_classes) / n
        return float(0.5 * np.abs(p - reference.prior).sum()), n
    wrong = logits[rows].argmax(axis=1) != d.y[rows]
    return float(wrong.mean()), n
```

- [ ] **Step 4: Run them and see them pass**

Run: `.venv/Scripts/python -m pytest tests/test_continuum_triggers.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/continuum/pace.py src/onion_fl/continuum/drift.py tests/test_continuum_triggers.py
git commit -m "feat(continuum): Measure drift per window and detect a sustained rise"
```

---

### Task 2: Trigger plugins and the config

**Files:**
- Create: `src/onion_fl/continuum/triggers.py`
- Modify: `src/onion_fl/data/stream.py` (`StreamConfig.round_every`), `src/onion_fl/experiment/config.py` (`ContinuumConfig`, `continuum`, `_stream_fits`, `REGISTRIES`)
- Test: `tests/test_continuum_triggers.py`, `tests/test_experiment_stream.py`

**Interfaces:**
- Consumes: `View` (Task 1), `KINDS`, `detectors`.
- Produces:
  - `triggers: Registry` with `schedule(every)`, `volume(samples)`, `drift(kind, detector="page_hinkley")` and `any(of)`;
  - every trigger's `.fired(view: View) -> str | None` (its name, `drift/<kind>` for drift);
  - `drift_detectors(*tracked) -> dict[str, Any]`;
  - `ExperimentConfig.continuum: ContinuumConfig | None`, with `trigger`, `edge_trigger` and `status_every` (data seconds).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_continuum_triggers.py`:

```python
from onion_fl.continuum.pace import View
from onion_fl.continuum.triggers import drift_detectors, triggers


def fired(spec, **view) -> str | None:
    name, params = (spec, {}) if isinstance(spec, str) else (spec["name"], spec)
    trigger = triggers.create(name, {k: v for k, v in params.items() if k != "name"})
    return trigger.fired(View(**({"now": 0.0, "since": 0.0, "volume": 0.0} | view)))


def test_a_schedule_fires_once_its_data_time_has_passed() -> None:
    every = {"name": "schedule", "every": "5m"}
    assert fired(every, now=299.0) is None
    assert fired(every, now=400.0, since=100.0) == "schedule"


def test_a_volume_trigger_fires_at_its_rows() -> None:
    assert fired({"name": "volume", "samples": 10}, volume=9) is None
    assert fired({"name": "volume", "samples": 10}, volume=10) == "volume"


def test_a_drift_trigger_fires_on_its_kind_only() -> None:
    prior = {"name": "drift", "kind": "prior"}
    assert fired(prior, drift={"data": 2}) is None
    assert fired(prior, drift={"prior": 1}) == "drift/prior"


def test_any_names_the_first_trigger_that_fired() -> None:
    both = {
        "name": "any",
        "of": [{"name": "schedule", "every": 600}, {"name": "volume", "samples": 5}],
    }
    assert fired(both, now=100.0, volume=7) == "volume"
    assert fired(both, now=700.0, volume=7) == "schedule"
    assert fired(both, now=100.0, volume=1) is None


def test_the_detectors_come_from_every_drift_trigger() -> None:
    nested = triggers.create(
        "any",
        {"of": [{"name": "drift", "kind": "prior"}, {"name": "volume", "samples": 1}]},
    )
    stricter = {"name": "page_hinkley", "threshold": 1.0}
    local = triggers.create("drift", {"kind": "data", "detector": stricter})

    found = drift_detectors(nested, local, None)

    assert found == {"prior": "page_hinkley", "data": stricter}


def test_one_kind_cannot_have_two_detectors() -> None:
    stricter = {"name": "page_hinkley", "threshold": 2.0}
    a = triggers.create("drift", {"kind": "prior"})
    b = triggers.create("drift", {"kind": "prior", "detector": stricter})
    with pytest.raises(ValueError, match="prior"):
        drift_detectors(a, b)
```

Append to `tests/test_experiment_stream.py`:

```python
# --- triggers and continuous federation (continuum C6) ----------------------------

SCHEDULE = {"trigger": {"name": "schedule", "every": 300}, "status_every": 300}
PACED = {k: v for k, v in STREAM.items() if k != "round_every"}


@pytest.mark.parametrize(
    "change, message",
    [
        ({"continuum": SCHEDULE}, "round_every"),  # both pace the rounds
        ({"stream": PACED}, "round_every"),  # neither does
        ({"stream": None, "labels": None, "continuum": SCHEDULE}, "continuum"),
        (
            {"stream": PACED, "continuum": {"trigger": {"name": "drift", "kind": "concept"}}},
            "kind",
        ),
        ({"stream": PACED, "continuum": {"trigger": "carrier_pigeon"}}, "carrier_pigeon"),
        (
            {
                "stream": PACED,
                "continuum": {
                    "trigger": {"name": "drift", "kind": "prior"},
                    "edge_trigger": {
                        "name": "drift",
                        "kind": "prior",
                        "detector": {"name": "page_hinkley", "threshold": 2},
                    },
                },
            },
            "prior",
        ),
    ],
)
def test_a_continuum_that_cannot_run_is_refused(
    workspace: Path, change: dict, message: str
) -> None:
    raw = {k: v for k, v in (experiment(workspace) | change).items() if v is not None}
    with pytest.raises(ConfigError, match=message):
        parse_experiment(raw)


def test_a_stream_paced_by_round_every_keeps_its_config_id(workspace: Path) -> None:
    dumped = parse_experiment(experiment(workspace)).dump()
    assert dumped["stream"]["round_every"] == 300 and "continuum" not in dumped
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_continuum_triggers.py tests/test_experiment_stream.py -q -k "trigger or detectors or continuum or keeps_its_config_id"`
Expected: FAIL with `ModuleNotFoundError: No module named 'onion_fl.continuum.triggers'`; the config tests fail with `DID NOT RAISE` or an `extra_forbidden` error about `continuum`.

- [ ] **Step 3: Write `triggers.py`**

```python
from __future__ import annotations

"""When a continuous federation trains (spec §10): triggers as plugins.

The federated trigger, at the coordinator, decides when a round opens; the
local one, at each edge, whether the edge has an update for it. Both get a
``View`` of their level and answer with their name when they fire. Every
duration is data time.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, PositiveInt, field_validator

from onion_fl.continuum.drift import detectors
from onion_fl.continuum.pace import View
from onion_fl.core.registry import PluginError, Registry
from onion_fl.data.stream import Positive
from onion_fl.roles.policies import create

triggers = Registry("trigger")


def _known(registry: Registry, spec: Any) -> Any:
    try:
        create(registry, spec)
    except PluginError as exc:
        raise ValueError(str(exc)) from None
    return spec


class ScheduleParams(BaseModel):
    every: Positive = Field(description="Tiempo de datos desde la ronda anterior")


@triggers.register(
    "schedule",
    title="Calendario",
    description="Abre ronda cada cierto tiempo de datos desde la anterior.",
    params=ScheduleParams,
)
class Schedule:
    def __init__(self, every: float) -> None:
        self.every = every

    def fired(self, view: View) -> str | None:
        return "schedule" if view.now - view.since >= self.every - 1e-9 else None


class VolumeParams(BaseModel):
    samples: PositiveInt = Field(description="Filas entrenables nuevas que hacen falta")


@triggers.register(
    "volume",
    title="Volumen",
    description="Abre ronda cuando hay bastantes filas entrenables nuevas.",
    params=VolumeParams,
    explain="En el coordinador cuenta las de toda la federación desde la ronda "
    "anterior; en un edge, las suyas sin usar.",
)
class Volume:
    def __init__(self, samples: int) -> None:
        self.samples = samples

    def fired(self, view: View) -> str | None:
        return "volume" if view.volume >= self.samples else None


class DriftParams(BaseModel):
    kind: Literal["data", "prior", "performance"] = Field(
        description="Qué deriva: de datos P(X), de prior P(Y) o de rendimiento"
    )
    detector: str | dict[str, Any] = Field(
        "page_hinkley", description="Detector del estadístico de esa deriva"
    )

    _detector = field_validator("detector")(lambda v: _known(detectors, v))


@triggers.register(
    "drift",
    title="Deriva",
    description="Abre ronda cuando se detecta deriva de un tipo desde la anterior.",
    params=DriftParams,
    explain="Cada edge, cada zona y la federación vigilan su estadístico; "
    "basta una detección en cualquiera de ellos.",
)
class Drift:
    def __init__(self, kind: str, detector: str | dict[str, Any] = "page_hinkley"):
        self.kind, self.detector = kind, detector

    def fired(self, view: View) -> str | None:
        return f"drift/{self.kind}" if view.drift.get(self.kind, 0) > 0 else None


class AnyParams(BaseModel):
    of: list[str | dict[str, Any]] = Field(
        min_length=1, description="Disparadores; basta con que salte uno"
    )

    _of = field_validator("of")(lambda v: [_known(triggers, t) for t in v])


@triggers.register(
    "any",
    title="Cualquiera",
    description="Salta en cuanto salta uno de sus disparadores.",
    params=AnyParams,
)
class AnyOf:
    def __init__(self, of: list[str | dict[str, Any]]) -> None:
        self.of = [create(triggers, t) for t in of]

    def fired(self, view: View) -> str | None:
        return next((name for t in self.of if (name := t.fired(view))), None)


def drift_detectors(*tracked: Any) -> dict[str, Any]:
    """The detector each kind of drift needs, from every trigger given
    (``any`` included); one kind cannot have two different detectors."""
    found: dict[str, Any] = {}

    def walk(trigger: Any) -> None:
        if isinstance(trigger, AnyOf):
            for inner in trigger.of:
                walk(inner)
        elif isinstance(trigger, Drift):
            if trigger.kind in found and found[trigger.kind] != trigger.detector:
                raise ValueError(
                    f"drift {trigger.kind!r} has two detectors, {found[trigger.kind]} "
                    f"and {trigger.detector}; give both triggers the same"
                )
            found[trigger.kind] = trigger.detector

    for trigger in tracked:
        if trigger is not None:
            walk(trigger)
    return found
```

- [ ] **Step 4: Make `round_every` optional and add the config**

In `src/onion_fl/data/stream.py`:

```python
    round_every: Positive | None = Field(
        None,
        description="Tiempo de datos entre rondas; con continuum, lo decide su disparador",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )
```

In `src/onion_fl/experiment/config.py`, import `from onion_fl.continuum.drift import detectors`, `from onion_fl.continuum.triggers import drift_detectors, triggers` and `Positive` from `onion_fl.data.stream`. Add `"trigger": triggers, "detector": detectors` to `REGISTRIES`, then:

```python
class ContinuumConfig(Strict):
    trigger: PluginRef = Field(
        description="Disparador federativo: cuándo abre ronda el coordinador"
    )
    edge_trigger: PluginRef | None = Field(
        None,
        description="Disparador local: cuándo un edge tiene una actualización; sin "
        "él, en cada ronda en que tenga filas nuevas",
        exclude_if=lambda v: v is None,
    )
    status_every: Positive = Field(
        300.0, description="Tiempo de datos entre los mensajes de estado"
    )

    _trigger = field_validator("trigger")(lambda v: _plugin(triggers, v))
    _edge_trigger = field_validator("edge_trigger")(
        lambda v: v if v is None else _plugin(triggers, v)
    )

    @model_validator(mode="after")
    def _one_detector_per_kind(self) -> ContinuumConfig:
        local = None if self.edge_trigger is None else create(triggers, self.edge_trigger)
        try:
            drift_detectors(create(triggers, self.trigger), local)
        except ValueError as exc:
            raise ValueError(f"continuum: {exc}") from None
        return self
```

Add the field to `ExperimentConfig`, after `continual`:

```python
    continuum: ContinuumConfig | None = Field(
        None,
        description="Federación continua: disparadores en vez de rondas a ritmo fijo",
        exclude_if=lambda v: v is None,  # unset, it keeps existing config_ids
    )
```

In `_stream_fits`, after the `continual` check and before `if self.stream is None: return self`:

```python
        if self.continuum is not None and self.stream is None:
            raise ValueError("continuum: triggers pace a stream; set stream too")
```

and after it, before the refusals:

```python
        paced = self.stream.round_every is not None
        if paced == (self.continuum is not None):
            raise ValueError(
                "stream.round_every: a continuum trigger paces the rounds instead"
                if paced
                else "stream.round_every: set it, or let a continuum trigger pace "
                "the rounds"
            )
```

- [ ] **Step 5: Run them and see them pass**

Run: `.venv/Scripts/python -m pytest tests/test_continuum_triggers.py tests/test_experiment_stream.py tests/test_experiment.py -q`
Expected: all pass. The schema test still passes, now with the `trigger` and `detector` catalogues.

- [ ] **Step 6: Commit**

```bash
git add src/onion_fl/continuum/triggers.py src/onion_fl/data/stream.py src/onion_fl/experiment/config.py tests/test_continuum_triggers.py tests/test_experiment_stream.py
git commit -m "feat(continuum): Add trigger plugins and the continuum config"
```

---

### Task 3: Edge statuses and the local trigger

**Files:**
- Modify: `src/onion_fl/roles/nodes.py` (`Edge`), `src/onion_fl/roles/federation.py`
- Test: `tests/test_roles_stream.py`

**Interfaces:**
- Consumes: `Continuum`, `View` (Task 1), `Reference`, `window`, `detectors`, triggers' `.fired(view)` (Task 2).
- Produces:
  - `Edge(..., continuum=None)`;
  - `build_federation(..., continuum=None)`, which gives the pace to every training edge with a stream;
  - status messages `kind="status"`, `round=None`, with `payload.metrics` keys `at`, `fresh` and `drift.<kind>.{sum,n,detected}`;
  - events `drift.detected` (tags `kind`, `value`, `at`) and `trigger.fired` at edges.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_roles_stream.py`:

```python
# --- statuses and the local trigger (continuum C6) --------------------------------


def continuous(
    trigger,
    edge_trigger=None,
    status_every: float = 60.0,
    rounds: int = 1000,
    fog=None,
    trainer=Recording,
    compute=None,
    before=lambda federation: None,
    **streams,
):
    from onion_fl.continuum.pace import Continuum
    from onion_fl.continuum.triggers import drift_detectors, triggers

    def built(spec):
        if spec is None:
            return None
        name, params = (spec, {}) if isinstance(spec, str) else (spec["name"], spec)
        return triggers.create(name, {k: v for k, v in params.items() if k != "name"})

    federated, local = built(trigger), built(edge_trigger)
    until = max(s.drain for s in streams.values()) / SPEED
    pace = Continuum(
        trigger=federated,
        edge_trigger=local,
        status_every=status_every / SPEED,
        speed=SPEED,
        until=until,
        drift=drift_detectors(federated, local),
    )
    trainers_by_edge = {name: trainer() for name in streams}
    specs = [
        EdgeSpec(
            name,
            ModularMLP(CONFIG, [A], seed=0),
            trainer=trainers_by_edge[name],
            stream=s,
            compute=compute,
        )
        for name, s in sorted(streams.items())
    ]
    fogs = min(2, len(specs))
    federation = build_federation(
        tree(fogs, fog=fog),
        {f"fog_{i}": specs[i::fogs] for i in range(fogs)},
        initial_state=INITIAL,
        rounds=rounds,
        continuum=pace,
    )
    before(federation)
    federation.run()
    return federation, trainers_by_edge


def test_an_edge_reports_its_stream_a_status_at_a_time() -> None:
    federation, _ = continuous({"name": "schedule", "every": 300}, a1=stream_of("a-1"))

    sent = events(federation, "message.sent", "a1")
    statuses = [e for e in sent if e["tags"]["kind"] == "status"]
    assert len(statuses) >= 15  # 20 rows at one a minute, a status a minute
```

That statuses carry no data is pinned in Task 4, where the fog's pooled status is checked key by key.

```python
def test_an_edge_trains_only_when_its_local_trigger_fires() -> None:
    federation, trainers_by_edge = continuous(
        {"name": "schedule", "every": 120},
        edge_trigger={"name": "volume", "samples": 5},
        a1=stream_of("a-1"),
    )

    calls = trainers_by_edge["a1"].calls
    fired = events(federation, "trigger.fired", "a1")
    assert len(fired) == len(calls) and all(e["value"] >= 5 for e in fired)
    idle = [
        e for e in events(federation, "message.sent", "a1")
        if e["tags"]["kind"] == "update"
    ]
    assert len(idle) > len(calls)  # some rounds answered idle, rows kept
    trained = np.concatenate([t for _, t in calls])
    assert len(trained) == len(set(trained))  # nothing lost or repeated


def labels_switch(subject: str, n: int = 40):
    """A stream whose label switches from 0 to 1 halfway: a prior shift."""
    t = np.arange(n) * 60.0
    data = SubjectData(
        X=np.stack([t / 600, np.cos(t)], axis=1).astype(np.float32),
        y=(np.arange(n) >= n // 2).astype(int),
        dataset="a",
        subject=subject,
        task="t",
        n_classes=2,
        feature_names=["f0", "f1"],
        t=t,
    )
    config = StreamConfig(bootstrap=600, batch_size=1, speed=SPEED)
    return edge_stream(data, config, LabelsConfig(fraction=1.0), seed=0)


def test_an_edge_detects_a_prior_shift_after_it_happens() -> None:
    federation, _ = continuous(
        {"name": "drift", "kind": "prior"}, a1=labels_switch("a-1")
    )

    found = events(federation, "drift.detected", "a1")
    assert found and all(e["tags"]["kind"] == "prior" for e in found)
    assert found[0]["tags"]["at"] > 20 * 60 - 600  # after the switch, in data time
```

`labels_switch` passes `StreamConfig(bootstrap=600, ...)` without `round_every`, which Task 2 made optional.

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_roles_stream.py -q -k "reports_its_stream or local_trigger or prior_shift"`
Expected: FAIL with `TypeError: build_federation() got an unexpected keyword argument 'continuum'`.

- [ ] **Step 3: Implement the edge**

In `src/onion_fl/roles/nodes.py`, import:

```python
from onion_fl.continuum.drift import Reference, detectors, window
from onion_fl.continuum.pace import View
```

Add `continuum: Any = None` to `Edge.__init__`'s keywords. After the stream block:

```python
        # A continuous federation (continuum C6): statuses, detectors and the
        # local trigger; only a training edge with a stream takes part.
        self.continuum = continuum if stream is not None and train else None
        self._ticked = -np.inf  # continuum time of the last status
        self._updated_at = float(stream.available_at.min()) if stream is not None else 0.0
        self._local_drift: dict[str, int] = {}
        drift = self.continuum.drift if self.continuum is not None else {}
        self._detectors = {k: create(detectors, spec) for k, spec in sorted(drift.items())}
        self._reference = Reference(stream) if self._detectors else None
```

Set `self._updated_at` after `self.stream` is assigned. In `on_start`, after `self._say_hello(meta, ctx)`:

```python
        if self.continuum is not None:
            ctx.set_timer(self.continuum.status_every, "status")
```

Replace `on_timer`:

```python
    def on_timer(self, name: str, ctx: Context) -> None:
        if name == "hello" and not self.acknowledged:
            self._send_hello(ctx)
        elif name == "status" and not self.stopped:
            self._status(ctx)
            ctx.set_timer(self.continuum.status_every, "status")
```

In `on_message`, in the stop branch, add `ctx.cancel_timer("status")`.

Move the prediction out of `_arrivals` into a helper. In `_arrivals`, replace

```python
        if self._served is not None and new.any():
            served = copy.deepcopy(self.model)
            load_arrays(served, self._served)
            self._logits[new] = predict(served, s.data.X[new])
            self._predicted |= new
```

with `self._predict(new)`, and add:

```python
    def _predict(self, rows: np.ndarray) -> None:
        """Predict the rows not predicted yet with the model served now."""
        todo = rows & ~self._predicted
        if self._served is None or not todo.any():
            return
        served = copy.deepcopy(self.model)
        load_arrays(served, self._served)
        self._logits[todo] = predict(served, self.stream.data.X[todo])
        self._predicted |= todo

    def _status(self, ctx: Context) -> None:
        """What reached the edge since its last status, sent up without data
        (spec §10). The rows that became trainable, and per kind of drift the
        window's statistic, which also feeds the edge's own detector. New rows
        are predicted now, by the model served, as the next round would."""
        s = self.stream
        lo, now = self._ticked, s.clock(ctx.now())
        self._ticked = now
        arrived = s.arrived(lo, now)
        self._predict(arrived & ~s.history)
        fresh = s.trainable(lo, now) & (self._consumed_by < 0)
        metrics = {"at": now, "fresh": float(fresh.sum())}
        labelled = s.labelled(lo, now)
        for kind, detector in self._detectors.items():
            value, n = window(
                kind, s, arrived, labelled, self._logits, self._predicted, self._reference
            )
            found = detector.update(value) if n else None
            if found is not None:
                self._local_drift[kind] = self._local_drift.get(kind, 0) + 1
                ctx.emit("drift.detected", found, kind=kind, value=value, at=now, **self.tags)
            metrics |= {
                f"drift.{kind}.sum": value * n,
                f"drift.{kind}.n": float(n),
                f"drift.{kind}.detected": float(found is not None),
            }
        ctx.send(
            Message(
                kind="status", src=self.id, dst=self.parent, payload=Payload(metrics=metrics)
            )
        )

    def _wants_update(self, round: int, ctx: Context) -> bool:
        """The local trigger (spec §10): whether this edge has an update for
        the round; without one, it always has."""
        trigger = None if self.continuum is None else self.continuum.edge_trigger
        if trigger is None:
            return True
        view = View(
            now=self._clock,
            since=self._updated_at,
            volume=float(self._pending.sum()),
            drift=dict(self._local_drift),
        )
        name = trigger.fired(view)
        if name is not None:
            ctx.emit(
                "trigger.fired", view.volume, trigger=name, at=view.now, round=round, **self.tags
            )
        return name is not None
```

In `_train`, change the idle check after `self._served = state_arrays(self.model)`:

```python
            if own is None or not self._wants_update(msg.round, ctx):
                _send_idle(self, msg.round, metrics, ctx)
                return
```

After a successful training, in the `if self.stream is not None:` block that sets `self._consumed_by`, add:

```python
            if self.continuum is not None:  # what the local trigger counts from
                self._updated_at, self._local_drift = self._clock, {}
```

In `src/onion_fl/roles/federation.py`, add `continuum: Any = None` to `build_federation` and pass `continuum=continuum` to `Edge(...)`. Task 4 also passes it to the collectors.

- [ ] **Step 4: Run them and see them pass**

Run: `.venv/Scripts/python -m pytest tests/test_roles_stream.py -q`
Expected: all pass. The federated trigger is not wired yet, so the coordinator still runs one round and finishes. `test_an_edge_trains_only_when_its_local_trigger_fires` needs Task 4's rounds. Mark it `xfail(strict=True, reason="needs the coordinator's triggers, Task 4")` until Task 4 removes the mark.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/roles/nodes.py src/onion_fl/roles/federation.py tests/test_roles_stream.py
git commit -m "feat(roles): Report each edge's stream and train only when its trigger fires"
```

---

### Task 4: Collectors pool statuses and the coordinator opens rounds on its trigger

**Files:**
- Modify: `src/onion_fl/roles/nodes.py` (`_Collector`, `Coordinator`, `Aggregator`), `src/onion_fl/roles/federation.py`
- Test: `tests/test_roles_stream.py`

**Interfaces:**
- Consumes: status messages (Task 3), `View`, `Continuum`, `detectors`.
- Produces:
  - `_Collector(..., continuum=None)`;
  - collectors tick every `status_every` (timer `status`) and emit `drift.detected` for zone and federation detections;
  - the coordinator emits `trigger.fired` (value: volume heard; tags `trigger`, `at`, `round`, `drift`) before each round, and `round.started` with `at`;
  - the final round's trigger is `horizon`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_roles_stream.py`:

```python
def opened(federation) -> list[tuple[str, float]]:
    return [
        (e["tags"]["trigger"], e["tags"]["at"])
        for e in events(federation, "trigger.fired", "cloud")
    ]


def test_a_schedule_opens_a_round_every_period_of_data_time() -> None:
    federation, _ = continuous({"name": "schedule", "every": 300}, status_every=60, a1=stream_of("a-1"))

    triggers = [name for name, _ in opened(federation)]
    times = [at for _, at in opened(federation)]
    assert triggers[0] == "start" and triggers[-1] == "horizon"
    assert set(triggers[1:-1]) == {"schedule"}
    assert all(b - a == pytest.approx(300.0) for a, b in zip(times[:-1], times[1:-1], strict=False))


def test_a_schedule_reproduces_the_paced_rounds() -> None:
    every = {"name": "schedule", "every": EVERY}
    paced, _ = streaming(rounds=6, a1=stream_of("a-1"), a2=stream_of("a-2"))
    federation, _ = continuous(every, status_every=EVERY, a1=stream_of("a-1"), a2=stream_of("a-2"))

    for key, value in paced.coordinator.state.items():
        np.testing.assert_array_equal(federation.coordinator.state[key], value)


def test_a_volume_trigger_waits_for_its_rows() -> None:
    federation, _ = continuous({"name": "volume", "samples": 6}, a1=stream_of("a-1"), a2=stream_of("a-2"))

    fired = events(federation, "trigger.fired", "cloud")
    volume = [e for e in fired if e["tags"]["trigger"] == "volume"]
    assert volume and all(e["value"] >= 6 for e in volume)


def test_a_drift_trigger_opens_a_round_when_the_prior_shifts() -> None:
    federation, _ = continuous({"name": "drift", "kind": "prior"}, a1=labels_switch("a-1"), a2=labels_switch("a-2"))

    names = [name for name, _ in opened(federation)]
    assert "drift/prior" in names
    zone = events(federation, "drift.detected", "fog_0") + events(federation, "drift.detected", "fog_1")
    assert zone  # the fogs detect it on their pooled statistic too


def test_statuses_carry_counts_and_statistics_only() -> None:
    received = []

    def spy(federation) -> None:
        cloud = federation.coordinator
        handle = cloud.on_message

        def on_message(msg, ctx) -> None:
            if msg.kind == "status":
                received.append(msg.payload)
            handle(msg, ctx)

        cloud.on_message = on_message

    continuous(
        {"name": "drift", "kind": "prior"},
        before=spy,
        a1=labels_switch("a-1"),
        a2=labels_switch("a-2"),
    )

    keys = {"at", "fresh", "drift.prior.sum", "drift.prior.n", "drift.prior.detected"}
    assert received
    assert all(not p.state and set(p.metrics) <= keys for p in received)


def test_a_drift_trigger_without_labels_never_fires() -> None:
    no_labels = edge_stream(rows("a-1"), StreamConfig(bootstrap=120, batch_size=1, speed=SPEED), LabelsConfig(fraction=0.0), seed=0)
    federation, _ = continuous({"name": "drift", "kind": "prior"}, a1=no_labels)

    assert not events(federation, "drift.detected")
    assert [name for name, _ in opened(federation)] == ["start", "horizon"]


def test_a_trigger_that_never_fires_still_ends_at_the_last_label() -> None:
    federation, _ = continuous({"name": "volume", "samples": 10_000}, a1=stream_of("a-1"))

    assert [name for name, _ in opened(federation)] == ["start", "horizon"]
    finished = events(federation, "run.finished")
    assert finished
    last = finished[0]["t"]
    ticks = [e for e in federation.runtime.events if e["t"] > last and e["name"] == "message.sent" and e["tags"]["kind"] == "status"]
    assert not ticks  # nothing keeps ticking after the end


def test_a_trigger_during_an_open_round_waits_for_it() -> None:
    from onion_fl.runtime.devices import compute_models

    slow = compute_models.create("samples_per_second", {"samples_per_second": 0.02})
    federation, _ = continuous(
        {"name": "schedule", "every": 60}, status_every=60, compute=slow, a1=stream_of("a-1")
    )

    started = [e["tags"]["round"] for e in events(federation, "round.started", "cloud")]
    closed = {e["tags"]["round"]: e["t"] for e in events(federation, "round.closed", "cloud")}
    assert started == sorted(set(started))  # never two open at once
    for e in events(federation, "round.started", "cloud")[1:]:
        before = closed[e["tags"]["round"] - 1]
        assert e["t"] >= before  # each opens once the previous has closed
```

Remove the `xfail` mark from `test_an_edge_trains_only_when_its_local_trigger_fires`.

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_roles_stream.py -q -k "schedule or volume or drift_trigger or never_fires or open_round or local_trigger"`
Expected: FAIL. Only round 1 opens and there are no `trigger.fired` events at the cloud.

- [ ] **Step 3: Implement the collectors and the coordinator**

`_Collector.__init__` gets `continuum: Any = None`, then:

```python
        self.continuum = continuum
        self._heard: dict[str, float] = {}  # children's statuses since the last tick
        drift = continuum.drift if continuum is not None else {}
        self._detectors = {k: create(detectors, spec) for k, spec in sorted(drift.items())}
```

`_Collector.on_start`, after the register timer:

```python
        if self.continuum is not None:
            ctx.set_timer(self.continuum.status_every, "status")
```

In `_Collector.on_message`, before the `else`:

```python
        elif msg.src in self.children and msg.kind == "status":
            for key, value in msg.payload.metrics.items():
                if key != "at":
                    self._heard[key] = self._heard.get(key, 0.0) + float(value)
```

In `_Collector.on_timer`, add `elif name == "status": self._tick(ctx)`, and:

```python
    def _tick(self, ctx: Context) -> None:
        """Pool what the children reported since the last tick, run this
        level's detectors on it (zone drift at a fog, federation drift at the
        root) and pass it on; until the run is over."""
        if self.stopped:
            return
        now = ctx.now() * self.continuum.speed
        heard, self._heard = self._heard, {}
        for kind, detector in self._detectors.items():
            n = heard.get(f"drift.{kind}.n", 0.0)
            if n <= 0:
                continue
            value = heard[f"drift.{kind}.sum"] / n
            found = detector.update(value)
            if found is not None:
                key = f"drift.{kind}.detected"
                heard[key] = heard.get(key, 0.0) + 1.0
                ctx.emit("drift.detected", found, kind=kind, value=value, at=now)
        self._heard_all(heard, now, ctx)
        ctx.set_timer(self.continuum.status_every, "status")

    def _heard_all(self, heard: dict[str, float], now: float, ctx: Context) -> None:
        raise NotImplementedError
```

`_Collector` has no `stopped`. Add `stopped: bool = False` as a class attribute on `_Collector`, so `Aggregator` (which sets it) and `Coordinator` (whose `stopped` property returns `finished`) both answer.

`Aggregator`:

```python
    def _heard_all(self, heard: dict[str, float], now: float, ctx: Context) -> None:
        payload = Payload(metrics={"at": now, **heard})
        ctx.send(Message(kind="status", src=self.id, dst=self.parent, payload=payload))
```

Its `on_timer` already delegates unknown names to `super().on_timer`, and so does the coordinator's.

`Coordinator.__init__`:

```python
        # A continuous federation (continuum C6): what the statuses said since
        # the last round opened, and when it opened (data time).
        self._volume, self._drift, self._opened_at = 0.0, {}, 0.0
```

```python
    def _heard_all(self, heard: dict[str, float], now: float, ctx: Context) -> None:
        self._volume += heard.get("fresh", 0.0)
        for key, value in heard.items():
            if key.startswith("drift.") and key.endswith(".detected") and value:
                kind = key.split(".")[1]
                self._drift[kind] = self._drift.get(kind, 0) + int(value)
        self._maybe_open(ctx)

    def _maybe_open(self, ctx: Context) -> None:
        """Open the next round if the trigger asks (spec §10). At the first tick
        after the last label, one final round, after which the run finishes."""
        if self.open or self.finished or self.round == 0 or self.round >= self.rounds:
            return
        now = ctx.now() * self.continuum.speed
        if ctx.now() >= self.continuum.until - 1e-9:
            name, self.rounds = "horizon", self.round + 1  # the last one
        else:
            view = View(now=now, since=self._opened_at, volume=self._volume, drift=dict(self._drift))
            name = self.continuum.trigger.fired(view)
            if name is None:
                return
        self._start(name, now, ctx)

    def _start(self, name: str, now: float, ctx: Context) -> None:
        ctx.emit(
            "trigger.fired", self._volume, trigger=name, at=now, round=self.round + 1, drift=dict(self._drift)
        )
        self._volume, self._drift, self._opened_at = 0.0, {}, now
        ctx.emit("round.started", self.round + 1, round=self.round + 1, at=now)
        self._open(self.round + 1, self.state, ctx, bootstrap=self.round == 0)
```

`Coordinator._next`, after the finish check:

```python
        if self.continuum is not None:  # rounds wait for a trigger (spec §10)
            if self.round == 0:
                self._start("start", ctx.now() * self.continuum.speed, ctx)
            else:
                self._maybe_open(ctx)
            return
```

In `federation.py`, pass `continuum=continuum` to the `Coordinator` and every `Aggregator`.

- [ ] **Step 4: Run them and see them pass**

Run: `.venv/Scripts/python -m pytest tests/test_roles_stream.py tests/test_roles_protocol.py tests/test_runtime_equivalence.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/roles/nodes.py src/onion_fl/roles/federation.py tests/test_roles_stream.py
git commit -m "feat(roles): Open rounds when the federated trigger fires, until the last label"
```

---

### Task 5: The runner builds the pace from the config

**Files:**
- Modify: `src/onion_fl/experiment/runner.py` (`_paced`, `build_scenario`, `_stream_summary`)
- Test: `tests/test_experiment_stream.py`

**Interfaces:**
- Consumes: `ContinuumConfig` (Task 2), `Continuum`, `drift_detectors`, `triggers`, `build_federation(continuum=...)`.
- Produces: `_paced(config, streams) -> tuple[int, float | None, Continuum | None]`; the plan's `stream` preview gains `"trigger"` and gives `"rounds": None` when triggers pace the rounds.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_experiment_stream.py`:

```python
def test_a_schedule_trigger_runs_as_round_every_did(workspace: Path) -> None:
    paced = run(workspace)
    triggered = run(workspace, stream=PACED, continuum=SCHEDULE)

    with np.load(paced / "model.npz") as a, np.load(triggered / "model.npz") as b:
        assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)


def test_the_events_carry_the_data_time_of_each_trigger(workspace: Path) -> None:
    path = run(
        workspace,
        stream=PACED,
        continuum={"trigger": {"name": "volume", "samples": 8}, "status_every": 60},
    )

    fired = events(path, "trigger.fired")
    assert fired[0]["tags"]["trigger"] == "start"
    assert fired[-1]["tags"]["trigger"] == "horizon"
    assert all("at" in e["tags"] for e in fired + events(path, "round.started"))


def test_the_plan_names_the_trigger(workspace: Path) -> None:
    raw = experiment(workspace, stream=PACED, continuum=SCHEDULE)
    (preview,) = plan(parse_experiment(raw))

    assert preview["stream"]["rounds"] is None
    assert preview["stream"]["trigger"] == "schedule"
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_experiment_stream.py -q -k "as_round_every or data_time or names_the_trigger"`
Expected: FAIL with `TypeError: unsupported operand type(s) for /: 'NoneType' and 'float'` in `_paced`.

- [ ] **Step 3: Implement**

```python
def _paced(
    config: ExperimentConfig, streams: Mapping[str, Any]
) -> tuple[int, Any, Continuum | None]:
    """Rounds, their virtual spacing, and a continuous federation's pace: a
    stream lasts until its last label; triggers or ``round_every`` pace it."""
    if config.stream is None:
        return config.rounds, None, None
    drain = max((s.drain for s in streams.values()), default=0.0)
    speed = config.stream.speed
    c = config.continuum
    if c is not None:
        trigger = create(triggers, c.trigger)
        local = None if c.edge_trigger is None else create(triggers, c.edge_trigger)
        pace = Continuum(
            trigger=trigger,
            edge_trigger=local,
            status_every=c.status_every / speed,
            speed=speed,
            until=drain / speed,
            drift=drift_detectors(trigger, local),
        )
        return config.rounds, None, pace  # rounds stays an upper bound
    every = config.stream.round_every
    rounds = min(config.rounds, math.ceil(drain / every) + 1)  # one at or after it
    return rounds, every / speed, None
```

In `build_scenario`: `rounds, round_every, pace = _paced(...)`, then pass `continuum=pace` to `build_federation`. In `_stream_summary`:

```python
    rounds, _, pace = _paced(config, learners)
    ...
        "rounds": None if pace is not None else rounds,
        **({} if pace is None else {"trigger": _name(config.continuum.trigger)}),
```

Imports: `from onion_fl.continuum.pace import Continuum` and `from onion_fl.continuum.triggers import drift_detectors, triggers`.

- [ ] **Step 4: Run them and see them pass**

Run: `.venv/Scripts/python -m pytest tests/test_experiment_stream.py tests/test_experiment.py tests/test_studio.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/onion_fl/experiment/runner.py tests/test_experiment_stream.py
git commit -m "feat(experiments): Pace a stream run by the continuum's triggers"
```

---

### Task 6: Documentation

**Files:** `docs/architecture.md` (continuum section), `README.md` (configuration list, status table, limitations), the spec's §10 (a "Decisiones de C6" list, in Spanish).

- [ ] **Step 1:** In `docs/architecture.md`, after the replay memory bullet, add a **Triggers** bullet. It covers:
  - `continuum: {trigger, edge_trigger, status_every}`;
  - the four triggers, and what the two levels see;
  - statuses without data;
  - pooled zone and federation detection;
  - the final `horizon` round;
  - `round_every` versus a schedule;
  - the events and their `at` tags.
- [ ] **Step 2:** In `README.md`:
  - add a **Triggers** item after **Replay**, with a YAML example;
  - mark C6 done in the Continuum row and the v0.7.0 row;
  - retitle the limitations block to C5 and C7, removing what C6 lifts (rounds at a fixed pace);
  - update the test count.
- [ ] **Step 3:** In the spec's §10, add "Decisiones de C6 (#161; el plan, `docs/superpowers/plans/2026-10-07-continuum-c6-triggers.md`, da el detalle)" with Rulings 1–9 in Spanish, as §7 does for C3.
- [ ] **Step 4:** Commit: `docs(continuum): Describe triggers and continuous federation`.

---

### Task 7: The triggers on real data

**Files:** Create `experiments/stream_triggers_swell_wesad.yaml` and `results/continuum_triggers/`.

- [ ] **Step 1:** Write the experiment. It uses the stream, labels and training of `stream_replay_swell_wesad.yaml` without replay and without `round_every`, and `status_every: 5m`. It sweeps `continuum.trigger` over:
  - `{name: schedule, every: 10m}` (C3's pace);
  - `{name: volume, samples: 500}`;
  - `{name: drift, kind: prior}`;
  - seeds 0–2.
- [ ] **Step 2:** Run it twice at a clean commit (`onion_fl run experiments/stream_triggers_swell_wesad.yaml --workers 3`), the second time as the determinism check.
- [ ] **Step 3:** Summarise it in `results/continuum_triggers/`: the INDEX plus CSVs, with run folders without `events.jsonl` or `bundle/`, as in `continuum_replay/`. Per scenario report:
  - rounds;
  - triggers fired by name;
  - zone and edge drift detections;
  - prequential macro-F1 per dataset;
  - the final model's offline test macro-F1;
  - the check that the schedule scenario equals C3's full-labels stream runs bit for bit.
- [ ] **Step 4:** Add the results to README §7. Commit: `docs(results): Run the triggers on the SWELL and WESAD streams`.

---

## Self-review

- **Spec coverage:**

  | Requirement | Where |
  |---|---|
  | §10: rounds opened by triggers, not N rounds | Task 4 |
  | §10: `schedule`, `volume`, `drift`, `any` | Task 2 |
  | §10: two triggers | Tasks 3 and 4 |
  | §10: statuses without data, aggregated by fogs | Tasks 3 and 4 |
  | §10: an edge joins only if its local trigger fired | Task 3 |
  | §10: durations in data time | Task 2; `Positive` parses durations |
  | §10: synchronous rounds | Task 4 |
  | §4.4: kinds of drift | Task 1, with Ruling 1 for the two left out |
  | Issue: drift per edge and per zone, as a trigger and as a diagnostic | Tasks 1, 3 and 4 |
  | Issue: SimRuntime runs days in minutes | `speed`, unchanged |
  | Issue: RealRuntime equivalence kept | Task 4 runs `test_runtime_equivalence.py` |
  | Issue: no dependency on #150 | Satisfied by design |
  | Acceptance: tests per trigger, data-time cursor in events | Tasks 4 and 5 |

- **Placeholders:** none; every step shows its code.
- **Type consistency:**
  - `Continuum` fields are used with the same names in Tasks 3–5.
  - `View(now, since, volume, drift)` is used the same way everywhere.
  - `drift_detectors(*tracked)` is used in Task 2's config and Task 5's runner.
  - Status metric keys (`fresh`, `drift.<kind>.sum`, `.n`, `.detected`, `at`) are the same in Tasks 3 and 4.
- **Review Focus:** each of the five has its test in Task 3 or Task 4.
