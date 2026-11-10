"""How a stream edge updates: warm, frozen and scratch (continuum C8, issue #163).

The baselines of the benchmark differ only in what an edge trains from and on.
Every edge uses the stub trainer (or a wrapper that records what it was
given), so nothing learns; the rows are the stream tests' format fixture
(docs/RULES.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from onion_fl.experiment.config import ConfigError, parse_experiment
from onion_fl.experiment.runner import _update
from onion_fl.experiment.sweep import scenarios
from onion_fl.learning.model import ModularMLP, state_arrays
from onion_fl.roles import EdgeSpec, build_federation
from tests.test_experiment_stream import experiment, workspace  # noqa: F401
from tests.test_roles_stream import (
    CONFIG,
    EVERY,
    INITIAL,
    SPEED,
    A,
    Recording,
    events,
    stream_of,
    tree,
)


class Starts(Recording):
    """Also keeps the weights each training starts from."""

    def __init__(self) -> None:
        super().__init__()
        self.starts: list[dict[str, np.ndarray]] = []

    def train(self, model, data=None, received=None, ctx=None):
        self.starts.append(state_arrays(model))
        return super().train(model, data, received, ctx)


def updating(update: str, rounds: int = 6, stream=None, **kw):
    trainer = Starts()
    spec = EdgeSpec(
        "a1",
        ModularMLP(CONFIG, [A], seed=0),
        trainer=trainer,
        stream=stream_of("a-1") if stream is None else stream,
        update=update,
        **kw,
    )
    federation = build_federation(
        tree(1),
        {"fog_0": [spec]},
        initial_state=INITIAL,
        rounds=rounds,
        round_every=EVERY / SPEED,
    )
    federation.run()
    return federation, trainer


def same(one: dict, two: dict) -> bool:
    return one.keys() == two.keys() and all(np.array_equal(one[k], two[k]) for k in one)


def test_warm_trains_each_row_once_from_the_model_received() -> None:
    federation, trainer = updating("warm")

    times = np.concatenate([t for _, t in trainer.calls])
    assert len(times) == len(set(times.tolist()))  # no row twice
    assert same(trainer.starts[0], INITIAL)
    assert not same(trainer.starts[1], INITIAL)  # from v0, as received


def test_frozen_trains_v0_only_and_is_idle_from_then_on() -> None:
    federation, trainer = updating("frozen")

    assert len(trainer.calls) == 1  # v0, on the history
    (v0,) = events(federation, "edge.trained", "a1")
    assert v0["tags"]["round"] == 1
    idle = events(federation, "round.idle", "cloud")
    assert len(idle) == len(events(federation, "round.started", "cloud")) - 1
    # it still predicts and scores what arrives, with v0
    scored = events(federation, "eval.accuracy", "a1")
    assert {e["tags"]["model"] for e in scored} >= {"prequential"}


def test_frozen_without_labels_in_its_history_never_trains() -> None:
    # labels 10 minutes late: none of the history's is there for v0
    late = stream_of("a-1", delay=600.0)
    federation, trainer = updating("frozen", stream=late)
    _, warm = updating("warm", stream=late)

    assert not trainer.calls
    assert warm.calls  # warm trains once the labels come


def test_scratch_starts_every_training_from_the_initial_weights() -> None:
    federation, trainer = updating("scratch")

    assert len(trainer.calls) > 2
    assert all(same(start, INITIAL) for start in trainer.starts)


def test_scratch_trains_on_every_row_it_has_and_weighs_them_all() -> None:
    federation, trainer = updating("scratch")

    sizes = [len(t) for _, t in trainer.calls]
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0]
    for (_, before), (_, after) in zip(trainer.calls, trainer.calls[1:], strict=False):
        assert set(before.tolist()) <= set(after.tolist())  # nothing dropped


def test_scratch_trains_only_when_new_rows_came_as_warm_does() -> None:
    federation, trainer = updating("scratch", rounds=8)
    warm_federation, _ = updating("warm", rounds=8)

    def rounds(f) -> list[int]:
        return [e["tags"]["round"] for e in events(f, "edge.trained", "a1")]

    assert rounds(federation) == rounds(warm_federation)
    assert len(events(federation, "round.idle", "cloud")) > 0


def test_an_unknown_update_is_refused() -> None:
    with pytest.raises(ValueError, match="update must be one of"):
        updating("cold")


@pytest.mark.parametrize("update", ["frozen", "scratch"])
def test_frozen_and_scratch_take_no_memory(update: str) -> None:
    from onion_fl.continual.memory import memories

    replay = memories.create("fifo", {"capacity": 8})
    with pytest.raises(ValueError, match="need warm"):
        updating(update, replay=replay)


# --- the configuration -----------------------------------------------------------


def test_warm_keeps_the_config_id(workspace: Path) -> None:  # noqa: F811
    plain = parse_experiment(experiment(workspace, continual={"memory": "none"}))
    warm = parse_experiment(
        experiment(workspace, continual={"memory": "none", "update": "warm"})
    )
    assert "update" not in plain.dump()["continual"]
    assert plain.dump() == warm.dump()
    assert _update(next(iter(scenarios(warm)))) == {}


@pytest.mark.parametrize("update", ["frozen", "scratch"])
def test_each_edge_gets_the_configured_update(
    workspace: Path,  # noqa: F811
    update: str,
) -> None:
    config = parse_experiment(experiment(workspace, continual={"update": update}))
    assert _update(next(iter(scenarios(config)))) == {"update": update}


@pytest.mark.parametrize(
    "change, message",
    [
        (
            {
                "continual": {
                    "update": "scratch",
                    "memory": {"name": "fifo", "capacity": 8},
                }
            },
            "needs warm",
        ),
        (
            {"continual": {"update": "frozen"}, "unlabelled": {"name": "consistency"}},
            "use warm",
        ),
    ],
)
def test_an_update_that_cannot_use_what_it_is_given_is_refused(
    workspace: Path,  # noqa: F811
    change: dict,
    message: str,
) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_experiment(experiment(workspace, **change))
