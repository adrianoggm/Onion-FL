"""Evaluation at the three levels over SimRuntime (issue #92).

The edges use the stub trainer and the evaluation function is a stub too: it
scores a model by the mean of its trunk weights and declares a fixed sample
count, so every expected number is arithmetic (docs/RULES.md).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from onion_fl.core.topology import parse_topology
from onion_fl.learning.model import (
    DataShape,
    ModularMLP,
    ModularMLPConfig,
    state_arrays,
)
from onion_fl.learning.trainers import trainers
from onion_fl.roles import EdgeSpec, build_federation

A = DataShape(dataset="a", task="t", n_features=2, n_classes=2)
B = DataShape(dataset="b", task="t", n_features=3, n_classes=2)
CONFIG = ModularMLPConfig(adapter_width=2, trunk_hidden=[2], dropout=0.0)
KEY = "trunk.0.weight"


def model(*shapes: DataShape) -> ModularMLP:
    return ModularMLP(CONFIG, list(shapes), seed=0)


INITIAL = state_arrays(model(A, B))
M0 = float(INITIAL[KEY].mean())


def stub_score(model, data):
    return {"accuracy": float(state_arrays(model)[KEY].mean())}, data.samples


def samples(n: int) -> SimpleNamespace:
    return SimpleNamespace(samples=n)


def tree(
    fog: dict | None = None,
    root: dict | None = None,
    edge: dict | None = None,
    n_fogs: int = 1,
):
    return parse_topology(
        {
            "name": "t",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"} | (root or {}),
            "fog": {
                "defaults": fog or {},
                "nodes": [{"id": f"fog_{i}"} for i in range(n_fogs)],
            },
            "edge": edge or {},
        }
    )


def trainer_edge(
    node_id: str, shift: float = 1.0, val: int | None = None, shape=A
) -> EdgeSpec:
    stub = trainers.create("stub", {"shift": shift})
    return EdgeSpec(
        node_id,
        model(shape),
        trainer=stub,
        val_data=None if val is None else samples(val),
    )


def evaluator(node_id: str, n: int, shape=A, **tags) -> EdgeSpec:
    return EdgeSpec(node_id, model(shape), data=samples(n), train=False, tags=tags)


def run(topology, edges, rounds: int = 1, **kw):
    federation = build_federation(
        topology, edges, initial_state=INITIAL, rounds=rounds, evaluate=stub_score, **kw
    )
    federation.run()
    return federation


def scores(federation, node: str, **match) -> list[dict]:
    return [
        e
        for e in federation.runtime.events
        if e["node"] == node
        and e["name"] == "eval.accuracy"
        and all(e["tags"].get(k) == v for k, v in match.items())
    ]


EDGE_EVAL = {"eval": {"every": 1, "models": ["received", "local"]}}


# --- edge -------------------------------------------------------------------------------------


def test_an_edge_scores_the_received_and_the_trained_model() -> None:
    federation = run(
        tree(edge=EDGE_EVAL), {"fog_0": [trainer_edge("e1", shift=1, val=4)]}, rounds=2
    )

    received = [e["value"] for e in scores(federation, "e1", model="received")]
    local = [e["value"] for e in scores(federation, "e1", model="local")]
    assert received == pytest.approx([M0, M0 + 1])
    assert local == pytest.approx([M0 + 1, M0 + 2])
    assert all(e["tags"]["samples"] == 4 for e in scores(federation, "e1"))


def test_edges_without_local_val_do_not_score() -> None:
    federation = run(tree(edge=EDGE_EVAL), {"fog_0": [trainer_edge("e1")]})

    assert scores(federation, "e1") == []


def test_edge_scores_follow_every() -> None:
    edge = {"eval": {"every": 2, "models": ["local"]}}

    federation = run(tree(edge=edge), {"fog_0": [trainer_edge("e1", val=1)]}, rounds=4)

    assert [e["tags"]["round"] for e in scores(federation, "e1")] == [2, 4]


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


class BrokenPersonal(PersonalStub):
    def personal(self):
        raise RuntimeError("personal model lost")


def test_a_failing_score_does_not_drop_the_edge_update() -> None:
    spec = EdgeSpec("e1", model(A), trainer=BrokenPersonal(), val_data=samples(4))

    federation = run(tree(edge=PERSONAL), {"fog_0": [spec]})

    assert federation.coordinator.state[KEY].mean() == pytest.approx(M0 + 1)
    (failed,) = [
        e for e in federation.runtime.events if e["name"] == "edge.eval_failed"
    ]
    assert "personal model lost" in failed["tags"]["error"]


class Failing:
    def train(self, model, data=None, received=None, ctx=None):
        raise RuntimeError("out of memory")


def test_a_failed_round_still_reports_the_scores_that_arrived() -> None:
    edges = {
        "fog_0": [trainer_edge("e1", shift=1, val=4)],
        "fog_1": [EdgeSpec("e2", model(A), trainer=Failing())],
    }

    federation = run(tree(fog={"deadline": 5}, edge=EDGE_EVAL, n_fogs=2), edges)

    failed = [
        e
        for e in federation.runtime.events
        if e["name"] == "round.quorum_failed" and e["node"] == "cloud"
    ]
    assert failed
    (cloud,) = scores(federation, "cloud", model="local", source="children")
    assert cloud["value"] == pytest.approx(M0 + 1)


def test_a_diverged_edge_recovers_and_reports_no_non_finite_score() -> None:
    class DivergesOnce:
        """Blows up on its first call and poisons its memory; a poisoned memory
        poisons every later round, so only a restored trainer recovers."""

        calls = 0  # on the class: the edge's snapshot cannot roll it back

        def __init__(self) -> None:
            self.memory = 0.0
            self.stub = trainers.create("stub", {"shift": 1.0})

        def snapshot(self) -> float:
            return self.memory

        def restore(self, saved: float) -> None:
            self.memory = saved

        def train(self, model, data=None, received=None, ctx=None):
            type(self).calls += 1
            result = self.stub.train(model, data, received, ctx)
            if type(self).calls == 1:
                self.memory = float("nan")
            if math.isnan(self.memory):
                with torch.no_grad():
                    dict(model.named_parameters())[KEY].fill_(float("nan"))
            return result

    edges = {
        "fog_0": [
            trainer_edge("e1", shift=2, val=4),
            EdgeSpec("e2", model(A), trainer=DivergesOnce(), val_data=samples(4)),
        ]
    }

    federation = run(tree(fog={"deadline": 5}, edge=EDGE_EVAL), edges, rounds=3)

    events = federation.runtime.events
    failed = [e for e in events if e["name"] == "edge.train_failed"]
    assert [e["node"] for e in failed] == ["e2"]
    scores = [e["value"] for e in events if e["name"].startswith("eval.")]
    assert scores and np.isfinite(scores).all()
    # round 1 is lost; rounds 2 and 3 move the trunk by (2 + 1) / 2 each
    assert federation.coordinator.state[KEY].mean() == pytest.approx(M0 + 3.0)


# --- fog --------------------------------------------------------------------------------------


def test_a_fog_combines_its_childrens_scores_by_samples() -> None:
    edges = {
        "fog_0": [
            trainer_edge("e1", shift=1, val=1),
            trainer_edge("e2", shift=3, val=3),
        ]
    }

    federation = run(tree(edge=EDGE_EVAL), edges)

    (local,) = scores(federation, "fog_0", model="local", source="children")
    assert local["value"] == pytest.approx(M0 + (1 * 1 + 3 * 3) / 4)
    assert local["tags"]["samples"] == 4


def test_the_root_combines_every_edge_through_the_fogs() -> None:
    edges = {
        "fog_0": [trainer_edge("e1", shift=1, val=1)],
        "fog_1": [trainer_edge("e2", shift=3, val=3)],
    }

    federation = run(tree(edge=EDGE_EVAL, n_fogs=2), edges)

    (local,) = scores(federation, "cloud", model="local", source="children")
    assert local["value"] == pytest.approx(M0 + 2.5)
    assert local["tags"]["samples"] == 4


def test_children_scores_can_stay_at_the_fog() -> None:
    fog = {"eval": {"aggregate_children": False}}

    federation = run(
        tree(fog=fog, edge=EDGE_EVAL), {"fog_0": [trainer_edge("e1", val=1)]}
    )

    assert scores(federation, "fog_0", source="children") == []
    assert scores(federation, "cloud", source="children") == []


def test_the_zone_model_is_scored_on_the_fogs_evaluators() -> None:
    edges = {
        "fog_0": [
            trainer_edge("e1", shift=2),
            trainer_edge("e2", shift=4),
            evaluator("v1", 5),
        ]
    }

    federation = run(tree(fog={"eval": {"every": 1}}), edges, rounds=2)

    zone = scores(federation, "fog_0", model="zone", source="evaluators")
    assert [e["value"] for e in zone] == pytest.approx([M0 + 3, M0 + 6])
    assert all(e["tags"]["samples"] == 5 for e in zone)


def test_holdout_can_be_switched_off() -> None:
    fog = {"eval": {"every": 1, "holdout": False}}

    federation = run(tree(fog=fog), {"fog_0": [trainer_edge("e1"), evaluator("v1", 5)]})

    assert scores(federation, "fog_0", model="zone") == []


# --- global -----------------------------------------------------------------------------------


def test_the_global_model_is_scored_per_dataset_on_the_test_evaluators() -> None:
    edges = {
        "fog_0": [trainer_edge("e1", shift=1)],
        "cloud": [
            evaluator("ta", 2, A, dataset="a"),
            evaluator("tb", 6, B, dataset="b"),
        ],
    }

    federation = run(tree(root={"eval": {"every": 1}}), edges)

    by_dataset = {
        e["tags"].get("dataset"): e for e in scores(federation, "cloud", model="global")
    }
    assert set(by_dataset) == {None, "a", "b"}
    for entry in by_dataset.values():
        assert entry["value"] == pytest.approx(M0 + 1)
    assert by_dataset["a"]["tags"]["samples"] == 2
    assert by_dataset[None]["tags"]["samples"] == 8


def test_global_scores_follow_every_and_always_include_the_last_round() -> None:
    edges = {"fog_0": [trainer_edge("e1")], "cloud": [evaluator("t", 1)]}

    federation = run(tree(root={"eval": {"every": 2}}), edges, rounds=5)

    rounds = [e["tags"]["round"] for e in scores(federation, "cloud", model="global")]
    assert rounds == [2, 4, 5]


def test_the_run_finishes_after_the_last_global_scores() -> None:
    edges = {"fog_0": [trainer_edge("e1")], "cloud": [evaluator("t", 1)]}

    federation = run(tree(root={"eval": {"every": 1}}), edges, rounds=2)

    (finished,) = [e for e in federation.runtime.events if e["name"] == "run.finished"]
    last_score = scores(federation, "cloud", model="global")[-1]
    assert finished["t"] >= last_score["t"]


def test_a_silent_evaluator_does_not_block_the_end_of_the_run() -> None:
    from onion_fl.runtime.devices import availability_models

    dead = availability_models.create(
        "schedule", {"offline": [[0.003, 1e9]]}
    )  # after its hello
    silent = evaluator("t", 1)
    silent.availability = dead
    edges = {"fog_0": [trainer_edge("e1")], "cloud": [silent]}

    federation = run(tree(root={"eval": {"every": 1}, "deadline": 30}), edges)

    names = [e["name"] for e in federation.runtime.events]
    assert names.count("run.finished") == 1
    assert "message.dropped_offline" in names
    assert scores(federation, "cloud", model="global") == []


def test_the_root_only_takes_evaluators() -> None:
    with pytest.raises(ValueError, match="evaluators only"):
        build_federation(
            tree(), {"cloud": [trainer_edge("e1")]}, initial_state=INITIAL, rounds=1
        )


# --- the default scorer uses the metric plugins (real data only) --------------------------------


@pytest.mark.skipif(
    not Path("data/samples/swell_real_sample.pkl").exists(),
    reason="swell_real_sample.pkl not available",
)
def test_the_metric_plugins_score_a_model_on_real_data() -> None:
    from onion_fl.datasets.samples import load_swell_sample_features
    from onion_fl.learning.metrics import evaluate

    X, y = load_swell_sample_features()
    shape = DataShape(dataset="s", task="t", n_features=X.shape[1], n_classes=2)
    data = SimpleNamespace(X=X, y=y, n_classes=2)

    out, n = evaluate(
        ModularMLP(CONFIG, [shape]), data, ["loss", "accuracy", "macro_f1"]
    )

    assert n == len(y) and 0 <= out["accuracy"] <= 1 and out["loss"] > 0


def test_global_scoring_without_evaluators_finishes_at_once() -> None:
    federation = run(tree(root={"eval": {"every": 1}}), {"fog_0": [trainer_edge("e1")]})

    assert [e["name"] for e in federation.runtime.events].count("run.finished") == 1


class Recorder:
    """Just enough Context for calling handlers by hand."""

    rng = None

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def send(self, msg):
        pass

    def emit(self, name, value=None, **tags):
        self.events.append((name, tags))

    def set_timer(self, delay, name):
        pass

    def cancel_timer(self, name):
        pass

    def now(self):
        return 0.0


def test_unexpected_eval_messages_are_rejected() -> None:
    from onion_fl.core.message import Message

    federation = build_federation(
        tree(),
        {"fog_0": [trainer_edge("e1"), evaluator("v1", 1)]},
        initial_state=INITIAL,
        rounds=1,
    )
    ctx = Recorder()
    fog = federation.aggregators["fog_0"]
    fog.on_message(
        Message(
            kind="hello", src="v1", dst="fog_0", meta={"role": "evaluator", "edges": 0}
        ),
        ctx,
    )

    fog.on_message(Message(kind="eval_report", src="v1", dst="fog_0", round=1), ctx)
    federation.edges["e1"].on_message(
        Message(kind="eval_request", src="fog_0", dst="e1", round=1), ctx
    )
    federation.edges["v1"].on_message(
        Message(kind="global_model", src="fog_0", dst="v1", round=1), ctx
    )

    rejected = [
        tags["reason"] for name, tags in ctx.events if name == "message.rejected"
    ]
    assert rejected[0] == "eval report nobody asked for"
    assert "trainer does not take eval_request" in rejected[1]
    assert "evaluator does not take global_model" in rejected[2]
