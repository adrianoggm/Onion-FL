"""Champion and challenger, promotion, rejection and rollback (continuum C7, #162).

Edges use the stub trainer, which moves every weight by one per training, and
validators a stub scorer that reads the model's weights: nothing learns. The
rows are a hand-written format fixture (docs/RULES.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from onion_fl.continual.promotion import promotions
from onion_fl.core.message import Message, Payload
from onion_fl.core.node import Node
from onion_fl.core.topology import parse_topology
from onion_fl.data.contract import SubjectData
from onion_fl.data.stream import LabelsConfig, StreamConfig, edge_stream
from onion_fl.learning.aggregators import server_optimizers
from onion_fl.learning.model import (
    DataShape,
    ModularMLP,
    ModularMLPConfig,
    state_arrays,
)
from onion_fl.learning.trainers import trainers
from onion_fl.roles import EdgeSpec, build_federation

A = DataShape(dataset="a", task="t", n_features=2, n_classes=2)
CONFIG = ModularMLPConfig(adapter_width=2, trunk_hidden=[2], dropout=0.0)
INITIAL = state_arrays(ModularMLP(CONFIG, [A], seed=0))
KEY = "trunk.0.weight"
SPEED, EVERY, BOOTSTRAP = 60.0, 300.0, 120.0


def tree(fog: dict | None = None, root: dict | None = None):
    return parse_topology(
        {
            "name": "t",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud", **(root or {})},
            "fog": {"defaults": fog or {}, "nodes": [{"id": "fog_0"}]},
        }
    )


def stream_of(subject: str, n: int = 20):
    t = np.arange(n) * 60.0
    rows = SubjectData(
        X=np.stack([t / 600, np.cos(t)], axis=1).astype(np.float32),
        y=(np.arange(n) // 4) % 2,
        dataset="a",
        subject=subject,
        task="t",
        n_classes=2,
        feature_names=["f0", "f1"],
        t=t,
    )
    config = StreamConfig(
        bootstrap=BOOTSTRAP, round_every=EVERY, batch_size=1, speed=SPEED
    )
    return edge_stream(rows, config, LabelsConfig(), seed=0)


class Serving:
    """The stub trainer, keeping the model each edge was served when it trained."""

    def __init__(self) -> None:
        self.stub = trainers.create("stub", {"shift": 1.0})
        self.served: list[float] = []

    def train(self, model, data=None, received=None, ctx=None):
        self.served.append(float(state_arrays(model)[KEY].mean()))
        return self.stub.train(model, data, received, ctx)


def scorer(forgets: float):
    """A stub scorer: one more unit of weight gains 0.01 on rows observed after
    t₀ and loses ``forgets`` on those up to it, the history."""
    base = float(INITIAL[KEY].mean())

    def evaluate(model, data):
        shift = float(state_arrays(model)[KEY].mean()) - base
        old = float(np.max(data.t)) <= BOOTSTRAP  # labelled by t₀
        return {"macro_f1": 0.5 + (-forgets if old else 0.01) * shift}, len(data.y)

    return evaluate


class Operator(Node):
    """Sends its orders to the coordinator at the given virtual times."""

    def __init__(self, orders: list[tuple[float, dict]], node_id="operator") -> None:
        super().__init__(node_id)
        self.orders = orders

    def on_start(self, ctx) -> None:
        for k, (at, _) in enumerate(self.orders):
            ctx.set_timer(at, f"order/{k}")

    def on_timer(self, name: str, ctx) -> None:
        meta = self.orders[int(name.split("/")[1])][1]
        ctx.send(Message(kind="control", src=self.id, dst="cloud", meta=meta))


def versioned(
    forgets: float = 0.001,
    policy: dict | None = None,
    rounds: int = 6,
    orders: list[tuple[float, dict]] | None = None,
    server_optimizer=None,
    continuum=None,
    evaluate=None,
    rogue: Node | None = None,
    operator: str = "operator",
    replace_validator: bool = False,
):
    serving = {name: Serving() for name in ("a1", "a2")}
    learners = [
        EdgeSpec(n, ModularMLP(CONFIG, [A], seed=0), trainer=t, stream=stream_of(n))
        for n, t in serving.items()
    ]
    validator = EdgeSpec(
        "validate-a-9",
        ModularMLP(CONFIG, [A], seed=0),
        train=False,
        stream=stream_of("a-9"),
        validates=True,
    )
    params = {"min_samples": 1} | (policy or {})
    rogues = (
        []
        if rogue is None or replace_validator
        else [EdgeSpec(rogue.id, ModularMLP(CONFIG, [A], seed=0), train=False)]
    )
    federation = build_federation(
        tree(),
        {"fog_0": learners, "cloud": [validator] + rogues},
        initial_state=INITIAL,
        rounds=rounds,
        round_every=None if continuum else EVERY / SPEED,
        continuum=continuum,
        evaluate=evaluate or scorer(forgets),
        promotion=promotions.create("champion_challenger", params),
        server_optimizer=server_optimizer,
    )
    if rogue is not None:  # a child of the root that is not what it says
        federation.runtime._nodes[rogue.id] = rogue
    if orders and operator != "rogue":
        federation.runtime.add_node(Operator(orders, operator))
        federation.runtime.add_link(operator, "cloud")
    federation.run()
    return federation, serving


def events(federation, name: str) -> list[dict]:
    return [e for e in federation.runtime.events if e["name"] == name]


def shift(state) -> float:
    return float(state[KEY].mean() - INITIAL[KEY].mean())


# --- the policy -------------------------------------------------------------------


def scores(recent: tuple[float, float], reference=(0.5, 0.5), n=(600, 300)) -> dict:
    return {
        model: {
            "recent": {"macro_f1": recent[i], "samples": n[0]},
            "reference": {"macro_f1": reference[i], "samples": n[1]},
        }
        for i, model in enumerate(("champion", "challenger"))
    }


def policy(**params):
    return promotions.create("champion_challenger", params)


def test_a_challenger_better_on_the_recent_window_is_promoted() -> None:
    decision = policy().decide(scores((0.60, 0.65)))

    assert decision.promote and decision.reason == "promoted"
    assert decision.measures["delta"] == pytest.approx(0.05)


def test_a_challenger_that_forgets_more_than_max_forgetting_is_rejected() -> None:
    decision = policy(max_forgetting=0.03).decide(
        scores((0.60, 0.70), reference=(0.80, 0.75))
    )

    assert not decision.promote and decision.reason == "max_forgetting"
    assert decision.measures["forgetting"] == pytest.approx(0.05)


def test_a_challenger_worse_than_min_delta_on_the_recent_window_is_rejected() -> None:
    assert policy(min_delta=-0.01).decide(scores((0.60, 0.595))).promote
    decision = policy(min_delta=-0.01).decide(scores((0.60, 0.58)))

    assert not decision.promote and decision.reason == "min_delta"


def test_too_few_recent_labels_decide_nothing_and_keep_the_champion() -> None:
    decision = policy(min_samples=500).decide(scores((0.1, 0.9), n=(499, 300)))

    assert not decision.promote and decision.reason == "min_samples"


def test_without_a_reference_there_is_nothing_to_forget() -> None:
    decision = policy().decide(scores((0.6, 0.6), n=(600, 0)))

    assert decision.promote and decision.measures["forgetting"] == 0.0


def test_a_loss_is_better_lower() -> None:
    lower = policy(metric="loss").decide(
        {
            "champion": {"recent": {"loss": 0.7, "samples": 600}, "reference": {}},
            "challenger": {"recent": {"loss": 0.5, "samples": 600}, "reference": {}},
        }
    )

    assert lower.promote and lower.measures["delta"] == pytest.approx(0.2)


# --- in a federation --------------------------------------------------------------


def test_a_challenger_that_forgets_too_much_never_reaches_the_edges() -> None:
    federation, serving = versioned(forgets=0.1, policy={"max_forgetting": 0.03})

    rejected = events(federation, "version.rejected")
    assert rejected and not events(federation, "version.promoted")
    # the first has no recent labels yet; every later one forgets the history
    assert [e["tags"]["reason"] for e in rejected][1:] == ["max_forgetting"] * (
        len(rejected) - 1
    )
    assert all(e["tags"]["forgetting"] > 0.03 for e in rejected[1:])
    assert all(e["tags"]["delta"] > 0 for e in rejected[1:])  # it did help lately
    # the edges are served the champion only, and it is still the initial model
    assert federation.coordinator.champion == 0
    assert shift(federation.coordinator.state) == pytest.approx(0.0)
    assert all(
        s == pytest.approx(float(INITIAL[KEY].mean())) for s in serving["a1"].served
    )


def test_a_promoted_challenger_is_what_the_edges_serve_next() -> None:
    federation, serving = versioned(forgets=0.001)

    promoted = events(federation, "version.promoted")
    candidates = events(federation, "version.candidate")
    assert promoted and len(candidates) == len(promoted) + len(
        events(federation, "version.rejected")
    )
    chain = [(e["tags"]["superseded"], e["value"]) for e in promoted]
    assert all(
        a == b_prev for (a, _), (_, b_prev) in zip(chain[1:], chain, strict=False)
    )
    coordinator = federation.coordinator
    assert coordinator.champion == promoted[-1]["value"]
    # every training starts from a champion, never from a challenger
    champions = {0.0} | {
        round(shift(coordinator.versions[e["value"]].state), 4) for e in promoted
    }
    base = float(INITIAL[KEY].mean())
    served = [round(s - base, 4) for s in serving["a1"].served]
    assert set(served) <= champions and served == sorted(served)


def test_every_candidate_names_its_parent_and_its_digest() -> None:
    federation, _ = versioned()

    candidates = events(federation, "version.candidate")
    versions = federation.coordinator.versions
    for e in candidates:
        version = versions[e["value"]]
        assert e["tags"]["parent"] == version.parent
        assert e["tags"]["digest"] == version.digest


def test_a_rollback_restores_the_champion_before_with_its_server_state() -> None:
    late = 4.5 * EVERY / SPEED  # after a few promotions
    optimizer = server_optimizers.create("fedavgm", {"momentum": 0.5})
    federation, _ = versioned(
        orders=[(late, {"rollback": None})], rounds=5, server_optimizer=optimizer
    )

    rollback = events(federation, "version.rollback")
    assert len(rollback) == 1
    coordinator, versions = federation.coordinator, federation.coordinator.versions
    left = versions[rollback[0]["tags"]["previous"]]
    back = versions[rollback[0]["value"]]
    assert left.status == "rolled_back" and left.replaced == back.v
    assert coordinator.champion == back.v and back.status == "active"
    for key, value in back.state.items():
        np.testing.assert_array_equal(coordinator.state[key], value)
    for key, value in back.server.items():
        np.testing.assert_array_equal(optimizer.state()[key], value)


def test_a_rejection_leaves_no_trace_in_the_server_optimizer() -> None:
    optimizer = server_optimizers.create("fedavgm", {"momentum": 0.5})
    federation, _ = versioned(forgets=0.1, server_optimizer=optimizer)

    assert not events(federation, "version.promoted")
    assert optimizer.state() == {}  # the initial model's: no momentum yet


def test_an_operator_promotes_a_rejected_version() -> None:
    federation, serving = versioned(
        forgets=0.1, orders=[(2.5 * EVERY / SPEED, {"promote": 1})]
    )

    promoted = events(federation, "version.promoted")
    assert (promoted[0]["value"], promoted[0]["tags"]["reason"]) == (1, "operator")
    assert promoted[0]["tags"]["superseded"] == 0
    assert federation.coordinator.versions[1].replaced == 0
    # the edges were served v1 from the next round on
    base = float(INITIAL[KEY].mean())
    assert round(serving["a1"].served[3] - base, 4) == 1.0


def test_an_order_for_a_version_that_cannot_take_it_is_refused() -> None:
    federation, _ = versioned(forgets=0.1, orders=[(EVERY / SPEED, {"rollback": 7})])

    refused = [
        e
        for e in events(federation, "message.rejected")
        if e["node"] == "cloud" and e["tags"]["src"] == "operator"
    ]
    assert len(refused) == 1 and not events(federation, "version.rollback")


def test_a_nan_score_is_never_promoted() -> None:
    nan = float("nan")
    for given in (scores((0.6, nan)), scores((0.6, 0.7), reference=(nan, 0.5))):
        decision = policy().decide(given)
        assert not decision.promote and decision.reason == "non_finite"
    decision = policy().decide(scores((0.6, 0.7), n=(nan, 300)))
    assert not decision.promote and decision.reason == "non_finite"


def test_a_missing_metric_rejects_instead_of_raising() -> None:
    given = scores((0.6, 0.7))
    del given["challenger"]["recent"]["macro_f1"]

    decision = policy().decide(given)

    assert not decision.promote and decision.reason == "missing_metric"
    assert not policy().decide({}).promote  # nothing reported at all


def test_the_champion_needs_its_recent_samples_too() -> None:
    given = scores((0.6, 0.7))
    given["champion"]["recent"]["samples"] = 10

    decision = policy(min_samples=500).decide(given)

    assert not decision.promote and decision.reason == "min_samples"


class Rogue(Node):
    """A child of the root that calls itself a validator and reports a perfect
    challenger, and gives orders, every virtual second."""

    def __init__(self) -> None:
        super().__init__("rogue")

    def on_start(self, ctx) -> None:
        meta = {"role": "validator", "edges": 0, "tags": {}}
        ctx.send(Message(kind="hello", src=self.id, dst="cloud", meta=meta))
        ctx.set_timer(1.0, "lie")

    def on_timer(self, name: str, ctx) -> None:
        perfect = {
            f"{m}.{p}.{k}": (1.0 if m == "challenger" else 0.0)
            if k != "samples"
            else 9e3
            for m in ("champion", "challenger")
            for p in ("recent", "reference")
            for k in ("macro_f1", "samples")
        }
        for v in range(1, 8):
            ctx.send(
                Message(
                    kind="eval_report",
                    src=self.id,
                    dst="cloud",
                    round=v,
                    payload=Payload(metrics=perfect),
                    meta={"validate": v},
                )
            )
        ctx.send(Message(kind="control", src=self.id, dst="cloud", meta={"promote": 1}))
        if ctx.now() < 30:
            ctx.set_timer(1.0, "lie")


def test_only_the_configured_validators_decide() -> None:
    federation, _ = versioned(forgets=0.1, rogue=Rogue())

    coordinator = federation.coordinator
    assert coordinator.validators() == ["validate-a-9"]
    assert not events(federation, "version.promoted")  # the lies changed nothing
    asked = [
        e
        for e in events(federation, "message.delivered")
        if e["node"] == "rogue" and e["tags"]["kind"] == "eval_request"
    ]
    assert not asked
    refused = [
        e
        for e in events(federation, "message.rejected")
        if e["node"] == "cloud" and e["tags"]["src"] == "rogue"
    ]
    assert {e["tags"]["kind"] for e in refused} == {"eval_report", "control"}


@pytest.mark.parametrize("sender", ["rogue", "intruder"])
def test_an_order_from_a_child_or_a_stranger_is_rejected(sender: str) -> None:
    # the rogue is a child of the root; the intruder is outside, not configured
    federation, _ = versioned(
        forgets=0.1,
        orders=[(2.5 * EVERY / SPEED, {"promote": 1})],
        operator=sender,
        rogue=Rogue() if sender == "rogue" else None,
    )

    assert not events(federation, "version.promoted")
    refused = [
        e
        for e in events(federation, "message.rejected")
        if e["node"] == "cloud" and e["tags"]["kind"] == "control"
    ]
    assert refused and all(
        e["tags"]["reason"] == f"{e['tags']['src']} is not an operator" for e in refused
    )


def test_a_validator_that_cannot_score_decides_nothing() -> None:
    def broken(model, data):
        return {"macro_f1": float("nan")}, len(data.y)

    federation, _ = versioned(evaluate=broken)

    rejected = events(federation, "version.rejected")
    assert rejected and not events(federation, "version.promoted")
    assert {e["tags"]["reason"] for e in rejected} <= {"min_samples", "missing_metric"}
    assert "missing_metric" in {e["tags"]["reason"] for e in rejected}


class Lying(Node):
    """The configured validator, reporting negative samples for its version."""

    def __init__(self) -> None:
        super().__init__("validate-a-9")

    def on_start(self, ctx) -> None:
        meta = {"role": "validator", "edges": 0, "tags": {}}
        ctx.send(Message(kind="hello", src=self.id, dst="cloud", meta=meta))

    def on_message(self, msg, ctx) -> None:
        if msg.kind != "eval_request":
            return
        lie = {
            f"{m}.{p}.{k}": (1.0 if m == "challenger" else 0.0)
            if k != "samples"
            else -5.0
            for m in ("champion", "challenger")
            for p in ("recent", "reference")
            for k in ("macro_f1", "samples")
        }
        ctx.send(
            Message(
                kind="eval_report",
                src=self.id,
                dst="cloud",
                round=msg.round,
                payload=Payload(metrics=lie),
                meta={"validate": msg.meta["validate"]},
            )
        )


def test_a_validation_report_with_bad_values_is_left_out() -> None:
    federation, _ = versioned(rogue=Lying(), replace_validator=True)

    refused = [
        e
        for e in events(federation, "message.rejected")
        if e["tags"]["src"] == "validate-a-9"
    ]
    assert refused and all("bad values" in e["tags"]["reason"] for e in refused)
    assert not events(federation, "version.promoted")
    assert {e["tags"]["reason"] for e in events(federation, "version.rejected")} == {
        "min_samples"
    }


def test_an_operator_cannot_be_a_child() -> None:
    learner = EdgeSpec("a1", ModularMLP(CONFIG, [A], seed=0), trainer=Serving())
    with pytest.raises(ValueError, match="operator"):
        build_federation(
            tree(),
            {"fog_0": [learner]},
            initial_state=INITIAL,
            rounds=1,
            promotion=promotions.create("champion_challenger"),
            operators=["fog_0"],
        )


def test_a_versioned_run_is_deterministic() -> None:
    def trace():
        federation, _ = versioned()
        return [
            (e["name"], e["node"], e["tags"].get("round"), e["value"])
            for e in federation.runtime.events
        ]

    assert trace() == trace()


def test_a_trigger_opens_no_round_before_the_decision() -> None:
    from onion_fl.continual.pace import Continuum
    from onion_fl.continual.triggers import triggers

    pace = Continuum(
        trigger=triggers.create("schedule", {"every": 60}),
        edge_trigger=None,
        status_every=60 / SPEED,
        speed=SPEED,
        until=stream_of("a1").drain / SPEED,
        drift={},
    )
    federation, _ = versioned(rounds=1000, continuum=pace)

    decided = {
        e["tags"]["round"]: e["t"]
        for e in events(federation, "version.promoted")
        + events(federation, "version.rejected")
    }
    started = events(federation, "round.started")
    assert len(decided) == len(events(federation, "version.candidate")) > 3
    for e in started[1:]:
        before = e["tags"]["round"] - 1
        assert before not in decided or decided[before] <= e["t"]
    assert events(federation, "run.finished")[0]["tags"]["reason"] == "horizon"


def test_groups_kept_below_the_root_cannot_be_versioned() -> None:
    learner = EdgeSpec("a1", ModularMLP(CONFIG, [A], seed=0), trainer=Serving())
    with pytest.raises(ValueError, match="kept below the root"):
        build_federation(
            tree(),
            {"fog_0": [learner]},
            initial_state=INITIAL,
            rounds=1,
            sharing={"name": "zone"},
            promotion=promotions.create("champion_challenger"),
        )


def test_a_validator_needs_a_stream() -> None:
    from onion_fl.roles import Edge

    with pytest.raises(ValueError, match="validator"):
        Edge(
            "v",
            "cloud",
            model=None,
            sharing=None,
            levels=["global", "edge"],
            train=False,
            validates=True,
        )
