"""Operator orders from outside the tree: signed, scheduled, paused (continuum #203).

The federation is the one of ``test_continual_versions``: edges with the stub
trainer and a validator with a stub scorer that reads the model's weights, on a
hand-written format fixture. Nothing learns (docs/RULES.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from onion_fl.continual.control import (
    ENV,
    INBOX,
    LiveControl,
    OperatorError,
    append_order,
    authenticate,
    load_operators,
    parse_order,
    read_inbox,
    read_state,
    refusal,
    sign,
)
from onion_fl.continual.promotion import promotions
from onion_fl.core.message import Message
from onion_fl.learning.model import ModularMLP
from onion_fl.roles import EdgeSpec, build_federation
from tests.test_continual_versions import (
    CONFIG,
    EVERY,
    INITIAL,
    SPEED,
    A,
    Operator,
    Serving,
    scorer,
    stream_of,
    tree,
)

TOKEN = "a-token-of-sixteen-or-more"
KEYS = {"alice": TOKEN}
RUN = "run-1"
ROUND = EVERY / SPEED  # virtual seconds between rounds


def federation(keys=KEYS, audience=RUN, forgets=0.001, rounds=6):
    learners = [
        EdgeSpec(n, ModularMLP(CONFIG, [A], seed=0), trainer=Serving(), stream=s)
        for n, s in (("a1", stream_of("a1")), ("a2", stream_of("a2")))
    ]
    validator = EdgeSpec(
        "validate-a-9",
        ModularMLP(CONFIG, [A], seed=0),
        train=False,
        stream=stream_of("a-9"),
        validates=True,
    )
    return build_federation(
        tree(),
        {"fog_0": learners, "cloud": [validator]},
        initial_state=INITIAL,
        rounds=rounds,
        round_every=ROUND,
        evaluate=scorer(forgets),
        promotion=promotions.create("champion_challenger", {"min_samples": 1}),
        operator_keys=keys,
        audience=audience,
    )


def events(federation, name: str) -> list[dict]:
    return [e for e in federation.runtime.events if e["name"] == name]


def signed(order: dict, operator="alice", token=TOKEN, audience=RUN) -> dict:
    return sign(order, operator=operator, token=token, audience=audience)


def rejected(federation) -> list[str]:
    return [
        e["tags"]["reason"]
        for e in events(federation, "message.rejected")
        if e["tags"]["kind"] == "control"
    ]


# --- operators and signatures ---------------------------------------------------------


def test_operators_come_from_the_file_the_environment_names(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    assert load_operators() is None
    path = tmp_path / "operators.yaml"
    path.write_text(f"alice: {TOKEN}\n", encoding="utf-8")
    monkeypatch.setenv(ENV, str(path))

    assert load_operators() == KEYS


@pytest.mark.parametrize(
    "text", ["alice: short\n", "'../x': a-token-of-sixteen-or-more\n", "- alice\n"]
)
def test_a_weak_or_broken_operators_file_is_refused(tmp_path: Path, text) -> None:
    path = tmp_path / "operators.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(OperatorError):
        load_operators(path)


def test_a_token_names_its_operator() -> None:
    keys = KEYS | {"bob": "another-token-sixteen"}

    assert authenticate(keys, TOKEN) == "alice"
    assert authenticate(keys, "another-token-sixteen") == "bob"
    assert authenticate(keys, "a-token-of-sixteen-or-mor") is None
    assert authenticate(keys, "") is None and authenticate(None, TOKEN) is None


def test_a_signature_covers_the_order_the_operator_and_the_run() -> None:
    order = signed({"rollback": None})

    assert refusal(order, operator="alice", operators=KEYS, audience=RUN) is None
    tampered = order | {"rollback": 0}
    for given, operator, audience, why in [
        (tampered, "alice", RUN, "bad signature"),
        (order, "alice", "run-2", "bad signature"),
        (order, "bob", RUN, "bob is not an operator"),
        ({"rollback": None}, "alice", RUN, "unsigned order"),
        (order | {"auth": {"nonce": 1, "mac": "x"}}, "alice", RUN, "unsigned order"),
    ]:
        assert (
            refusal(given, operator=operator, operators=KEYS, audience=audience) == why
        )
    other = signed({"rollback": None}, token="another-token-sixteen")
    assert refusal(other, operator="alice", operators=KEYS, audience=RUN)


@pytest.mark.parametrize(
    "body, order",
    [
        ({"order": "promote", "version": 2}, {"promote": 2}),
        ({"order": "rollback"}, {"rollback": None}),
        ({"order": "rollback", "version": 1, "at": 12}, {"rollback": 1, "at": 12.0}),
        ({"order": "pause", "at": 3.5}, {"pause": True, "at": 3.5}),
        ({"order": "resume"}, {"resume": True}),
    ],
)
def test_an_api_order_takes_the_coordinators_form(body, order) -> None:
    assert parse_order(body) == order


@pytest.mark.parametrize(
    "body",
    [
        {"order": "stop"},
        {"order": "promote"},
        {"order": "promote", "version": True},
        {"order": "rollback", "version": -1},
        {"order": "pause", "version": 1},
        {"order": "resume", "at": 3},
        {"order": "pause", "at": float("nan")},
        {"order": "pause", "at": -1},
    ],
)
def test_a_bad_api_order_is_refused(body) -> None:
    with pytest.raises(OperatorError):
        parse_order(body)


# --- the coordinator checks every order ------------------------------------------------


def with_operator(orders, keys=KEYS, audience=RUN, operator="alice"):
    fed = federation(keys=keys, audience=audience)
    fed.runtime.add_node(Operator(orders, operator))
    fed.runtime.add_link(operator, "cloud")
    fed.run()
    return fed


def test_a_signed_order_rolls_back() -> None:
    fed = with_operator([(4.5 * ROUND, signed({"rollback": None}))])

    assert len(events(fed, "version.rollback")) == 1 and not rejected(fed)


@pytest.mark.parametrize(
    "order, why",
    [
        ({"rollback": None}, "unsigned order"),
        (signed({"rollback": None}, audience="another-run"), "bad signature"),
        (signed({"rollback": None}, token="another-token-sixteen"), "bad signature"),
    ],
)
def test_an_order_not_signed_for_this_run_is_refused(order, why) -> None:
    fed = with_operator([(4.5 * ROUND, order)])

    assert rejected(fed) == [why] and not events(fed, "version.rollback")


def test_a_replayed_order_is_refused() -> None:
    order = signed({"rollback": None})
    fed = with_operator([(3.5 * ROUND, order), (4.5 * ROUND, order)])

    assert len(events(fed, "version.rollback")) == 1
    assert rejected(fed) == ["replayed order"]


def test_without_operator_keys_nobody_is_an_operator() -> None:
    fed = with_operator([(4.5 * ROUND, signed({"rollback": None}))], keys={})

    assert rejected(fed) == ["alice is not an operator"]


# --- the live inbox ---------------------------------------------------------------------


def inbox(folder: Path, *orders: dict, operator: str = "alice") -> None:
    for order in orders:
        append_order(folder, operator, order)


def live(folder: Path, sleep=None, paused=False, **kw):
    fed = federation(**kw)
    control = LiveControl(
        folder,
        KEYS,
        RUN,
        "cloud",
        paused=paused,
        poll=0.0,
        sleep=sleep or (lambda s: None),
    )
    control.run(fed.runtime)
    return fed


def lifecycle(fed) -> list[tuple]:
    """What a run did, without the wall clock."""
    return [
        (
            e["t"],
            e["node"],
            e["name"],
            json.dumps(e["tags"], sort_keys=True, default=str),
        )
        for e in fed.runtime.events
    ]


def test_with_an_empty_inbox_a_live_run_is_the_plain_run(tmp_path: Path) -> None:
    plain = federation()
    plain.run()
    fed = live(tmp_path)

    assert lifecycle(fed) == lifecycle(plain)
    assert read_state(tmp_path) | {"t": None} == {
        "listening": False,
        "paused": False,
        "read": 0,
        "scheduled": [],
        "t": None,
    }


def test_a_scheduled_order_applies_at_its_time_and_repeats(tmp_path: Path) -> None:
    at = 4.5 * ROUND
    inbox(tmp_path, signed({"rollback": None, "at": at}))
    first = live(tmp_path)
    second = live(tmp_path)

    (rollback,) = events(first, "version.rollback")
    assert rollback["t"] >= at
    (received,) = events(first, "control.received")
    assert received["tags"]["order"] == {"rollback": None, "at": at}
    assert lifecycle(first) == lifecycle(second)


def test_a_pause_holds_the_clock_until_a_resume(tmp_path: Path) -> None:
    at = 2.5 * ROUND
    inbox(tmp_path, signed({"pause": True, "at": at}))
    seen = []

    def sleep(_):
        seen.append(read_state(tmp_path))
        if len(seen) == 3:  # the operator looks, rolls back and resumes
            inbox(tmp_path, signed({"rollback": None}), signed({"resume": True}))

    fed = live(tmp_path, sleep=sleep)

    assert all(s["paused"] and s["t"] == at for s in seen)
    names = [e["name"] for e in fed.runtime.events if e["name"].startswith("control.")]
    assert names == ["control.received", "control.paused"] + [
        "control.received",
        "control.received",
        "control.resumed",
    ]
    (rollback,) = events(fed, "version.rollback")
    assert rollback["t"] == at and fed.coordinator.finished
    # handled while paused, so the Studio sees it before the run goes on
    order = [e["name"] for e in fed.runtime.events]
    assert order.index("version.rollback") < order.index("control.resumed")


def test_a_run_can_start_paused(tmp_path: Path) -> None:
    calls = []

    def sleep(_):
        calls.append(read_state(tmp_path))
        inbox(tmp_path, signed({"resume": True}))

    fed = live(tmp_path, sleep=sleep, paused=True)

    assert calls[0]["paused"] and calls[0]["t"] == 0.0
    assert fed.coordinator.finished and read_state(tmp_path)["listening"] is False


def test_forged_and_broken_lines_never_reach_the_coordinator(tmp_path: Path) -> None:
    good = signed({"rollback": None, "at": 4.5 * ROUND})
    inbox(tmp_path, {"rollback": None}, signed({"pause": True}, audience="run-2"))
    inbox(tmp_path, signed({"pause": True}), operator="mallory")
    inbox(tmp_path, good, good)
    with (tmp_path / INBOX).open("a", encoding="utf-8") as out:
        out.write("not json\n")
    fed = live(tmp_path)

    reasons = [e["tags"]["reason"] for e in events(fed, "control.refused")]
    assert reasons == [
        "unsigned order",
        "bad signature",
        "mallory is not an operator",
        "replayed order",
        "not an order",
    ]
    assert not events(fed, "control.paused") and not rejected(fed)
    assert len(events(fed, "version.rollback")) == 1
    assert [o["order"] for o in read_inbox(tmp_path)][0] == {"rollback": None}
    assert all("auth" not in o["order"] for o in read_inbox(tmp_path))


def test_an_order_whose_time_passed_applies_at_once(tmp_path: Path) -> None:
    fed = federation()
    fed.runtime.run(until=4.5 * ROUND)
    inbox(tmp_path, signed({"rollback": None, "at": 1.0}))
    LiveControl(tmp_path, KEYS, RUN, "cloud", poll=0.0).run(fed.runtime)

    (rollback,) = events(fed, "version.rollback")
    assert rollback["t"] >= 4.5 * ROUND


def test_an_external_message_to_an_unknown_node_is_dropped() -> None:
    fed = federation()
    fed.runtime.inject(Message(kind="control", src="alice", dst="nobody", meta={}))
    fed.run()

    (dropped,) = [
        e for e in events(fed, "link.dropped") if e["tags"].get("dst") == "nobody"
    ]
    assert dropped["node"] == "alice"
