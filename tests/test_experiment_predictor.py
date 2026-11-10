"""Tests for the factorised predictor (issue #211, spec N7 and I1 §12).

The models and the tree are checked against values worked out by hand. The
runs use the stub trainer on a CSV format fixture and a stub scorer that reads
the stub's weights, so nothing is trained or evaluated for real
(docs/RULES.md): the stub adds ``shift`` to every weight per round, so the
scorer reaches a target after a known number of rounds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from onion_fl.experiment.cli import main
from onion_fl.experiment.config import parse_experiment
from onion_fl.experiment.fidelity import design_id, fidelity, timings, write_html
from onion_fl.experiment.predictor import (
    Plan,
    control_id,
    has_deadline,
    link_time,
    mean_duration,
    plan_of,
    predict,
    rounds_to_target,
    within,
)
from onion_fl.experiment.runner import run_experiment
from onion_fl.observability.events import read_events
from onion_fl.runtime.devices import LognormalRate, Measured, SamplesPerSecond, Trace
from onion_fl.runtime.network import LinkChannel, LinkProfile
from tests.test_experiment import TOPOLOGY, experiment, workspace  # noqa: F401

# --- the models ----------------------------------------------------------------------


def test_the_mean_duration_of_each_compute_model() -> None:
    assert mean_duration(SamplesPerSecond(10, overhead_s=0.5), 20) == 2.5
    assert mean_duration(LognormalRate(10, sigma=0.8), 20) == 2.0  # the same mean
    assert mean_duration(Trace([1.0, 2.0, 6.0]), 20) == 3.0
    assert mean_duration(None, 20) == 0.0
    assert mean_duration(SamplesPerSecond(10), 0) == 0.0  # no work, no time
    assert mean_duration(Measured(), 20) is None  # this machine's wall clock


def test_a_deterministic_model_fits_a_budget_or_not() -> None:
    model = SamplesPerSecond(10, overhead_s=0.5)

    assert within(model, 20, 2.5) == 1.0 and within(model, 20, 2.4) == 0.0
    assert within(model, 20, 0.4) == 0.0  # not even its overhead
    assert within(Trace([1.0, 2.0, 6.0]), 20, 2.0) == pytest.approx(2 / 3)


def test_a_lognormal_rate_fits_a_budget_with_its_analytic_probability() -> None:
    model = LognormalRate(10, sigma=0.5, overhead_s=0.2)
    rng = np.random.default_rng(0)
    draws = np.array([model.duration(20, 0.0, rng=rng) for _ in range(20000)])

    # At the mean the factor must be <= 1: P = Phi(sigma / 2).
    assert within(model, 20, 2.2) == pytest.approx(stats.norm.cdf(0.25))
    for budget in (1.5, 2.2, 3.5):
        assert within(model, 20, budget) == pytest.approx(
            float(np.mean(draws <= budget)), abs=0.01
        )


def test_a_link_takes_its_transmission_plus_its_mean_latency() -> None:
    profile = LinkProfile(latency_s=0.1, jitter_s=0.05, bandwidth_down_bps=800)
    rng = np.random.default_rng(0)

    assert link_time(LinkChannel(profile, "down", rng), 100) == pytest.approx(1.1)
    assert link_time(LinkChannel(profile, "up", rng), 100) == pytest.approx(0.1)


# --- the tree, by hand -----------------------------------------------------------------


def _star() -> Plan:
    """A root ``c`` with a fast edge ``a`` and a slow edge ``b``: every link takes
    1.1 s for 100 bytes; ``a`` trains in 1 s and ``b`` in 5 s."""
    profile = LinkProfile(latency_s=0.1, bandwidth_up_bps=800, bandwidth_down_bps=800)
    rng = np.random.default_rng(0)
    return Plan(
        compute={"c": None, "a": SamplesPerSecond(10), "b": SamplesPerSecond(2)},
        channels={
            (src, dst): LinkChannel(profile, "down" if src == "c" else "up", rng)
            for src, dst in (("c", "a"), ("a", "c"), ("c", "b"), ("b", "c"))
        },
        deadlines={"c": 5.0},
        combine={"c": 0.0},
    )


def _events(rounds: int, scores: list[float]) -> list[dict]:
    """The events of a star run of ``rounds`` rounds, written by hand."""
    out = []

    def emit(node, role, name, value=None, **tags):
        out.append(
            {
                "node": node,
                "role": role,
                "name": name,
                "value": value,
                "round": tags.get("round"),
                "tags": tags,
                "t_virtual": 0.0,
            }
        )

    for r in range(1, rounds + 1):
        emit("c", "coordinator", "round.started", r, round=r)
        for edge, examples in (("a", 30), ("b", 10)):
            emit(
                "c",
                "coordinator",
                "message.sent",
                100,
                dst=edge,
                round=r,
                kind="global_model",
            )
            # outside rounds: not counted
            emit(
                "c",
                "coordinator",
                "message.sent",
                50,
                dst=edge,
                round=None,
                kind="control",
            )
            emit(
                edge,
                "edge",
                "edge.trained",
                0.0,
                round=r,
                examples=examples,
                samples=10,
            )
            emit(edge, "edge", "message.sent", 100, dst="c", round=r, kind="update")
        emit("c", "coordinator", "eval.accuracy", scores[r - 1], round=r)
    return out


def test_a_round_lasts_its_slowest_path_plus_aggregation() -> None:
    out = predict(_star(), _events(3, [0.1, 0.9, 0.95]), ("accuracy", 0.8))

    plain = out["factorised"]
    assert plain["round_times"] == pytest.approx([7.2, 7.2, 7.2])  # b: 1.1+5+1.1
    assert plain["time"] == pytest.approx(21.6)
    assert plain["rounds_to_target"] == 2 and not plain["censored"]
    assert plain["time_to_target"] == pytest.approx(14.4)  # 2 rounds x 7.2 s
    assert plain["participation"] == 1.0


def test_the_floss_variant_corrects_for_who_misses_the_deadline() -> None:
    out = predict(_star(), _events(3, [0.1, 0.9, 0.95]), ("accuracy", 0.8))

    floss = out["floss"]
    # b needs 5 s and has 5 - 2.2: it never counts; the round waits for 5 s.
    assert floss["round_times"] == pytest.approx([5.0, 5.0, 5.0])
    assert floss["participation"] == pytest.approx(0.75)  # a's 30 of 40 examples
    assert floss["rounds_to_target"] == pytest.approx(2 / 0.75)
    assert floss["time_to_target"] == pytest.approx(2 / 0.75 * 5.0)


def test_a_prediction_past_the_run_is_censored_at_its_duration() -> None:
    out = predict(_star(), _events(2, [0.1, 0.9]), ("accuracy", 0.8))

    floss = out["floss"]  # 2.67 rounds of 5 s in a run of 2 rounds
    assert floss["censored"] and floss["time_to_target"] == pytest.approx(10.0)


def test_rounds_to_target_are_censored_when_never_reached() -> None:
    events = _events(3, [0.1, 0.2, 0.3])

    assert rounds_to_target(events, "accuracy", 0.8) == (3, True)
    assert rounds_to_target(events, "accuracy", 0.2) == (2, False)
    losses = [
        e | {"name": "eval.loss"} if e["name"] == "eval.accuracy" else e for e in events
    ]
    assert rounds_to_target(losses, "loss", 0.15) == (1, False)  # lower is better


# --- on runs ---------------------------------------------------------------------------

FIXED = {"latency_s": 0.01, "bandwidth_up_bps": 1e5, "bandwidth_down_bps": 2e5}


def weights_score(model, data):
    """A stub scorer: the mean weight, which the stub raises by ``shift`` a round."""
    weights = [p.detach().numpy().mean() for p in model.parameters()]
    return {"accuracy": float(np.mean(weights))}, data.n_samples


def _topology(sps: float = 10, deadline: str | None = None) -> dict:
    fog = TOPOLOGY["fog"] | {"defaults": {"link_up": {"profile": FIXED}}}
    if deadline is not None:
        fog["defaults"] = fog["defaults"] | {"deadline": deadline}
    edge = {"device": {"samples_per_second": sps}, "link_up": {"profile": FIXED}}
    return TOPOLOGY | {"fog": fog, "edge": edge}


def _run(workspace: Path, folder: str, **changes) -> Path:  # noqa: F811
    raw = experiment(workspace, **changes)
    raw["paths"] = raw["paths"] | {"runs": str(workspace / folder)}
    run_experiment(parse_experiment(raw), evaluate=weights_score)
    return workspace / folder


def test_past_the_first_the_predicted_round_is_the_simulated_one(
    workspace: Path,  # noqa: F811
) -> None:
    # Fixed links and a deterministic compute model: the closed form is exact
    # but for the acknowledgements still queued on the links of round 1.
    runs = _run(workspace, "runs", topology=_topology(), rounds=3)
    path = next(runs.glob("*/run.json")).parent
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    events = read_events(path / "events.jsonl")

    plan = plan_of(meta["config"], meta["seed"], meta["scenario"])
    predicted = predict(plan, events)["factorised"]["round_times"]
    simulated = timings(events)["round_time"]

    assert predicted[1:] == pytest.approx([simulated[2], simulated[3]], rel=1e-9)
    assert predicted[0] == pytest.approx(simulated[1], abs=0.01)
    assert plan.unknown == []


def test_the_control_drops_deadlines_and_coupling(workspace: Path) -> None:  # noqa: F811
    base = parse_experiment(experiment(workspace, topology=_topology())).dump()
    timed = parse_experiment(
        experiment(workspace, topology=_topology(deadline="3s"))
    ).dump()

    assert has_deadline(timed) and not has_deadline(base)
    assert design_id(base) != design_id(timed)
    assert control_id(base) == control_id(timed)
    assert control_id(base) != control_id(base | {"rounds": 9})


# Three designs, the stub's weights rising by ``shift`` a round on devices of
# ``samples_per_second`` (10 examples a round): to reach 2.7 the first needs 3
# rounds of ~2 s, the second 6 of ~0.5 s (the fastest) and the third 2 of ~5 s.
DESIGNS = [[1.0, 5.0], [0.5, 20.0], [2.0, 2.0]]
SWEEP = {"learning.trainer.shift,topology.edge.device.samples_per_second": DESIGNS}
TARGET = ("accuracy", 2.7)


def _control(workspace: Path, folder: str = "control", **changes) -> Path:  # noqa: F811
    return _run(
        workspace,
        folder,
        topology=changes.pop("topology", _topology()),
        rounds=8,
        seeds=[0, 1],
        sweep=SWEEP,
        **changes,
    )


def test_in_sync_without_deadline_the_predictor_picks_the_winner_of_sim(
    workspace: Path,  # noqa: F811
) -> None:
    runs = _control(workspace)

    result = fidelity(
        runs,
        "predictor",
        metrics=["time_to_target", "time", "accuracy"],
        target=TARGET,
        resamples=200,
    )

    assert result["functional"] is None and result["pairs"] == []
    assert result["backends"] == {"a": ["sim"], "b": ["predictor"]}
    assert len(result["configurations"]) == 3 and result["seeds"] == [0, 1]
    for metric in ("time_to_target", "time"):
        decided = result["decision"][metric]
        assert decided["top1"] and decided["regret"] == 0.0, metric
        assert decided["tau_b"] == 1.0 and decided["minimize"]
    assert "accuracy" not in result["decision"]
    assert any("scores time and time_to_target only" in w for w in result["warnings"])
    rounds = {
        p["factorised"]["rounds_to_target"] for p in result["predictions"]
    }  # 3, 6 and 2 rounds, as the stub's shifts give
    assert rounds == {2, 3, 6}


def test_the_predictor_can_be_named_first_and_floss_chosen(
    workspace: Path,  # noqa: F811
) -> None:
    runs = _control(workspace)

    plain = fidelity("predictor", runs, metrics=["time"], resamples=50)
    floss = fidelity(runs, "predictor:floss", metrics=["time"], resamples=50)

    assert plain["a"] == str(runs) and plain["b"] == "predictor"
    # Without deadlines both variants are the same model.
    assert floss["decision"]["time"]["scores"] == plain["decision"]["time"]["scores"]
    with pytest.raises(ValueError, match="not with itself"):
        fidelity("predictor", "predictor:floss")


def test_a_comparison_of_backends_shows_the_predictor_next_to_it(
    workspace: Path,  # noqa: F811
) -> None:
    a, b = _control(workspace, "runs_a"), _control(workspace, "runs_b")

    result = fidelity(
        a, b, metrics=["time_to_target", "bytes"], target=TARGET, resamples=50
    )

    slot = result["decision"]["time_to_target"]["predictor"]
    assert set(slot) == {"factorised", "floss"}
    assert slot["factorised"]["top1"] and slot["factorised"]["regret"] == 0.0
    assert slot["factorised"]["b_regret_below"] is False  # B ties it: no better
    assert result["decision"]["bytes"]["predictor"] is None  # not a time metric
    page = write_html(result, workspace / "f.html").read_text(encoding="utf-8")
    assert "FLOSS regret" in page
    assert not any(w.startswith("predictor:") for w in result["warnings"])

    off = fidelity(a, b, metrics=["time"], resamples=50, predictor=False)
    assert off["decision"]["time"]["predictor"] is None


def test_a_truth_with_deadlines_takes_its_rounds_from_the_control(
    workspace: Path,  # noqa: F811
) -> None:
    # The truth closes the fog rounds at 1.5 s: the edges of the slow designs
    # (2 s and 5 s of compute) never make it, the fast one (0.5 s) always does.
    control = _control(workspace)
    truth = _control(workspace, "truth", topology=_topology(deadline="1.5s"))

    result = fidelity(
        truth,
        "predictor:floss",
        metrics=["time"],
        control=control,
        resamples=50,
    )

    assert not any("has no control run" in w for w in result["warnings"])
    shares = sorted(p["floss"]["participation"] for p in result["predictions"])
    assert shares[:4] == [0.0] * 4 and shares[4:] == [1.0, 1.0]
    plain = fidelity(truth, "predictor", metrics=["time"], resamples=50)
    assert any("have deadlines" in w for w in plain["warnings"])


def test_the_command_line_takes_the_predictor_as_a_side(
    capsys,
    workspace: Path,  # noqa: F811
) -> None:
    runs = _control(workspace)

    code = main(
        ["fidelity", "predictor", str(runs), "--metric", "time", "--bootstrap", "20"]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["b"] == "predictor"
    assert out["decision"]["time"]["top1"]
