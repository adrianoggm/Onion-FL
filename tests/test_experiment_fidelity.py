"""Tests for the fidelity harness (issue #191, spec I1 §12).

The decision metrics are checked on rankings written by hand. The runs use the
stub trainer on a CSV format fixture and a stub scorer, so nothing is trained
or evaluated for real (docs/RULES.md).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from onion_fl.experiment.cli import main
from onion_fl.experiment.config import parse_experiment
from onion_fl.experiment.fidelity import (
    decision,
    design_id,
    fidelity,
    kendall_tau_b,
    ranking_agreement,
    selection_regret,
    ties,
    write_csv,
    write_html,
)
from onion_fl.experiment.runner import run_experiment
from tests.test_experiment import experiment, stub_score, workspace  # noqa: F401

# --- decision metrics on hand-written rankings --------------------------------------


def test_identical_rankings_agree_fully() -> None:
    result = ranking_agreement([0.1, 0.5, 0.9], [0.1, 0.5, 0.9])

    assert result == {"tau_b": 1.0, "top1": True, "regret": 0.0, "regret_relative": 0.0}


def test_reversed_rankings_disagree_fully() -> None:
    result = ranking_agreement([0.1, 0.5, 0.9], [0.9, 0.5, 0.1])

    assert result["tau_b"] == pytest.approx(-1.0) and not result["top1"]
    assert result["regret"] == pytest.approx(0.8)
    assert result["regret_relative"] == pytest.approx(0.8 / 0.9)


def test_ties_use_tau_b() -> None:
    # 4 concordant pairs, no discordant ones, 2 pairs tied in b only:
    # tau-b = 4 / sqrt(6 * 4), not tau-a = 4 / 6.
    assert kendall_tau_b([1, 2, 3, 4], [1, 1, 2, 2]) == pytest.approx(4 / math.sqrt(24))


def test_a_known_selection_regret() -> None:
    # B picks the first configuration; in A it scores 0.70, and A's best is 0.80.
    regret, relative = selection_regret([0.70, 0.80, 0.75], [0.9, 0.1, 0.2])

    assert regret == pytest.approx(0.10) and relative == pytest.approx(0.125)


def test_the_regret_of_a_lower_is_better_metric() -> None:
    # B's fastest is the second (5); in A it takes 20 and A's fastest takes 10.
    regret, relative = selection_regret([10, 20, 30], [25, 5, 40], minimize=True)

    assert regret == pytest.approx(10) and relative == pytest.approx(1.0)


def test_a_constant_ranking_has_no_tau_and_ties() -> None:
    assert kendall_tau_b([0.5, 0.5, 0.5], [0.1, 0.2, 0.3]) is None
    assert ties([0.5, 0.5, 0.5]) and not ties([0.5, 0.6])


def test_the_bootstrap_is_paired_by_seed() -> None:
    # Seed 0 ranks the configurations one way and seed 1 the other way, in
    # both tables: resampling the same seeds keeps A and B in agreement.
    a = np.array([[0.9, 0.1], [0.1, 0.8]])
    b = a + 0.01

    result = decision(a, b, resamples=500, seed=3)

    assert result["top1"] and result["p_top1"] == 1.0
    assert result["ci"]["regret"] == [0.0, 0.0]
    assert result["seeds"] == 2 and result["configurations"] == 2


def test_the_bootstrap_probability_of_top1_sees_a_close_call() -> None:
    # B agrees with A on two seeds and swaps the third, which decides A's
    # winner only in some resamples.
    a = np.array([[0.9, 0.1], [0.1, 0.8], [0.5, 0.4]])
    b = np.array([[0.9, 0.1], [0.1, 0.8], [0.4, 0.5]])

    result = decision(a, b, resamples=1000, seed=0)

    assert result["top1"] and 0.5 < result["p_top1"] < 1.0
    low, high = result["ci"]["regret"]
    assert low == 0.0 and high == pytest.approx(0.1)


# --- the harness on runs -------------------------------------------------------------


def _runs(workspace: Path, folder: str, **changes) -> Path:  # noqa: F811
    raw = experiment(
        workspace,
        seeds=[0, 1],
        sweep={"rounds": [1, 2]},
        **changes,
    )
    raw["paths"] = raw["paths"] | {"runs": str(workspace / folder)}
    run_experiment(parse_experiment(raw), evaluate=stub_score)
    return workspace / folder


def test_the_design_id_ignores_where_and_how_a_config_ran(
    workspace: Path,  # noqa: F811
) -> None:
    base = parse_experiment(experiment(workspace)).dump()
    moved = base | {"paths": base["paths"] | {"runs": "elsewhere"}, "backend": "local"}
    real = base | {"runtime": base["runtime"] | {"mode": "real", "timeout": 5}}

    assert design_id(base) == design_id(moved) == design_id(real)
    assert design_id(base) != design_id(base | {"rounds": 9})


def test_the_design_id_ignores_the_transport_of_the_links() -> None:
    def config(transport, profile="lan"):
        link = {"transport": transport, "profile": profile}
        return {
            "name": "x",
            "topology": {
                "fog": {"defaults": {"link_up": link}},
                "edge": {"link_up": link},
            },
        }

    mqtt = config({"name": "mqtt", "broker": "localhost:1883"})

    assert design_id(mqtt) == design_id(config("grpc"))
    assert design_id(mqtt) != design_id(config("grpc", profile="wan"))


def test_a_backend_against_itself_has_complete_fidelity(
    workspace: Path,  # noqa: F811
) -> None:
    a, b = _runs(workspace, "runs_a"), _runs(workspace, "runs_b")

    result = fidelity(a, b, metrics=["time", "bytes"], resamples=200)

    assert result["functional"] == {
        "pairs": 4,
        "complete": True,
        "model": True,
        "protocol": True,
        "bytes": True,
    }
    assert result["backends"] == {"a": ["sim"], "b": ["sim"]}
    assert result["seeds"] == [0, 1] and len(result["configurations"]) == 2
    for perf in result["performance"].values():
        assert perf["relative_error"] == 0.0
    bytes_ = result["decision"]["bytes"]
    assert bytes_["tau_b"] == 1.0 and bytes_["top1"] and bytes_["regret"] == 0.0
    assert bytes_["ceiling"]["source"] == "seeds of A"
    assert bytes_["predictor"] is None
    assert any("at least 5" in w for w in result["warnings"])


def test_configurations_that_tie_on_a_metric_are_flagged(
    workspace: Path,  # noqa: F811
) -> None:
    a, b = _runs(workspace, "runs_a"), _runs(workspace, "runs_b")

    result = fidelity(a, b, metrics=["accuracy"], resamples=100)  # stub: always 0.5

    assert any("ties on accuracy in A" in w for w in result["warnings"])
    assert result["decision"]["accuracy"]["tau_b"] is None


def test_a_given_noise_ceiling_is_shown_next_to_the_decision(
    workspace: Path,  # noqa: F811
) -> None:
    a, b = _runs(workspace, "runs_a"), _runs(workspace, "runs_b")
    noise = {
        "metric": "bytes",
        "top1": 0.75,
        "top1_chance": 0.5,
        "regret": {"mean": 3.0, "relative_mean": 0.01},
    }

    result = fidelity(a, b, metrics=["bytes"], noise_ceiling=noise, resamples=50)

    decided = result["decision"]["bytes"]
    assert decided["ceiling"]["source"] == "ceiling"
    assert decided["ceiling"]["top1"] == 0.75
    assert decided["top1_below_ceiling"] is False
    assert decided["regret_above_ceiling"] is False


def test_runs_of_different_experiments_do_not_pair(
    workspace: Path,  # noqa: F811
) -> None:
    a = _runs(workspace, "runs_a")
    b = _runs(workspace, "runs_b", name="other")

    with pytest.raises(ValueError, match="shares a configuration and seed"):
        fidelity(a, b)


def test_the_report_goes_to_csv_and_html(workspace: Path) -> None:  # noqa: F811
    a, b = _runs(workspace, "runs_a"), _runs(workspace, "runs_b")
    result = fidelity(a, b, metrics=["time"], resamples=50)

    table = list(
        csv.DictReader(
            write_csv(result, workspace / "f.csv").open(encoding="utf-8", newline="")
        )
    )
    page = write_html(result, workspace / "f.html").read_text(encoding="utf-8")

    levels = {row["level"] for row in table}
    assert levels == {"functional", "performance", "decision"}
    regret = [r for r in table if r["metric"] == "time" and r["statistic"] == "regret"]
    assert regret and regret[0]["ceiling"] != ""
    assert "Functional" in page and "predictor regret" in page
    assert result["pairs"][0]["a"]["run_hash"] in page


def test_the_command_line_compares_two_folders(
    capsys,
    workspace: Path,  # noqa: F811
) -> None:
    a, b = _runs(workspace, "runs_a"), _runs(workspace, "runs_b")

    code = main(
        [
            "fidelity",
            str(a),
            str(b),
            "--metric",
            "time",
            "--bootstrap",
            "50",
            "--csv",
            str(workspace / "f.csv"),
            "--html",
            str(workspace / "f.html"),
        ]
    )

    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["functional"]["complete"]
    assert (workspace / "f.csv").exists() and (workspace / "f.html").exists()


def test_a_target_needs_metric_and_value(capsys, workspace: Path) -> None:  # noqa: F811
    a, b = _runs(workspace, "runs_a"), _runs(workspace, "runs_b")

    code = main(["fidelity", str(a), str(b), "--target", "0.8"])

    assert code == 2 and "metric=value" in capsys.readouterr().err
