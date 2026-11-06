# ruff: noqa: F811 (the workspace fixture is imported)
"""Tests for the random search and the noise ceiling (issue #210).

The dataset is the tiny CSV format fixture of ``test_experiment``; edges use
the stub trainer with seeded noise and the score is the mean weight of the
model, so nothing is trained or evaluated for real (docs/RULES.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from onion_fl.experiment.ceiling import ceiling, replicas, run_ceiling
from onion_fl.experiment.config import ConfigError, parse_experiment
from onion_fl.experiment.sweep import draw, scenarios
from tests.test_experiment import experiment, workspace  # noqa: F401

NOISY = {
    "model": {"name": "modular_mlp", "adapter_width": 4, "trunk_hidden": [4]},
    "trainer": {"name": "stub", "noise": 0.1},
}
SEARCH = {
    "budget": 4,
    "seed": 7,
    "space": {
        "learning.trainer.shift": [0.0, 1.0, 2.0],
        "learning.trainer.noise": {"low": 0.01, "high": 0.1, "log": True},
    },
}


def mean_weight(model, data):
    """A score that only the seeded noise moves: the mean of every weight."""
    weights = [p.detach().numpy().ravel() for p in model.parameters()]
    return {"accuracy": float(np.concatenate(weights).mean())}, data.n_samples


def same_configs(workspace: Path, k: int, **overrides) -> dict:
    """``k`` configurations that differ only in their description."""
    sweep = {"description": [f"copy {i}" for i in range(k)]}
    return experiment(workspace, learning=NOISY, rounds=1, sweep=sweep, **overrides)


# --- random search -----------------------------------------------------------------------------


def test_a_search_turns_each_swept_scenario_into_its_budget(workspace: Path) -> None:
    raw = experiment(workspace, search=SEARCH, sweep={"rounds": [1, 2]})

    every = scenarios(parse_experiment(raw))

    assert len(every) == 2 * 4
    assert every[0].name == "rounds=1,search.trial=0"
    assert {s.configuration for s in every} == {"rounds=1", "rounds=2"}
    trials = draw(parse_experiment(raw).search)
    for scenario in every:
        trial = trials[scenario.config.search.trial]
        trainer = scenario.config.learning.trainer
        assert trainer == {"name": "stub", **{k[17:]: v for k, v in trial.items()}}
        assert 0.01 <= trainer["noise"] <= 0.1 and trainer["shift"] in (0, 1, 2)
    assert len({s.config_id for s in every}) == 8


def test_the_draws_come_from_the_search_seed_alone(workspace: Path) -> None:
    config = parse_experiment(experiment(workspace, search=SEARCH))
    other = parse_experiment(experiment(workspace, search=SEARCH | {"seed": 8}))

    assert draw(config.search) == draw(parse_experiment(config.dump()).search)
    assert draw(config.search) != draw(other.search)
    # the search seed is part of what a trial is, even for the same values
    ids = {s.config_id for s in scenarios(config)}
    assert ids.isdisjoint(s.config_id for s in scenarios(other))


def test_a_scenario_config_is_one_trial_and_does_not_expand_again(
    workspace: Path,
) -> None:
    scenario = scenarios(parse_experiment(experiment(workspace, search=SEARCH)))[2]

    (again,) = scenarios(scenario.config)

    assert again.config_id == scenario.config_id
    assert again.config.learning.trainer == scenario.config.learning.trainer


def test_no_search_keeps_the_config_ids(workspace: Path) -> None:
    config = parse_experiment(experiment(workspace))

    assert "search" not in config.dump()
    assert scenarios(config)[0].configuration == "base"


@pytest.mark.parametrize(
    "search, sweep, message",
    [
        (SEARCH, {"learning.trainer": ["stub"]}, "overlaps the swept"),
        (SEARCH, {"learning.trainer.shift": [1.0]}, "overlaps the swept"),
        (SEARCH | {"space": {"seeds": [[1]]}}, {}, "cannot be searched"),
        (SEARCH | {"space": {"rounds": []}}, {}, "at least one value"),
        (SEARCH | {"space": {"rounds": {"low": 2, "high": 1}}}, {}, "above low"),
        (SEARCH | {"space": {"x": {"low": 0, "high": 1, "log": True}}}, {}, "low > 0"),
        (SEARCH | {"trial": 4}, {}, "outside a budget"),
    ],
)
def test_bad_searches_are_refused(
    workspace: Path, search: dict, sweep: dict, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_experiment(experiment(workspace, search=search, sweep=sweep))


# --- the statistics ------------------------------------------------------------------------------


def test_top1_and_regret_between_replicas() -> None:
    scores = [[0.9, 0.8, 0.1], [0.7, 0.8, 0.2], [0.9, 0.5, 0.3]]

    out = ceiling(scores)

    assert out["winners"] == [0, 1, 0] and out["pairs"] == 6
    assert out["top1"] == pytest.approx(2 / 6) and out["top1_chance"] == 1 / 3
    # (a, b) -> best of b minus b's score of a's winner
    regrets = [0.1, 0.0, 0.1, 0.4, 0.1, 0.0]  # (0,1) (0,2) (1,0) (1,2) (2,0) (2,1)
    assert out["regret"]["mean"] == pytest.approx(np.mean(regrets))
    assert out["regret"]["max"] == pytest.approx(0.4)


def test_lower_is_better_flips_the_winner() -> None:
    out = ceiling([[1.0, 2.0], [3.0, 2.5]], minimize=True)

    assert out["winners"] == [0, 1] and out["top1"] == 0
    assert out["regret"]["mean"] == pytest.approx((0.5 + 1.0) / 2)


def test_one_replica_is_not_a_ceiling(workspace: Path) -> None:
    with pytest.raises(ValueError, match="R >= 2"):
        ceiling([[1.0, 2.0]])
    with pytest.raises(ConfigError, match="at least 2"):
        replicas(parse_experiment(same_configs(workspace, 2)), 1)


# --- replicas ------------------------------------------------------------------------------------


def test_replica_seeds_are_independent_unless_paired(workspace: Path) -> None:
    config = parse_experiment(same_configs(workspace, 3, seeds=[10]))

    unpaired = {
        (t.replica, t.configuration): t.scenario.seed for t in replicas(config, 2)
    }
    paired = replicas(config, 2, paired=True)

    assert sorted(unpaired.values()) == list(range(10, 16))
    assert {(t.replica, t.scenario.seed) for t in paired} == {(0, 10), (1, 11)}


def test_resample_draws_the_search_anew_per_replica(workspace: Path) -> None:
    config = parse_experiment(same_configs(workspace, 1, search=SEARCH))

    kept = replicas(config, 2)
    fresh = replicas(config, 2, resample=True)

    def trials(items, r):
        return [t.scenario.config.learning.trainer for t in items if t.replica == r]

    assert len(kept) == len(fresh) == 2 * 4
    assert trials(kept, 0) == trials(kept, 1)
    assert trials(fresh, 0) != trials(fresh, 1)
    with pytest.raises(ConfigError, match="has none"):
        replicas(parse_experiment(same_configs(workspace, 2)), 2, resample=True)


# --- the acceptance test -------------------------------------------------------------------------


def _null_top1(replicas: int, k: int, draws: int = 20_000) -> np.ndarray:
    """Top-1 agreement of ``replicas`` uniform winners among ``k``, many times over."""
    counts = np.random.default_rng(0).multinomial(replicas, [1 / k] * k, size=draws)
    return ((counts**2).sum(axis=1) - replicas) / (replicas * (replicas - 1))


def test_identical_configs_agree_on_top1_as_by_chance(workspace: Path) -> None:
    k, n = 3, 12
    config = parse_experiment(same_configs(workspace, k))

    out = run_ceiling(config, n=n, evaluate=mean_weight)

    # Under the null each replica's winner is uniform among the k: the top-1
    # agreement falls inside the null's 99 % range (its mean is 1/k), and the
    # winners pass a chi-square test of uniformity.
    null = _null_top1(n, k)
    low, high = np.quantile(null, [0.005, 0.995])
    assert out["top1_chance"] == pytest.approx(null.mean(), abs=0.01)
    assert low <= out["top1"] <= high
    counts = np.bincount(out["winners"], minlength=k)
    assert stats.chisquare(counts).pvalue > 0.01
    assert len(out["scores"]) == n and all(len(row) == k for row in out["scores"])
    assert out["regret"]["mean"] > 0
    runs = {cell["run_id"] for row in out["cells"] for cell in row.values()}
    assert len(runs) == n * k  # one signed run per cell
    assert all((workspace / "runs" / r / "run.json").exists() for r in runs)


def test_paired_identical_configs_tie_and_the_first_wins(workspace: Path) -> None:
    config = parse_experiment(same_configs(workspace, 2))

    out = run_ceiling(config, n=2, paired=True, evaluate=mean_weight)

    assert all(row[0] == row[1] for row in out["scores"])
    assert out["winners"] == [0, 0] and out["top1"] == 1.0


def test_a_search_scores_the_best_trial_of_each_cell(workspace: Path) -> None:
    space = {"learning.trainer.shift": [0.0, 1.0, 2.0]}
    search = {"budget": 3, "seed": 1, "space": space}
    config = parse_experiment(same_configs(workspace, 2, search=search))

    out = run_ceiling(config, n=2, resample=True, evaluate=mean_weight)

    for row in out["cells"]:
        for cell in row.values():
            meta = json.loads(
                (workspace / "runs" / cell["run_id"] / "run.json").read_text(
                    encoding="utf-8"
                )
            )
            assert len(cell["trials"]) == 3 and meta["config_id"] == cell["config_id"]
            assert cell["score"] == max(cell["trials"])
    # the score grows with the shift, so each cell keeps its largest drawn shift
    todo = replicas(config, 2, resample=True)
    shift = {
        t.scenario.config_id: t.scenario.config.learning.trainer["shift"] for t in todo
    }
    for r, row in enumerate(out["cells"]):
        for name, cell in row.items():
            drawn = [
                shift[t.scenario.config_id]
                for t in todo
                if (t.replica, t.configuration) == (r, name)
            ]
            assert shift[cell["config_id"]] == max(drawn)
