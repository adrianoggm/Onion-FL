from __future__ import annotations

"""Noise ceiling: how often replicas of one study pick the same winner (spec N6).

``onion_fl ceiling <experiment>`` runs ``R`` replicas of every configuration
(every swept scenario). A replica is a whole new run of the study:

- a new training seed, by default independent for every (configuration,
  replica) pair; ``paired`` gives every configuration of a replica the same
  seed, as the ``seeds`` of a sweep do;
- with a ``search``, its trials run inside the replica and the configuration
  scores its best trial; ``resample`` also draws the search anew, from the
  search seed plus the same offset as the training seed.

From the table of scores (replica × configuration) it gives the top-1
agreement and the selection regret **between replicas**, the same quantities
the fidelity harness compares backends with (I1 §12): over every ordered pair
of replicas ``(a, b)``, whether they pick the same winner, and the score in
``b`` of the winner of ``a`` minus the best score in ``b``. With ``K``
configurations that do not differ, the top-1 agreement expected by chance is
``1/K``. A decision metric only counts where it beats this ceiling.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from onion_fl.experiment.config import ConfigError, ExperimentConfig, parse_experiment
from onion_fl.experiment.sweep import Scenario, scenarios

LOWER_IS_BETTER = ("loss",)


@dataclass(frozen=True)
class Replica:
    configuration: str
    replica: int
    scenario: Scenario


def replicas(
    config: ExperimentConfig,
    n: int,
    resample: bool = False,
    paired: bool = False,
) -> list[Replica]:
    """Every scenario of ``n`` replicas of every configuration, with its seed.

    The seed of replica ``r`` of configuration ``k`` (of ``K``) is the first
    of ``seeds`` plus ``r`` when paired, plus ``r * K + k`` otherwise.
    """
    if n < 2:
        raise ConfigError("ceiling: it needs at least 2 replicas to compare")
    if resample and config.search is None:
        raise ConfigError("ceiling: resample redraws a search; the config has none")
    names = list(dict.fromkeys(s.configuration for s in scenarios(config)))
    first, out = config.seeds[0], []
    for r in range(n):
        for k, name in enumerate(names):
            offset = r if paired else r * len(names) + k
            raw = config.dump() | {"seeds": [first + offset]}
            if resample:
                raw["search"] = raw["search"] | {"seed": config.search.seed + offset}
            out += [
                Replica(name, r, s)
                for s in scenarios(parse_experiment(raw))
                if s.configuration == name
            ]
    return out


def final_score(run: Path, metric: str, series: str | None = None) -> float:
    """The last ``metric`` scored at the root of a run, from its ``summary.json``."""
    final = json.loads((run / "summary.json").read_text(encoding="utf-8"))["final"]
    keys = sorted(k for k, scores in final.items() if metric in scores)
    if series is not None:
        keys = [k for k in keys if k == series]
    elif len(keys) > 1:  # the root model over the pooled test samples
        keys = [k for k in keys if k.endswith("/*")] or keys
    if len(keys) != 1:
        what = f"no series {series!r} with" if series else f"{len(keys)} series with"
        raise ValueError(
            f"{run.name}: {what} {metric!r} at the root ({sorted(final)}); "
            "choose one with --series"
        )
    return float(final[keys[0]][metric])


def ceiling(scores: Any, minimize: bool = False) -> dict[str, Any]:
    """Top-1 agreement and selection regret between the replicas of a score table.

    ``scores[r][k]`` is the score of configuration ``k`` in replica ``r``. A
    tie goes to the first configuration. The regret is in the metric's units,
    always ``>= 0``; the relative one divides it by the best score of ``b``.
    """
    table = np.asarray(scores, dtype=float)
    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 1:
        raise ValueError("ceiling: scores must be replicas x configurations, R >= 2")
    better = -table if minimize else table
    winners = better.argmax(axis=1)
    n = table.shape[0]
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    agree = [winners[a] == winners[b] for a, b in pairs]
    regret = np.array([better[b].max() - better[b, winners[a]] for a, b in pairs])
    best = np.array([np.abs(table[b, winners[b]]) for _, b in pairs])
    relative = np.divide(regret, best, out=np.full(len(pairs), np.nan), where=best > 0)
    return {
        "replicas": n,
        "configurations": table.shape[1],
        "pairs": len(pairs),
        "winners": winners.tolist(),
        "top1": float(np.mean(agree)),
        "top1_chance": 1.0 / table.shape[1],
        "regret": {
            "mean": float(regret.mean()),
            "max": float(regret.max()),
            "relative_mean": float(np.nanmean(relative))
            if np.isfinite(relative).any()
            else None,
        },
    }


def run_ceiling(
    config: ExperimentConfig,
    n: int | None = None,
    metric: str = "accuracy",
    series: str | None = None,
    resample: bool = False,
    paired: bool = False,
    minimize: bool | None = None,
    workers: int = 1,
    evaluate: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run the replicas, score each configuration per replica and compare them.

    ``n`` defaults to the number of ``seeds``. Every run is an ordinary signed
    run under ``paths.runs``; the result lists them per cell of the table.
    """
    from onion_fl.experiment.runner import run_scenarios

    n = len(config.seeds) if n is None else n
    minimize = metric in LOWER_IS_BETTER if minimize is None else minimize
    todo = replicas(config, n, resample=resample, paired=paired)
    paths = run_scenarios(
        [t.scenario for t in todo], workers=workers, evaluate=evaluate
    )
    names = list(dict.fromkeys(t.configuration for t in todo))
    cells: dict[tuple[int, str], list[tuple[float, Replica, Path]]] = {}
    for item, path in zip(todo, paths, strict=True):
        score = final_score(path, metric, series)
        cells.setdefault((item.replica, item.configuration), []).append(
            (score, item, path)
        )
    table, chosen = [], []
    for r in range(n):
        row, picks = [], {}
        for name in names:
            score, item, path = _best(cells[(r, name)], minimize)
            row.append(score)
            picks[name] = {
                "score": score,
                "seed": item.scenario.seed,
                "scenario": item.scenario.name,
                "config_id": item.scenario.config_id,
                "run_id": path.name,
                "trials": [t[0] for t in cells[(r, name)]],
            }
        table.append(row)
        chosen.append(picks)
    return {
        "experiment": config.name,
        "metric": metric,
        "series": series,
        "minimize": minimize,
        "resample": resample,
        "paired": paired,
        "configurations": names,
        "scores": table,
        "cells": chosen,
    } | ceiling(table, minimize=minimize)


def _best(
    trials: Sequence[tuple[float, Replica, Path]], minimize: bool
) -> tuple[float, Replica, Path]:
    """The best trial of a cell; a tie goes to the first trial."""
    pick = min if minimize else max
    return pick(trials, key=lambda t: t[0])
