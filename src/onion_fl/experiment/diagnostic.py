from __future__ import annotations

"""Ex ante diagnostic of the bias a deadline induces (N8, issue #212).

Computed before the federation trains, from what the factorised predictor
already knows of the configuration (``predictor.plan_from`` and
``planned_round``) and a cheap statistic of each training edge's data:

- ``p_i``, the probability that edge i's update reaches the root in time: the
  per-hop deadline probabilities of the predictor's FLOSS-style variant
  (lognormal CDF, share of a trace, yes or no for deterministic models);
- ``w_i``, its aggregation weight, its examples over those of every edge;
- ``d_i``, its statistic: ``label_divergence`` (the KL divergence of its labels
  from those of its dataset, the primary one) and ``gradient_norm`` (the norm of
  the loss gradient of the initial model on its data), both from
  ``runner.edge_statistics``.

With ``u_i = w_i d_i``, ``D = Cov_i(p_i, u_i) / (mean(p) mean(u))``, which is
``sum(p u) / (mean(p) sum(u)) - 1``: the share of the divergence that arrives
over the share of edges that do, minus one. 0 when who arrives is unrelated to
what they bring (or nothing diverges); negative when the edges that miss the
deadline are the ones that bring something different, so the model trained is
not the one a predictor without deadlines assumes; None when nobody arrives.

The baselines it is compared with: the mean participation rate alone, and the
FLOSS-style correction (the participation weighted by examples and the factor
by which it multiplies the rounds to target).
"""

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from onion_fl.experiment.predictor import (
    Plan,
    Round,
    arrival_probabilities,
    plan_from,
    planned_round,
)

STATISTICS = ("label_divergence", "gradient_norm")


def bias(
    p: Sequence[float], w: Sequence[float], d: Sequence[float]
) -> dict[str, float | None]:
    """The covariance of ``p`` and ``w * d`` over the edges, and its normalised
    ``D``."""
    p, u = np.asarray(p, float), np.asarray(w, float) * np.asarray(d, float)
    if not len(p):
        return {"covariance": 0.0, "D": None}
    covariance = float(np.mean(p * u) - np.mean(p) * np.mean(u))
    if np.mean(u) == 0:  # nothing diverges: nothing to lose
        normalised = 0.0
    elif np.mean(p) == 0:  # nobody arrives
        normalised = None
    else:
        normalised = covariance / float(np.mean(p) * np.mean(u))
    return {"covariance": covariance, "D": normalised}


def diagnose(
    plan: Plan,
    root: str,
    rnd: Round,
    statistics: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """The diagnostic of a round planned on a configuration's models, with each
    statistic's value per training edge (the first one is the primary)."""
    reach = arrival_probabilities(plan, root, rnd)
    edges = sorted(rnd.examples)
    total = sum(rnd.examples.values())
    p = [reach.get(e, 0.0) for e in edges]
    w = [rnd.examples[e] / total if total else 0.0 for e in edges]
    by = {
        name: bias(p, w, [values.get(e, 0.0) for e in edges])
        for name, values in statistics.items()
    }
    weighted = float(np.dot(p, w)) if total else 0.0
    primary = next(iter(statistics), None)
    return {
        "statistic": primary,
        **(by[primary] if primary else {"covariance": 0.0, "D": None}),
        "statistics": by,
        "participation": float(np.mean(p)) if p else 0.0,
        "floss": {
            "participation": weighted,
            "rounds_factor": 1 / weighted if weighted > 0 else None,
        },
        "deadlines": sorted(n for n, t in plan.deadlines.items() if t is not None),
        "unknown": plan.unknown,  # no closed-form compute time: assumed in time
        "edges": {
            e: {"p": pi, "w": wi}
            | {name: float(values.get(e, 0.0)) for name, values in statistics.items()}
            for e, pi, wi in zip(edges, p, w, strict=True)
        },
    }


def diagnose_scenario(
    federation: Any,
    scenario: Any,
    placement: Any,
    statistics: Sequence[str] = STATISTICS,
) -> dict[str, Any]:
    """``diagnose`` with the statistics of a scenario's placed edges on the
    federation's initial model, timed (``seconds``)."""
    from onion_fl.experiment.runner import edge_statistics

    start = time.perf_counter()
    initial = federation.coordinator.state
    values = {
        name: edge_statistics(name, scenario, placement, initial) for name in statistics
    }
    out = diagnose(plan_from(federation), *planned_round(federation), values)
    return out | {"seconds": time.perf_counter() - start}
