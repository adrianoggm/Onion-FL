from __future__ import annotations

"""Run and analyse the deadline-coupling pilot (issue #213, capabilities spec §5).

The design, the target and the stopping criteria are pre-registered in
``results/pilot_deadline_coupling/PREREGISTRATION.md``; this script implements
them and nothing else.

    python scripts/pilot_deadline_coupling.py run --step 0 --workers 8
    python scripts/pilot_deadline_coupling.py analyse --step 0
    python scripts/pilot_deadline_coupling.py run --step 1 --workers 8
    python scripts/pilot_deadline_coupling.py analyse
    python scripts/pilot_deadline_coupling.py export

``run`` skips the runs already finished in ``runs/`` (same ``config_id`` and
seed), so an interrupted batch resumes. ``analyse`` needs the runs' events and
the data (the predictor builds each federation); ``export`` copies each run's
``run.json`` (without ``host``, ``pid`` and the per-edge inputs of the diagnostic)
and ``summary.json`` to the results.
"""

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

EXPERIMENT = "experiments/pilot_deadline_coupling.yaml"
RESULTS = Path("results/pilot_deadline_coupling")
TARGET = ("accuracy", 0.35)  # pre-registered
FEDERATIONS = ("pilot_dc_flat", "pilot_dc_two_fogs", "pilot_dc_four_fogs")
ALGORITHMS = ("fedavg", "fedadam")
CONFIGURATIONS = [(f, a) for f in FEDERATIONS for a in ALGORITHMS]
REGIMES = ("deadline", "fedbuff")
NAIVE = ("samples", 0.0)
COUPLED = (
    ("samples", -0.6),
    ("samples", 0.6),
    ("label_divergence", -0.6),
    ("label_divergence", 0.6),
)
STEP0_SEEDS = tuple(range(10))
SEEDS = tuple(range(5))
GATE = 0.6  # split-half top-1 below this: the regime is not identifiable
NOISE = 0.05  # the share of replica regrets above the ceiling's threshold
ALPHA = 0.01  # a count of exceeding cells this unlikely under noise is a gap
BOOTSTRAP = 2000


# --- what a run is -------------------------------------------------------------------


def classify(config: Mapping[str, Any]) -> dict[str, Any]:
    """Regime, coupling and configuration of a run of the pilot, from its config."""
    topology = config["topology"]
    fog = topology["fog"]["defaults"]
    regime = (
        "fedbuff"
        if fog.get("close_at_quorum")
        else "deadline"
        if fog.get("deadline") is not None
        else "sync"
    )
    optimizer = config["learning"].get("server_optimizer")
    name = optimizer if isinstance(optimizer, str) else (optimizer or {}).get("name")
    edges = config["infrastructure"]["assignment"]["edges"]
    return {
        "regime": regime,
        "coupling": (edges["statistic"], float(edges["rho"])),
        "configuration": (
            topology["name"],
            "fedavg" if name in (None, "replace") else name,
        ),
    }


def wanted(step: int, kind: Mapping[str, Any], seed: int) -> bool:
    """Is this scenario and seed one the pre-registered step runs?"""
    naive = kind["coupling"] == NAIVE
    if step == 0:
        if kind["regime"] == "sync":
            return naive and seed in SEEDS
        return naive and seed in STEP0_SEEDS
    return kind["regime"] in REGIMES and not naive and seed in SEEDS


# --- the statistics, pure ------------------------------------------------------------


def relative_regret(truth: Sequence[float], pick: int) -> float:
    """How much longer than the best the picked configuration takes, relative."""
    truth = np.asarray(truth, float)
    best = float(truth.min())
    return float((truth[pick] - best) / best) if best > 0 else 0.0


def split_half(table: Any) -> dict[str, Any]:
    """Two disjoint halves of the seeds as two replicas of a study.

    ``table[s][k]`` is the time to target of configuration ``k`` with seed ``s``
    (lower is better). Each half decides on its mean over seeds; over every way
    of splitting the seeds in two, the share of splits whose halves pick the same
    winner, and the relative regret of each half's winner in the other half.
    """
    table = np.asarray(table, float)
    n = table.shape[0]
    if n < 2 or n % 2:
        raise ValueError("split_half needs an even number of seeds, at least 2")
    agree, regrets = [], []
    for half in combinations(range(n), n // 2):
        if 0 not in half:  # each split once
            continue
        other = [s for s in range(n) if s not in half]
        a, b = table[list(half)].mean(axis=0), table[other].mean(axis=0)
        wa, wb = int(np.argmin(a)), int(np.argmin(b))
        agree.append(wa == wb)
        regrets += [relative_regret(b, wa), relative_regret(a, wb)]
    regrets = np.asarray(regrets)
    return {
        "splits": len(agree),
        "top1": float(np.mean(agree)),
        "top1_chance": 1.0 / table.shape[1],
        "regret_relative_mean": float(regrets.mean()),
        "regret_relative_p95": float(np.quantile(regrets, 1 - NOISE)),
    }


def auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Mann-Whitney AUC of ``scores`` for ``labels`` (ties count half); None with
    one class."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return None
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (
        pos[:, None] == neg[None, :]
    ).sum()
    return float(wins / (len(pos) * len(neg)))


def auc_ci(
    scores: Sequence[float],
    labels: Sequence[int],
    resamples: int = BOOTSTRAP,
    seed: int = 0,
) -> list[float] | None:
    """95 % percentile interval of the AUC over units resampled with replacement;
    resamples with one class are left out."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        rows = rng.integers(0, len(scores), size=len(scores))
        value = auc(scores[rows], labels[rows])
        if value is not None:
            draws.append(value)
    if not draws:
        return None
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def gap_verdict(exceeding: int, cells: int) -> str:
    """``exists`` when that many cells above the ceiling is unlikely as noise
    (binomial tail at ``NOISE`` <= ``ALPHA``), ``falls`` at 0 or 1, else
    ``inconclusive``."""
    if cells == 0:
        return "not identifiable"
    if stats.binom.sf(exceeding - 1, cells, NOISE) <= ALPHA:
        return "exists"
    return "falls" if exceeding <= 1 else "inconclusive"


def overlap(a: Sequence[float] | None, b: Sequence[float] | None) -> bool | None:
    if a is None or b is None:
        return None
    return bool(a[0] <= b[1] and b[0] <= a[1])


# --- running -------------------------------------------------------------------------


def _finished(runs: Path) -> set[tuple[str, int]]:
    done = set()
    for run_json in runs.glob("*/run.json"):
        meta = json.loads(run_json.read_text(encoding="utf-8"))
        if meta.get("status") == "finished":
            done.add((meta["config_id"], int(meta["seed"])))
    return done


def run(step: int, workers: int, dry: bool) -> None:
    from onion_fl.experiment.config import load_experiment
    from onion_fl.experiment.runner import run_scenarios
    from onion_fl.experiment.sweep import scenarios

    config = load_experiment(EXPERIMENT)
    if step == 1:
        passed = json.loads((RESULTS / "step0.json").read_text(encoding="utf-8"))
        regimes = [r for r, g in passed["gate"].items() if g["passed"]]
    else:
        regimes = list(REGIMES)
    todo = [
        s
        for s in scenarios(config)
        if wanted(step, kind := classify(s.config.dump()), s.seed)
        and (step == 0 or kind["regime"] in regimes)
    ]
    done = _finished(Path(config.paths.runs))
    left = [s for s in todo if (s.config_id, s.seed) not in done]
    print(f"step {step}: {len(todo)} runs, {len(left)} left", flush=True)
    if dry or not left:
        return
    for i in range(0, len(left), 4 * workers):  # a failure loses one chunk only
        chunk = left[i : i + 4 * workers]
        run_scenarios(chunk, workers=workers)
        print(f"  {i + len(chunk)}/{len(left)}", flush=True)


# --- analysis ------------------------------------------------------------------------


def _load(runs: Path) -> dict[tuple, Any]:
    from onion_fl.experiment.fidelity import Cell, design_id

    cells: dict[tuple, Any] = {}
    for run_json in sorted(runs.glob("*/run.json")):
        meta = json.loads(run_json.read_text(encoding="utf-8"))
        config = meta.get("config") or {}
        if config.get("name") != "pilot_deadline_coupling":
            continue
        if meta.get("status") != "finished":
            continue
        kind = classify(config)
        key = (kind["regime"], kind["coupling"], kind["configuration"], meta["seed"])
        cell = Cell(design_id(config), int(meta["seed"]), run_json.parent, meta)
        older = cells.get(key)
        if older is None or older.meta["started_at"] < meta["started_at"]:
            cells[key] = cell
    return cells


def _measure(cell: Any) -> dict[str, Any]:
    from onion_fl.experiment.fidelity import time_to_target
    from onion_fl.experiment.predictor import rounds_to_target
    from onion_fl.observability.events import read_events

    events = read_events(cell.path / "events.jsonl")
    rounds, censored = rounds_to_target(events, *TARGET)
    summary = json.loads((cell.path / "summary.json").read_text(encoding="utf-8"))
    final = summary["final"]
    accuracy = {k: v["accuracy"] for k, v in final.items() if "accuracy" in v}
    diagnostic = cell.meta.get("diagnostic") or {}
    return {
        "run_id": cell.meta["run_id"],
        "run_hash": cell.meta["run_hash"],
        "config_id": cell.meta["config_id"],
        "topology_id": cell.meta["topology_id"],
        "code": cell.meta.get("code_version"),
        "seed": cell.seed,
        "time_to_target": time_to_target(events, *TARGET),
        "rounds_to_target": rounds,
        "censored": censored,
        "time": summary["t_virtual_end"],
        "final_accuracy": accuracy,
        "diagnostic": {
            "D": diagnostic.get("D"),
            "D_gradient_norm": (diagnostic.get("statistics") or {})
            .get("gradient_norm", {})
            .get("D"),
            "participation": diagnostic.get("participation"),
            "floss_rounds_factor": (diagnostic.get("floss") or {}).get("rounds_factor"),
            "error": diagnostic.get("error"),
        },
    }


def _table(
    measured: Mapping[tuple, Any], regime, coupling, seeds, field="time_to_target"
):
    return np.array(
        [
            [measured[(regime, coupling, c, s)][field] for c in CONFIGURATIONS]
            for s in seeds
        ],
        float,
    )


def _complete(measured, regime, coupling, seeds) -> bool:
    return all(
        (regime, coupling, c, s) in measured for c in CONFIGURATIONS for s in seeds
    )


def step0(measured: Mapping[tuple, Any]) -> dict[str, Any]:
    from onion_fl.experiment.ceiling import ceiling

    out: dict[str, Any] = {"target": list(TARGET), "gate": {}, "control": None}
    for regime in REGIMES:
        if not _complete(measured, regime, NAIVE, STEP0_SEEDS):
            out["gate"][regime] = {"passed": False, "missing": True}
            continue
        table = _table(measured, regime, NAIVE, STEP0_SEEDS)
        halves = split_half(table)
        single = ceiling(table, minimize=True)
        out["gate"][regime] = {
            "passed": halves["top1"] >= GATE,
            "split_half": halves,
            "single_run": {
                "top1": single["top1"],
                "regret_relative_mean": single["regret"]["relative_mean"],
            },
            "means": table.mean(axis=0).tolist(),
            "censored": int(
                sum(
                    measured[(regime, NAIVE, c, s)]["censored"]
                    for c in CONFIGURATIONS
                    for s in STEP0_SEEDS
                )
            ),
        }
    if _complete(measured, "sync", NAIVE, SEEDS):
        table = _table(measured, "sync", NAIVE, SEEDS)
        single = ceiling(table, minimize=True)
        out["control"] = {
            "means": table.mean(axis=0).tolist(),
            "single_run_top1": single["top1"],
        }
    out["identifiable"] = any(g.get("passed") for g in out["gate"].values())
    return out


def _predicted(cells, measured) -> dict[tuple, Any]:
    from onion_fl.experiment.predictor import predictions

    control = {k: c for k, c in cells.items() if k[0] == "sync"}
    timed = {k: c for k, c in cells.items() if k[0] in REGIMES and k[3] in SEEDS}
    predicted, warnings = predictions(timed, control, TARGET)
    for warning in warnings:
        print(warning, file=sys.stderr)
    return predicted


def step2(measured, predicted, ceilings: Mapping[str, float]) -> dict[str, Any]:
    cells = []
    for regime in ceilings:
        naive = _table(measured, regime, NAIVE, SEEDS).mean(axis=0)
        for coupling in (NAIVE, *COUPLED):
            if not _complete(measured, regime, coupling, SEEDS):
                continue
            truth = _table(measured, regime, coupling, SEEDS).mean(axis=0)
            row = {
                "regime": regime,
                "statistic": coupling[0],
                "rho": coupling[1],
                "truth": truth.tolist(),
                "truth_winner": CONFIGURATIONS[int(np.argmin(truth))],
                "ceiling_p95": ceilings[regime],
            }
            picks = {"naive": int(np.argmin(naive))}
            for variant in ("factorised", "floss"):
                try:
                    values = np.array(
                        [
                            [
                                predicted[(regime, coupling, c, s)][variant][
                                    "time_to_target"
                                ]
                                for c in CONFIGURATIONS
                            ]
                            for s in SEEDS
                        ]
                    ).mean(axis=0)
                except KeyError:
                    continue
                row[f"{variant}_predicted"] = values.tolist()
                picks[variant] = int(np.argmin(values))
            for who, pick in picks.items():
                regret = relative_regret(truth, pick)
                row[who] = {
                    "winner": CONFIGURATIONS[pick],
                    "regret_relative": regret,
                    "above_ceiling": regret > ceilings[regime],
                }
            cells.append(row)
    coupled = [c for c in cells if c["rho"] != 0.0]
    verdicts = {}
    for who in ("floss", "factorised", "naive"):
        rows = [c for c in coupled if who in c]
        count = sum(c[who]["above_ceiling"] for c in rows)
        verdicts[who] = {
            "cells": len(rows),
            "above_ceiling": count,
            "verdict": gap_verdict(count, len(rows)),
        }
    return {"cells": cells, "coupled": verdicts}


def step3(measured, cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    units = [c for c in cells if "floss" in c]
    labels = [int(c["floss"]["above_ceiling"]) for c in units]

    def per_cell(c, field, reduce):
        values = [
            measured[(c["regime"], (c["statistic"], c["rho"]), k, s)]["diagnostic"][
                field
            ]
            for k in CONFIGURATIONS
            for s in SEEDS
        ]
        values = [v for v in values if v is not None]
        return reduce(values) if values else math.nan

    scores = {
        "D": [per_cell(c, "D", lambda v: max(abs(x) for x in v)) for c in units],
        "D_gradient_norm": [
            per_cell(c, "D_gradient_norm", lambda v: max(abs(x) for x in v))
            for c in units
        ],
        "participation": [
            per_cell(c, "participation", lambda v: 1 - float(np.mean(v))) for c in units
        ],
        "floss": [per_cell(c, "floss_rounds_factor", max) for c in units],
    }
    out: dict[str, Any] = {
        "units": len(units),
        "positives": int(sum(labels)),
        "scores": scores,
    }
    for name, values in scores.items():
        out[name] = {"auc": auc(values, labels), "ci": auc_ci(values, labels)}
    out["identifiable"] = 0 < sum(labels) < len(labels)
    out["D_overlaps_participation"] = overlap(
        out["D"]["ci"], out["participation"]["ci"]
    )
    out["D_overlaps_floss"] = overlap(out["D"]["ci"], out["floss"]["ci"])
    return out


def analyse(runs: Path, only_step0: bool) -> dict[str, Any]:
    cells = _load(runs)
    measured = {k: _measure(c) for k, c in cells.items()}
    result: dict[str, Any] = {"step0": step0(measured)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "step0.json").write_text(
        json.dumps(result["step0"], indent=2) + "\n", encoding="utf-8"
    )
    if not only_step0:
        gate = result["step0"]["gate"]
        ceilings = {
            r: g["split_half"]["regret_relative_p95"]
            for r, g in gate.items()
            if g.get("passed")
        }
        predicted = _predicted(cells, measured)
        result["step2"] = step2(measured, predicted, ceilings)
        result["step3"] = step3(measured, result["step2"]["cells"])
    result["runs"] = [
        {
            "regime": k[0],
            "statistic": k[1][0],
            "rho": k[1][1],
            "federation": k[2][0],
            "algorithm": k[2][1],
        }
        | m
        for k, m in sorted(measured.items(), key=lambda kv: str(kv[0]))
    ]
    name = "step0_analysis.json" if only_step0 else "analysis.json"
    (RESULTS / name).write_text(
        json.dumps(result, indent=1, default=_plain) + "\n", encoding="utf-8"
    )
    return result


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def export(runs: Path) -> None:
    out = RESULTS / "runs"
    for cell in _load(runs).values():
        target = out / cell.path.name
        target.mkdir(parents=True, exist_ok=True)
        meta = {k: v for k, v in cell.meta.items() if k not in ("host", "pid")}
        if isinstance(meta.get("diagnostic"), dict):  # per-edge inputs: ~70 % of it
            meta["diagnostic"] = {
                k: v for k, v in meta["diagnostic"].items() if k != "edges"
            }
        (target / "run.json").write_text(
            json.dumps(meta, indent=1) + "\n", encoding="utf-8"
        )
        summary = (cell.path / "summary.json").read_text(encoding="utf-8")
        (target / "summary.json").write_text(summary, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="The deadline-coupling pilot (#213)")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--step", type=int, choices=[0, 1], required=True)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--dry", action="store_true", help="only count the runs")
    a = sub.add_parser("analyse")
    a.add_argument("--runs", default="runs")
    a.add_argument("--step", type=int, choices=[0], help="step 0 only")
    e = sub.add_parser("export")
    e.add_argument("--runs", default="runs")
    args = p.parse_args(argv)
    if args.command == "run":
        run(args.step, args.workers, args.dry)
    elif args.command == "analyse":
        result = analyse(Path(args.runs), args.step == 0)
        summary = {k: v for k, v in result.items() if k != "runs"}
        print(json.dumps(summary, indent=1, default=_plain))
    else:
        export(Path(args.runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
