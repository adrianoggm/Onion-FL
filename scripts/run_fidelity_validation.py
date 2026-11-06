from __future__ import annotations

"""Run experiments/fidelity_sim_emulated.yaml on one backend (issue #198).

    python scripts/run_fidelity_validation.py --backend emulated|sim [--runs DIR]

The experiment carries the emulated backend's block (image, project, CPUs);
``--backend sim`` replaces it, which ``design_id`` leaves out, so the runs of
both backends pair. Each backend's runs go to their own folder,
``runs/fidelity_validation_<backend>`` by default, which is what ``onion_fl
fidelity`` compares. Emulated runs go one at a time, since a run's wall clock
must not include another's work; sim runs, on a virtual clock, may take
``--workers``. They go seed by seed, every configuration once per seed, so
that a drift of the machine during the session falls on seeds, which the
analysis pairs, and not on configurations. A run already finished in the
folder (same ``config_id`` and seed) is not run again, so an interrupted
session resumes.

Before running, it checks what the pre-registration fixes
(results/fidelity_sim_emulated/PREREGISTRATION.md): every profile and
calibrated link of the infrastructure cites one calibration, the one
``--calibration`` names when given, and that calibration exists.

``--seeds`` and ``--rounds`` are for smoke tests only: they change the
``design_id``, so their runs never pair with the real ones.
"""

import argparse
import json
from pathlib import Path

from onion_fl.experiment.config import load_experiment, parse_experiment
from onion_fl.experiment.runner import run_scenarios
from onion_fl.experiment.sweep import scenarios
from onion_fl.infrastructure.calibration import cited
from onion_fl.infrastructure.model import load_infrastructure

EXPERIMENT = Path("experiments/fidelity_sim_emulated.yaml")


def calibration_of(raw: dict) -> str:
    """The one calibration the experiment's infrastructure cites."""
    folder = Path(raw["paths"].get("infrastructures", "infrastructures"))
    infra = load_infrastructure(folder / f"{raw['infrastructure']['name']}.yaml")
    ids = cited(infra)
    if len(ids) != 1:
        raise SystemExit(f"the infrastructure cites {ids}; the validation needs one")
    return ids[0]


def finished(runs: Path) -> set[tuple[str, int]]:
    """``(config_id, seed)`` of the runs in ``runs`` that finished."""
    done = set()
    for meta in runs.glob("*/run.json"):
        run = json.loads(meta.read_text(encoding="utf-8"))
        if run.get("status") == "finished":
            done.add((run.get("config_id"), run.get("seed")))
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sim", "emulated"], required=True)
    parser.add_argument("--runs", help="default runs/fidelity_validation_<backend>")
    parser.add_argument("--calibration", help="the calibration_id it must cite")
    parser.add_argument("--seeds", type=int, nargs="+", help="smoke tests only")
    parser.add_argument("--rounds", type=int, help="smoke tests only")
    parser.add_argument("--scenario", help="only the scenario with this name")
    parser.add_argument("--workers", type=int, default=1, help="sim only")
    args = parser.parse_args()
    if args.backend == "emulated" and args.workers != 1:
        raise SystemExit("emulated runs go one at a time: leave --workers out")

    raw = load_experiment(EXPERIMENT).dump()
    cid = calibration_of(raw)
    if args.calibration is not None and cid != args.calibration:
        raise SystemExit(f"the infrastructure cites {cid}, not {args.calibration}")
    calibrations = Path(raw["paths"].get("calibrations", "calibrations"))
    if not (calibrations / cid / "calibration.json").is_file():
        raise SystemExit(f"calibration {cid} is not in {calibrations}/")
    print(f"calibration {cid}", flush=True)

    if args.backend == "sim":
        raw["backend"] = "sim"
    if args.seeds:
        raw["seeds"] = args.seeds
    if args.rounds:
        raw["rounds"] = args.rounds
    runs = args.runs or f"runs/fidelity_validation_{args.backend}"
    raw["paths"] = raw["paths"] | {"runs": runs}
    config = parse_experiment(raw)
    every = scenarios(config)
    order = {name: i for i, name in enumerate(dict.fromkeys(s.name for s in every))}
    if args.scenario is not None and args.scenario not in order:
        raise SystemExit(f"no scenario named {args.scenario!r}; they are {list(order)}")
    done = finished(Path(runs))
    todo = sorted(
        (
            s
            for s in every
            if args.scenario in (None, s.name) and (s.config_id, s.seed) not in done
        ),
        key=lambda s: (s.seed, order[s.name]),
    )
    print(f"{len(todo)} run(s) to go, {len(done)} already finished", flush=True)
    for path in run_scenarios(todo, workers=args.workers) if todo else []:
        print(path, flush=True)


if __name__ == "__main__":
    main()
