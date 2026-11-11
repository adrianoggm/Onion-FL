from __future__ import annotations

"""Reduced #198 comparison: cloud-1ep and fog_a-1ep x seeds, emulated with Nagle
(backend nodelay: false, as the pre-registered runs) and without (the default
since the fix), seed by seed with the arms alternating; then the sim.

    python scripts/latency_followup/run_reduced.py OUT --seeds 0 1 2 3 4 [--sim]
"""

import argparse
from pathlib import Path

from onion_fl.experiment.config import load_experiment, parse_experiment
from onion_fl.experiment.runner import run_scenarios
from onion_fl.experiment.sweep import scenarios

KEEP = (
    "local_epochs=1,topology,infrastructure.assignment=swell_cloud_npz",
    "local_epochs=1,topology,infrastructure.assignment=swell_one_fog_npz,{'root': 'cloud-1', 'aggregators': {'fog_1': 'fog-a'}",
)


def chosen(raw, seed):
    r = dict(raw) | {"seeds": [seed]}
    return [s for s in scenarios(parse_experiment(r)) if any(k in s.name for k in KEEP)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--sim", action="store_true")
    args = ap.parse_args()
    raw = load_experiment(Path("experiments/fidelity_sim_emulated.yaml")).dump()
    if args.sim:
        r = raw | {
            "backend": "sim",
            "paths": raw["paths"] | {"runs": f"{args.out}/sim"},
        }
        todo = [s for seed in args.seeds for s in chosen(r, seed)]
        for p in run_scenarios(todo, workers=3):
            print("sim", p, flush=True)
        return
    for seed in args.seeds:
        for i, _ in enumerate(KEEP):
            for arm in ("nagle", "nodelay") if seed % 2 == 0 else ("nodelay", "nagle"):
                backend = dict(raw["backend"])
                if arm == "nagle":
                    backend["nodelay"] = False
                r = raw | {
                    "backend": backend,
                    "paths": raw["paths"] | {"runs": f"{args.out}/{arm}"},
                }
                (s,) = [x for x in chosen(r, seed) if KEEP[i] in x.name]
                for p in run_scenarios([s], workers=1):
                    print(arm, seed, i, p, flush=True)


if __name__ == "__main__":
    main()
