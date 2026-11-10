from __future__ import annotations

"""Traced, sniffed emulated runs of experiments/fidelity_sim_emulated.yaml (#198).

    python scripts/latency_followup/traced_session.py --out runs/a198/<session> \
        --arms nagle nodelay --seeds 0 1 --scenario <part of a name> [--sniff]

Every run has ``ONIONFL_TRACE=1``, so its node processes record ``trace.sent``
and ``trace.delivered``. ``--sniff`` also starts, once the lab is ready, one
``sniff_mqtt.py`` container in the network namespace of ``cloud-1`` (the
broker's), of the fogs and of ``field-edge_device-1``, writing
``emulation/sniff-<host>.jsonl`` in the run; ``path_breakdown.py`` joins both.

Arms: ``nagle`` is ``backend.nodelay: false`` (the runs before #198), ``nodelay``
the default. Runs go seed by seed, the arms alternating, so a drift of the
machine falls on seeds. Session s1 of the INDEX also had a client-only and a
broker-only arm, run with a temporary patch that is not committed.
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

from onion_fl.experiment import emulated
from onion_fl.experiment.config import load_experiment, parse_experiment
from onion_fl.experiment.runner import run_scenarios
from onion_fl.experiment.sweep import scenarios

SNIFF = Path(__file__).with_name("sniff_mqtt.py").resolve()
HOSTS = ("cloud-1", "fog-a", "fog-b", "field-edge_device-1")
STATE: dict = {"sniff": False, "sniffers": []}
_wait_ready, _take_down = emulated.wait_ready, emulated.take_down


def wait_ready(plan) -> None:
    _wait_ready(plan)
    if not STATE["sniff"]:
        return
    folder = plan.folder.resolve().as_posix()
    for host in HOSTS:
        if host not in plan.emulation.layout.hosts:
            continue
        container = plan.emulation.container(host)
        name = f"{container}-sniff"
        subprocess.run(
            [
                "docker", "run", "-d", "--rm", "--name", name,
                "--network", f"container:{container}",
                "--cap-add", "NET_RAW", "--cap-add", "NET_ADMIN",
                "--cpuset-cpus", "20-23",
                "-v", f"{folder}:/out",
                "-v", f"{SNIFF.parent.as_posix()}:/sniff:ro",
                plan.params.image, "python", f"/sniff/{SNIFF.name}",
                f"/out/sniff-{host}.jsonl",
            ],
            check=True,
            capture_output=True,
        )  # fmt: skip
        STATE["sniffers"].append(name)
    time.sleep(1.0)  # their sockets are open before the nodes start


def take_down(plan) -> None:
    for name in STATE["sniffers"]:
        subprocess.run(["docker", "stop", "-t", "3", name], capture_output=True)
    STATE["sniffers"].clear()
    _take_down(plan)


emulated.wait_ready, emulated.take_down = wait_ready, take_down


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=["nagle", "nodelay"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--scenario", nargs="+", help="parts of the scenario names")
    ap.add_argument("--sniff", action="store_true")
    args = ap.parse_args()
    STATE["sniff"] = args.sniff
    os.environ["ONIONFL_TRACE"] = "1"
    raw = load_experiment(Path("experiments/fidelity_sim_emulated.yaml")).dump()
    names = list(dict.fromkeys(s.name for s in scenarios(parse_experiment(raw))))
    chosen = [
        n for n in names if not args.scenario or any(k in n for k in args.scenario)
    ]
    for seed in args.seeds:
        for name in chosen:
            for arm in args.arms:
                backend = dict(raw["backend"])
                if arm == "nagle":
                    backend["nodelay"] = False
                r = raw | {
                    "seeds": [seed],
                    "backend": backend,
                    "paths": raw["paths"] | {"runs": f"{args.out}/{arm}"},
                }
                todo = [s for s in scenarios(parse_experiment(r)) if s.name == name]
                for path in run_scenarios(todo, workers=1):
                    print(arm, seed, path, flush=True)


if __name__ == "__main__":
    main()
