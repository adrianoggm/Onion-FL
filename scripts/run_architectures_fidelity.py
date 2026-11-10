from __future__ import annotations

"""Run infrastructure_architectures_logical.yaml on one backend (issue #192).

    python scripts/run_architectures_fidelity.py --backend sim|local \
        --broker host:port [--workers N]

The topologies' MQTT links name no broker (``localhost:1883`` by default); this
writes copies of them that point at ``--broker`` under ``runs/``, so both
backends run the very same topologies (the broker is part of the
``topology_id``), and keeps each backend's runs in its own folder,
``runs/architectures_<backend>``, which is what ``onion_fl fidelity`` compares.
The local backend's runs go one at a time by default, so that one run's wall
clock does not include another's work.
"""

import argparse
from pathlib import Path

import yaml

from onion_fl.experiment.config import load_experiment, parse_experiment
from onion_fl.experiment.runner import run_experiment

EXPERIMENT = Path("experiments/infrastructure_architectures_logical.yaml")
TOPOLOGIES = ("cloud_swell_wesad", "four_fogs_swell_wesad_physical")


def with_broker(raw: dict, broker: str) -> dict:
    def link(spec: dict | None) -> None:
        if spec is not None:
            spec["transport"] = {"name": "mqtt", "broker": broker}

    link(raw.get("edge", {}).get("link_up"))
    for block in raw.values():
        if isinstance(block, dict):
            link(block.get("defaults", {}).get("link_up"))
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sim", "local"], required=True)
    parser.add_argument("--broker", required=True, help="host:port")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    folder = Path("runs") / f"topologies_{args.broker.replace(':', '_')}"
    folder.mkdir(parents=True, exist_ok=True)
    for name in TOPOLOGIES:
        raw = yaml.safe_load(Path(f"topologies/{name}.yaml").read_text("utf-8"))
        text = yaml.safe_dump(with_broker(raw, args.broker), sort_keys=False)
        (folder / f"{name}.yaml").write_text(text, encoding="utf-8")

    raw = load_experiment(EXPERIMENT).dump()
    raw["backend"] = args.backend
    raw["paths"] = raw["paths"] | {
        "topologies": folder.as_posix(),
        "runs": f"runs/architectures_{args.backend}",
    }
    for path in run_experiment(parse_experiment(raw), workers=args.workers):
        print(path, flush=True)


if __name__ == "__main__":
    main()
