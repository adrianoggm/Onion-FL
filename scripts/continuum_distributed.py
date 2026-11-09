"""The distributed continuum on containers acting as machines (issue #164).

    python scripts/continuum_distributed.py up --prefix onionfl-a164 --into DIR
    python scripts/continuum_distributed.py run experiments/continuum_distributed.yaml \
        --inventory DIR/inventory.yaml --runs runs --seed 0 --mode sim
    python scripts/continuum_distributed.py down --prefix onionfl-a164
    python scripts/continuum_distributed.py resumed runs/<run_id> ...  > resumed.csv

``up`` starts five containers from the ``distributed`` image (built from
``src/onion_fl/infrastructure/distributed.Dockerfile`` on the emulated host
image, ``--image``), with this checkout's ``onion_fl`` mounted read-only and
a throwaway SSH key, plus the broker in ``cloud-1``'s network namespace, as
``test_runtime_equivalence`` does; it writes their inventory. The launcher
reaches them only through SSH, as it would real machines.

``run`` runs one seed of the experiment with ``--mode``:

- ``sim``: the simulation of the same config (the reference);
- ``distributed``: on the machines, every edge saving its part of the bundle
  on its machine (``checkpoints``), the run's clock starting 60 s after the
  launch so every node is up before the streams start;
- ``faults``: the same with real failures: ``field-1``'s process killed at
  60 s and started again 15 s later with ``--resume``, and ``field-3``'s
  edges joining at 90 s.

``times`` writes, per run, its identity and what the simulation is compared
on besides the benchmark's metrics: rounds, messages and bytes, the clock
(when round 1 opened, the end, the wall seconds), the root's round
durations, failed quorums and late updates.

``resumed`` writes, per run and edge, what each edge trained and consumed and,
for the restarted ones, what they came back with (``node.resumed``).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

MACHINES = ["cloud-1", "fog-1", "field-1", "field-2", "field-3"]
FAULTS = {
    "restarts": [{"host": "field-1", "at": 60, "down": 15}],
    "joins": {"field-3": 90},
}
START_IN = 60.0


def docker(*args: str, check: bool = True) -> str:
    done = subprocess.run(["docker", *args], capture_output=True, text=True)
    if check and done.returncode != 0:
        raise RuntimeError(f"docker {args[0]}: {done.stderr.strip()}")
    return done.stdout.strip()


def up(prefix: str, image: str, into: Path) -> Path:
    import onion_fl
    from onion_fl.experiment.distributed import ssh_command
    from onion_fl.infrastructure.inventory import parse_inventory

    into.mkdir(parents=True, exist_ok=True)
    key = into / "id_ed25519"
    if not key.exists():
        subprocess.run(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                prefix,
                "-f",
                str(key),
            ],
            check=True,
        )
    public = (into / "id_ed25519.pub").read_text(encoding="utf-8").strip()
    source = Path(onion_fl.__file__).resolve().parent.as_posix()
    mount = f"type=bind,source={source},target=/onionfl/src/onion_fl,readonly"
    docker("network", "create", f"{prefix}-net")
    for host in MACHINES:
        docker(
            "run", "-d", "--name", f"{prefix}-{host}", "--hostname", host,
            "--network", f"{prefix}-net", "-p", "127.0.0.1::22",
            "-e", f"AUTHORIZED_KEY={public}", "--mount", mount, image,
        )  # fmt: skip
    docker(
        "run", "-d", "--name", f"{prefix}-mqtt",
        "--network", f"container:{prefix}-cloud-1",
        "eclipse-mosquitto:2", "mosquitto", "-c", "/mosquitto-no-auth.conf",
    )  # fmt: skip
    raw = {
        "machines": {
            host: {
                "address": "127.0.0.1",
                "port": int(docker("port", f"{prefix}-{host}", "22").rsplit(":")[-1]),
                "user": "root",
                "key": key.as_posix(),
                "python": "python",
                "root": "/root/onionfl",
                "advertise": f"{prefix}-{host}",
                "env": {"PYTHONPATH": "/onionfl/src"},
            }
            for host in MACHINES
        },
        "known_hosts": (into / "known_hosts").as_posix(),
        # Docker Desktop's VM keeps its clock close to the host's, not exact
        # (0.73 s off on 2026-11-09). Every container shares the VM's clock, so
        # the nodes agree with each other; run.json records the offset.
        "clock": {"max_offset": 1.0},
    }
    inventory = parse_inventory(raw)
    deadline = time.monotonic() + 60
    for host in MACHINES:
        line = ssh_command(
            "ssh", inventory, inventory.machine(host), "true", no_stdin=True
        )
        while subprocess.run(line, capture_output=True).returncode:
            if time.monotonic() > deadline:
                raise TimeoutError(f"no SSH to {host}")
            time.sleep(0.5)
    path = into / "inventory.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
    return path


def down(prefix: str) -> None:
    for host in [*MACHINES, "mqtt"]:
        docker("rm", "-f", f"{prefix}-{host}", check=False)
    docker("network", "rm", f"{prefix}-net", check=False)


def run(experiment: Path, inventory: Path, runs: Path, seed: int, mode: str) -> Path:
    from onion_fl.experiment.config import parse_experiment
    from onion_fl.experiment.runner import run_experiment

    raw = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    raw["seeds"] = [seed]
    raw["paths"] = raw.get("paths", {}) | {"runs": str(runs)}
    if mode != "sim":
        backend = {
            "name": "distributed",
            "inventory": str(inventory),
            "start_in": START_IN,
            "checkpoints": True,
        }
        raw["backend"] = backend | (FAULTS if mode == "faults" else {})
    (path,) = run_experiment(parse_experiment(raw))
    return path


def resumed(paths: list[Path]) -> None:
    """Per run and edge: trainings, rows trained, rows predicted, and resumes."""
    out = csv.writer(sys.stdout, lineterminator="\n")
    out.writerow(
        ["run_id", "run_hash", "backend", "faults", "node", "host", "trainings",
         "examples", "predicted", "resumed_at", "resumed_round", "resumed_consumed",
         "resumed_predicted", "first_training"]
    )  # fmt: skip
    for path in paths:
        meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
        faults = bool(meta.get("distributed", {}).get("faults"))
        scenario = path / "scenario.json"
        groups = (
            json.loads(scenario.read_text(encoding="utf-8")).get("groups", {})
            if scenario.is_file()
            else {}
        )
        hosts = {node: host for host, nodes in groups.items() for node in nodes}
        per: dict[str, dict] = defaultdict(lambda: defaultdict(float))
        for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            node, name = e["node"], e["name"]
            if name == "edge.trained":
                per[node]["trainings"] += 1
                per[node]["examples"] += e["tags"].get("examples", 0)
                per[node].setdefault("first_training", e["t_virtual"])
            elif name == "eval.accuracy" and e["tags"].get("model") == "prequential":
                per[node]["predicted"] += e["tags"].get("samples", 0)
            elif name == "node.resumed":
                per[node]["resumed_at"] = e["t_virtual"]
                per[node]["resumed_round"] = e["tags"].get("round", "")
                per[node]["resumed_consumed"] = e["tags"].get("consumed", "")
                per[node]["resumed_predicted"] = e["tags"].get("predicted", "")
        for node in sorted(per):
            row = per[node]
            out.writerow(
                [meta["run_id"], meta.get("run_hash"), meta.get("backend"), faults,
                 node, hosts.get(node, ""), int(row["trainings"]), int(row["examples"]),
                 int(row["predicted"]), row.get("resumed_at", ""),
                 row.get("resumed_round", ""), row.get("resumed_consumed", ""),
                 row.get("resumed_predicted", ""), row.get("first_training", "")]
            )  # fmt: skip


def times(paths: list[Path]) -> None:
    out = csv.writer(sys.stdout, lineterminator="\n")
    out.writerow(
        ["run_id", "config_id", "topology_id", "run_hash", "code_version", "seed",
         "backend", "faults", "status", "rounds", "messages", "bytes", "quorum_failed",
         "late_updates", "round1_at", "t_end", "wall_seconds", "root_round_mean",
         "root_round_max", "max_clock_offset"]
    )  # fmt: skip
    for path in paths:
        meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        first, closed, late = None, [], 0
        for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e["node"] == "cloud" and e["name"] == "round.started" and first is None:
                first = e["t_virtual"]
            elif e["node"] == "cloud" and e["name"] == "round.closed":
                closed.append(e["value"])
            elif e["name"] == "update.late":
                late += 1
        machines = meta.get("distributed", {}).get("machines", {})
        offsets = [abs(m.get("offset") or 0) for m in machines.values()]
        out.writerow(
            [meta["run_id"], meta["config_id"], meta["topology_id"], meta["run_hash"],
             meta.get("code_version"), meta["seed"], meta.get("backend"),
             bool(meta.get("distributed", {}).get("faults")), meta["status"],
             summary["rounds"], summary["messages"]["sent"], summary["messages"]["bytes"],
             summary.get("quorum_failed"), late, round(first or 0.0, 3),
             round(summary["t_virtual_end"], 3), round(summary["wall_seconds"], 1),
             round(sum(closed) / len(closed), 3) if closed else "",
             round(max(closed), 3) if closed else "",
             round(max(offsets), 3) if offsets else ""]
        )  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    u = sub.add_parser("up")
    u.add_argument("--prefix", required=True)
    u.add_argument("--image", default="onionfl-distributed:latest")
    u.add_argument("--into", type=Path, required=True)
    d = sub.add_parser("down")
    d.add_argument("--prefix", required=True)
    r = sub.add_parser("run")
    r.add_argument("experiment", type=Path)
    r.add_argument("--inventory", type=Path)
    r.add_argument("--runs", type=Path, default=Path("runs"))
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--mode", choices=["sim", "distributed", "faults"], required=True)
    for name in ("resumed", "times"):
        sub.add_parser(name).add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "up":
        print(up(args.prefix, args.image, args.into))
    elif args.command == "down":
        down(args.prefix)
    elif args.command == "run":
        print(run(args.experiment, args.inventory, args.runs, args.seed, args.mode))
    elif args.command == "times":
        times(args.runs)
    else:
        resumed(args.runs)


if __name__ == "__main__":
    main()
