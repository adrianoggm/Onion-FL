#!/usr/bin/env python3
"""Run an end-to-end SWELL federated demo from an existing manifest.

This launcher uses the current module layout:
  - `flower_basic.servers.swell`
  - `flower_basic.clients.fog_bridge_swell`
  - `flower_basic.brokers.fog`
  - `flower_basic.clients.swell`
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LaunchSpec:
    role: str
    cmd: list[str]


def read_manifest(path: str | os.PathLike) -> dict:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _base_env(repo_root: Path) -> dict[str, str]:
    py_path = str(repo_root / "src")
    if os.getenv("PYTHONPATH"):
        py_path = py_path + os.pathsep + os.getenv("PYTHONPATH")
    return {"PYTHONPATH": py_path}


def _has_non_empty_train_split(split_path: Path) -> bool:
    if not split_path.exists():
        return False
    try:
        arr = np.load(split_path, allow_pickle=True)
        return bool(arr["X"].shape[0] > 0)
    except Exception:
        return False


def _select_manifest_clients(
    manifest_path: str | os.PathLike,
    manifest: dict,
    max_clients_per_node: int,
) -> dict[str, list[tuple[str, Path]]]:
    base = Path(manifest_path).parent
    nodes = manifest.get("nodes", {})
    clients_map = manifest.get("clients", {})
    global_subjects = manifest.get("global_subjects", {})
    split_strategy = (
        manifest.get("config", {}).get("split", {}).get("strategy", "global")
    )

    if split_strategy == "per_subject":
        train_subjects = {
            str(subject_id) for subject_id in global_subjects.get("all", [])
        }
    else:
        train_subjects = {
            str(subject_id) for subject_id in global_subjects.get("train", [])
        }

    selected: dict[str, list[tuple[str, Path]]] = {}
    for node_id in nodes.keys():
        node_clients: list[tuple[str, Path]] = []
        manifest_clients = clients_map.get(node_id, {})

        if manifest_clients:
            for client_id, subject_id in manifest_clients.items():
                subject_str = str(subject_id)
                if subject_str not in train_subjects:
                    continue
                subject_dir = base / node_id / f"subject_{subject_str}"
                if _has_non_empty_train_split(subject_dir / "train.npz"):
                    node_clients.append((client_id, subject_dir))
        else:
            # Backward-compatible fallback for manifests without explicit client mapping.
            node_dir = base / node_id
            if _has_non_empty_train_split(node_dir / "train.npz"):
                node_clients.append((f"{node_id}_client_0", node_dir))

        if max_clients_per_node > 0:
            node_clients = node_clients[:max_clients_per_node]
        selected[node_id] = node_clients

    return selected


def build_launch_plan(
    manifest_path: str | os.PathLike,
    rounds: int,
    server_addr: str,
    max_clients_per_node: int,
    k_per_region: int,
    mqtt_broker: str,
    mqtt_port: int,
    topic_updates: str,
    topic_partial: str,
    topic_global: str,
    python_exec: str | None = None,
) -> tuple[list[LaunchSpec], dict[str, list[tuple[str, Path]]]]:
    manifest = read_manifest(manifest_path)
    input_dim = manifest.get("meta", {}).get("n_features")
    if input_dim is None:
        raise RuntimeError(
            "manifest.meta.n_features not found; cannot infer input dimension"
        )

    selected_clients = _select_manifest_clients(
        manifest_path, manifest, max_clients_per_node=max_clients_per_node
    )
    active_nodes = {
        node_id: clients for node_id, clients in selected_clients.items() if clients
    }
    if not active_nodes:
        raise RuntimeError("No trainable clients found in manifest")

    python_exec = python_exec or sys.executable
    commands: list[LaunchSpec] = []

    server_cmd = [
        python_exec,
        "-m",
        "flower_basic.servers.swell",
        "--server_addr",
        server_addr,
        "--input_dim",
        str(int(input_dim)),
        "--rounds",
        str(rounds),
        "--mqtt-broker",
        mqtt_broker,
        "--mqtt-port",
        str(mqtt_port),
        "--topic-global",
        topic_global,
        "--min-fit-clients",
        str(len(active_nodes)),
        "--min-available-clients",
        str(len(active_nodes)),
        "--manifest",
        str(manifest_path),
    ]
    commands.append(LaunchSpec(role="server", cmd=server_cmd))

    for node_id in active_nodes.keys():
        bridge_cmd = [
            python_exec,
            "-m",
            "flower_basic.clients.fog_bridge_swell",
            "--server",
            server_addr,
            "--input_dim",
            str(int(input_dim)),
            "--region",
            node_id,
            "--mqtt-broker",
            mqtt_broker,
            "--mqtt-port",
            str(mqtt_port),
            "--topic-partial",
            topic_partial,
        ]
        commands.append(LaunchSpec(role=f"fog_bridge_{node_id}", cmd=bridge_cmd))

    k_map = {
        node_id: max(1, min(k_per_region, len(clients)))
        for node_id, clients in active_nodes.items()
    }
    broker_cmd = [
        python_exec,
        "-m",
        "flower_basic.brokers.fog",
        "--mqtt-broker",
        mqtt_broker,
        "--mqtt-port",
        str(mqtt_port),
        "--topic-updates",
        topic_updates,
        "--topic-partial",
        topic_partial,
        "--topic-global",
        topic_global,
        "--k",
        str(next(iter(k_map.values()))),
    ]
    if len(set(k_map.values())) > 1:
        broker_cmd.extend(["--k-map", json.dumps(k_map)])
    commands.append(LaunchSpec(role="broker", cmd=broker_cmd))

    client_index = 0
    for node_id, clients in active_nodes.items():
        for client_id, node_dir in clients:
            client_cmd = [
                python_exec,
                "-m",
                "flower_basic.clients.swell",
                "--node_dir",
                str(node_dir),
                "--region",
                node_id,
                "--rounds",
                str(rounds),
                "--mqtt-broker",
                mqtt_broker,
                "--mqtt-port",
                str(mqtt_port),
                "--topic-updates",
                topic_updates,
                "--topic-global",
                topic_global,
                "--client-index",
                str(client_index),
                "--client-id",
                client_id,
            ]
            commands.append(LaunchSpec(role=f"client_{client_id}", cmd=client_cmd))
            client_index += 1

    return commands, active_nodes


def start_proc(
    spec: LaunchSpec, cwd: str | None = None, extra_env: dict[str, str] | None = None
) -> subprocess.Popen:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"[LAUNCH] {spec.role}: {' '.join(spec.cmd)}")
    return subprocess.Popen(spec.cmd, cwd=cwd, env=env)


def _shutdown_processes(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

    for proc in procs:
        try:
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SWELL federated demo")
    ap.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.json produced by prepare_swell_federated.py",
    )
    ap.add_argument(
        "--rounds", type=int, default=3, help="Federated rounds on the server"
    )
    ap.add_argument(
        "--server-addr",
        default="localhost:8080",
        help="Flower server address for fog bridges",
    )
    ap.add_argument(
        "--clients-per-node",
        type=int,
        default=1,
        help="Maximum number of manifest clients to launch per fog node (0 = all)",
    )
    ap.add_argument(
        "--k-per-region",
        type=int,
        default=1,
        help="Broker aggregation threshold per active region",
    )
    ap.add_argument("--mqtt-broker", default=os.getenv("MQTT_BROKER", "localhost"))
    ap.add_argument(
        "--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883"))
    )
    ap.add_argument(
        "--topic-updates", default=os.getenv("MQTT_TOPIC_UPDATES", "fl/updates")
    )
    ap.add_argument(
        "--topic-partial", default=os.getenv("MQTT_TOPIC_PARTIAL", "fl/partial")
    )
    ap.add_argument(
        "--topic-global", default=os.getenv("MQTT_TOPIC_GLOBAL", "fl/global_model")
    )
    args = ap.parse_args()

    repo_root = _repo_root()
    commands, active_nodes = build_launch_plan(
        manifest_path=args.manifest,
        rounds=args.rounds,
        server_addr=args.server_addr,
        max_clients_per_node=args.clients_per_node,
        k_per_region=args.k_per_region,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        topic_updates=args.topic_updates,
        topic_partial=args.topic_partial,
        topic_global=args.topic_global,
    )
    env = _base_env(repo_root)

    procs: list[subprocess.Popen] = []
    try:
        for spec in commands:
            procs.append(start_proc(spec, cwd=str(repo_root), extra_env=env))
            time.sleep(1.0)

        total_clients = sum(len(clients) for clients in active_nodes.values())
        print("\n[RUNNING] SWELL federated demo is running. Press Ctrl+C to stop.")
        print(
            f"[INFO] Active fog nodes={len(active_nodes)}, launched clients={total_clients}"
        )

        server_proc = procs[0] if procs else None
        while True:
            time.sleep(0.5)
            if server_proc and server_proc.poll() is not None:
                print(
                    f"[INFO] Server exited with code {server_proc.returncode}; stopping remaining processes..."
                )
                break
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping all processes...")
    finally:
        _shutdown_processes(procs)


if __name__ == "__main__":
    main()
