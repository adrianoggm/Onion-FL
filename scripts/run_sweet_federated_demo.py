#!/usr/bin/env python3
"""Run a SWEET federated demo using the current 3-layer architecture."""

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

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class LaunchSpec:
    role: str
    cmd: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _base_env(repo_root: Path) -> dict[str, str]:
    py_path = str(repo_root / "src")
    if os.getenv("PYTHONPATH"):
        py_path = py_path + os.pathsep + os.getenv("PYTHONPATH")
    return {"PYTHONPATH": py_path}


def _load_config(config_path: str | os.PathLike) -> dict:
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config files")
        return yaml.safe_load(text)
    return json.loads(text)


def _manifest_path_from_config(config_data: dict) -> Path:
    federation = config_data.get("federation", {})
    output_dir = federation.get("output_dir", "federated_runs/sweet")
    run_name = federation.get(
        "run_name", f"run_{config_data.get('split', {}).get('seed', 42)}"
    )
    return Path(output_dir) / run_name / "manifest.json"


def _has_non_empty_train_split(split_path: Path) -> bool:
    if not split_path.exists():
        return False
    try:
        arr = np.load(split_path, allow_pickle=True)
        return bool(arr["X"].shape[0] > 0)
    except Exception:
        return False


def _select_manifest_clients(
    manifest_path: str | os.PathLike, manifest: dict
) -> dict[str, list[str]]:
    base = Path(manifest_path).parent
    selected: dict[str, list[str]] = {}
    for node_id, clients_dict in manifest.get("clients", {}).items():
        subjects: list[str] = []
        for subject_id in clients_dict.values():
            subject_str = str(subject_id)
            train_file = base / node_id / f"subject_{subject_str}" / "train.npz"
            if _has_non_empty_train_split(train_file):
                subjects.append(subject_str)
        selected[node_id] = subjects
    return selected


def build_launch_plan(
    manifest_path: str | os.PathLike,
    config_data: dict,
    num_rounds: int,
    server_addr: str,
    mqtt_broker: str,
    mqtt_port: int,
    enable_telemetry: bool,
    enable_prometheus: bool,
    hidden_dims: list[int] | None = None,
    num_classes: int | None = None,
    python_exec: str | None = None,
) -> tuple[list[LaunchSpec], dict[str, list[str]], int]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    input_dim = manifest.get("meta", {}).get("n_features")
    if input_dim is None:
        raise RuntimeError("manifest.meta.n_features not found")

    label_strategy = config_data.get("dataset", {}).get("label_strategy", "ordinal")
    inferred_num_classes = 2 if label_strategy == "binary" else 3
    num_classes = num_classes or inferred_num_classes
    hidden_dims = hidden_dims or [64, 32]

    selected_clients = _select_manifest_clients(manifest_path, manifest)
    active_nodes = {
        node_id: subject_ids
        for node_id, subject_ids in selected_clients.items()
        if subject_ids
    }
    if not active_nodes:
        raise RuntimeError("No trainable SWEET clients found in manifest")

    python_exec = python_exec or sys.executable
    commands: list[LaunchSpec] = []

    server_cmd = [
        python_exec,
        "-m",
        "flower_basic.servers.sweet",
        "--input-dim",
        str(int(input_dim)),
        "--hidden-dims",
        *[str(dim) for dim in hidden_dims],
        "--num-classes",
        str(num_classes),
        "--rounds",
        str(num_rounds),
        "--server-addr",
        server_addr,
        "--mqtt-broker",
        mqtt_broker,
        "--mqtt-port",
        str(mqtt_port),
        "--manifest",
        str(manifest_path),
        "--min-fit-clients",
        str(len(active_nodes)),
        "--min-available-clients",
        str(len(active_nodes)),
    ]
    commands.append(LaunchSpec(role="server", cmd=server_cmd))

    k_map = {node_id: len(subject_ids) for node_id, subject_ids in active_nodes.items()}
    broker_cmd = [
        python_exec,
        "-m",
        "flower_basic.brokers.sweet_fog",
        "--mqtt-broker",
        mqtt_broker,
        "--mqtt-port",
        str(mqtt_port),
        "--k",
        str(next(iter(k_map.values()))),
    ]
    if len(set(k_map.values())) > 1:
        broker_cmd.extend(["--k-map", json.dumps(k_map)])
    commands.append(LaunchSpec(role="broker", cmd=broker_cmd))

    for node_id in active_nodes.keys():
        bridge_cmd = [
            python_exec,
            "-m",
            "flower_basic.clients.fog_bridge_sweet",
            "--server",
            server_addr,
            "--region",
            node_id,
            "--input-dim",
            str(int(input_dim)),
            "--hidden-dims",
            *[str(dim) for dim in hidden_dims],
            "--num-classes",
            str(num_classes),
            "--mqtt-broker",
            mqtt_broker,
            "--mqtt-port",
            str(mqtt_port),
        ]
        commands.append(LaunchSpec(role=f"fog_bridge_{node_id}", cmd=bridge_cmd))

    for node_id, subject_ids in active_nodes.items():
        node_dir = Path(manifest["output_dir"]) / node_id
        for subject_id in subject_ids:
            client_cmd = [
                python_exec,
                "-m",
                "flower_basic.clients.sweet",
                "--node-dir",
                str(node_dir),
                "--region",
                node_id,
                "--subject-id",
                subject_id,
                "--input-dim",
                str(int(input_dim)),
                "--hidden-dims",
                *[str(dim) for dim in hidden_dims],
                "--num-classes",
                str(num_classes),
                "--rounds",
                str(num_rounds),
                "--mqtt-broker",
                mqtt_broker,
                "--mqtt-port",
                str(mqtt_port),
            ]
            if enable_telemetry:
                client_cmd.append("--enable-telemetry")
            if enable_prometheus:
                client_cmd.append("--enable-prometheus")
            commands.append(
                LaunchSpec(role=f"client_{node_id}_{subject_id}", cmd=client_cmd)
            )

    total_clients = sum(len(subject_ids) for subject_ids in active_nodes.values())
    return commands, active_nodes, total_clients


def _run_prepare_step(config_path: str) -> None:
    prep_cmd = [
        sys.executable,
        "scripts/prepare_sweet_federated.py",
        "--config",
        config_path,
    ]
    result = subprocess.run(prep_cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("Failed to prepare SWEET federated splits")


def _start_proc(
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
    parser = argparse.ArgumentParser(description="Run SWEET federated learning demo")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of federated rounds",
    )
    parser.add_argument("--mqtt-broker", default="localhost", help="MQTT broker")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument(
        "--server-addr",
        default="localhost:8080",
        help="Flower server address for fog bridges",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden layer dimensions for the MLP",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override inferred number of output classes",
    )
    parser.add_argument(
        "--enable-telemetry",
        action="store_true",
        help="Enable telemetry in SWEET clients",
    )
    parser.add_argument(
        "--enable-prometheus",
        action="store_true",
        help="Enable Prometheus metrics in SWEET clients",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SWEET Federated Learning Demo")
    print("=" * 80)

    print("\n[1/4] Preparing federated data splits...")
    _run_prepare_step(args.config)

    config_data = _load_config(args.config)
    manifest_path = _manifest_path_from_config(config_data)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print("\n[2/4] Checking MQTT broker...")
    print(f"  Expected broker: {args.mqtt_broker}:{args.mqtt_port}")
    print("  Make sure MQTT broker is running (e.g., mosquitto)")
    time.sleep(2)

    print("\n[3/4] Building launch plan...")
    commands, active_nodes, total_clients = build_launch_plan(
        manifest_path=manifest_path,
        config_data=config_data,
        num_rounds=args.num_rounds,
        server_addr=args.server_addr,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        enable_telemetry=args.enable_telemetry,
        enable_prometheus=args.enable_prometheus,
        hidden_dims=args.hidden_dims,
        num_classes=args.num_classes,
    )

    print("\n[4/4] Starting SWEET federated processes...")
    repo_root = _repo_root()
    env = _base_env(repo_root)
    procs: list[subprocess.Popen] = []

    try:
        for spec in commands:
            procs.append(_start_proc(spec, cwd=str(repo_root), extra_env=env))
            time.sleep(1.0)

        print("\n" + "=" * 80)
        print("✓ SWEET Federated Learning Demo Running")
        print("=" * 80)
        print(f"  Manifest: {manifest_path}")
        print(f"  Active fog nodes: {len(active_nodes)}")
        print(f"  Total clients: {total_clients}")
        print(f"  Target rounds: {args.num_rounds}")
        print("\nPress Ctrl+C to stop...")

        server_proc = procs[0] if procs else None
        while True:
            time.sleep(0.5)
            if server_proc and server_proc.poll() is not None:
                print(
                    f"\n[INFO] Server exited with code {server_proc.returncode}; stopping remaining processes..."
                )
                break
    except KeyboardInterrupt:
        print("\n\nStopping federated learning...")
    finally:
        _shutdown_processes(procs)


if __name__ == "__main__":
    main()
