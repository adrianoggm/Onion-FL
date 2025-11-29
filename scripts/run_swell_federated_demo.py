#!/usr/bin/env python3
"""
Run an end-to-end SWELL federated demo using the existing fog architecture.

It launches:
  - Central Flower server for SWELL (publishes global model via MQTT)
  - Fog-Flower bridge client (receives MQTT partials, forwards to server)
  - Fog broker (aggregates K client updates per region)
  - N local clients per region reading prepared NPZ splits

Prerequisites:
  1) Prepare splits: `python scripts/prepare_swell_federated.py --config <config.json|yaml>`
  2) Start an MQTT broker (e.g., Mosquitto) at localhost:1883

Usage:
  python scripts/run_swell_federated_demo.py \
    --manifest federated_runs/swell/example_auto/manifest.json \
    --rounds 3 \
    --clients-per-node 1 \
    --k-per-region 1 \
    --mqtt-broker localhost \
    --mqtt-port 1883 \
    --topic-updates fl/updates \
    --topic-partial fl/partial \
    --topic-global fl/global_model

Notes:
  - Ensure `k-per-region` in this script matches K in src/flower_basic/broker_fog.py
    (or set K=1 there for a quick demo with one client per region).
  - Use Ctrl+C to stop all processes started by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np


def read_manifest(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def start_proc(cmd: List[str], cwd: str | None = None) -> subprocess.Popen:
    print(f"[LAUNCH] {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=cwd)


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
        "--clients-per-node",
        type=int,
        default=1,
        help="Number of local clients per fog node",
    )
    ap.add_argument(
        "--k-per-region",
        type=int,
        default=1,
        help="Broker aggregation K (informational; set same K in broker_fog.py)",
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

    repo_root = Path(__file__).resolve().parents[1]
    manifest = read_manifest(args.manifest)

    input_dim = manifest.get("meta", {}).get("n_features")
    if input_dim is None:
        raise RuntimeError(
            "manifest.meta.n_features not found; cannot infer input dimension"
        )

    # Compute node directories
    manifest_dir = Path(args.manifest).parent
    nodes = manifest.get("nodes", {})
    if not nodes:
        raise RuntimeError("No nodes in manifest")

    procs: List[subprocess.Popen] = []

    try:
        # Start central server
        server_cmd = [
            sys.executable,
            str(repo_root / "src/flower_basic/server_swell.py"),
            "--input_dim",
            str(input_dim),
            "--rounds",
            str(args.rounds),
            "--mqtt-broker",
            args.mqtt_broker,
            "--mqtt-port",
            str(args.mqtt_port),
            "--topic-global",
            args.topic_global,
        ]
        procs.append(start_proc(server_cmd))

        time.sleep(1.5)

        # Start fog bridge
        bridge_cmd = [
            sys.executable,
            str(repo_root / "src/flower_basic/fog_flower_client_swell.py"),
            "--input_dim",
            str(input_dim),
            "--mqtt-broker",
            args.mqtt_broker,
            "--mqtt-port",
            str(args.mqtt_port),
            "--topic-partial",
            args.topic_partial,
        ]
        procs.append(start_proc(bridge_cmd))

        time.sleep(1.0)

        # Start fog broker
        broker_cmd = [
            sys.executable,
            str(repo_root / "src/flower_basic/broker_fog.py"),
            "--k",
            str(args.k_per_region),
            "--mqtt-broker",
            args.mqtt_broker,
            "--mqtt-port",
            str(args.mqtt_port),
            "--topic-updates",
            args.topic_updates,
            "--topic-partial",
            args.topic_partial,
            "--topic-global",
            args.topic_global,
        ]
        procs.append(start_proc(broker_cmd))

        time.sleep(1.0)

        # Start local clients per node (skip nodes with empty train)
        for node_name in nodes.keys():
            node_dir = manifest_dir / node_name
            train_npz = node_dir / "train.npz"
            if not train_npz.exists():
                print(f"[WARN] Missing train.npz for {node_name}, skipping")
                continue
            try:
                arr = np.load(train_npz, allow_pickle=True)
                X = arr.get("X")
                if X is None or X.size == 0:
                    print(
                        f"[WARN] Empty train split for {node_name}, skipping client start"
                    )
                    continue
            except Exception as e:
                print(
                    f"[WARN] Could not inspect train.npz for {node_name}: {e}; skipping"
                )
                continue

            for i in range(args.clients_per_node):
                client_cmd = [
                    sys.executable,
                    str(repo_root / "src/flower_basic/swell_client.py"),
                    "--node_dir",
                    str(node_dir),
                    "--region",
                    node_name,
                    "--rounds",
                    str(args.rounds),
                    "--mqtt-broker",
                    args.mqtt_broker,
                    "--mqtt-port",
                    str(args.mqtt_port),
                    "--topic-updates",
                    args.topic_updates,
                    "--topic-global",
                    args.topic_global,
                ]
                procs.append(start_proc(client_cmd))

        print("\n[RUNNING] SWELL federated demo is running. Press Ctrl+C to stop.")
        print(
            f"[INFO] clients-per-node={args.clients_per_node}, remember to align K in broker_fog.py (current: {args.k_per_region} expected)"
        )
        # Wait for all processes; if any exits early, continue until Ctrl+C
        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping all processes...")
    finally:
        for p in procs:
            try:
                if p.poll() is None:
                    if os.name == "nt":
                        p.terminate()
                    else:
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass


if __name__ == "__main__":
    main()
