#!/usr/bin/env python3
"""Demo SWEET federated learning setup.

Similar to run_swell_federated_demo.py but for SWEET dataset.

Usage:
    python scripts/run_sweet_federated_demo.py \\
        --config configs/sweet_federated.example.yaml \\
        --num-rounds 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Add src to path
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def main():
    parser = argparse.ArgumentParser(description="Run SWEET federated learning demo")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML/JSON config file",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of federated rounds",
    )
    parser.add_argument(
        "--mqtt-broker",
        default="localhost",
        help="MQTT broker address",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        help="MQTT broker port",
    )
    parser.add_argument(
        "--enable-telemetry",
        action="store_true",
        help="Enable OpenTelemetry",
    )
    parser.add_argument(
        "--enable-prometheus",
        action="store_true",
        help="Enable Prometheus metrics",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SWEET Federated Learning Demo")
    print("=" * 80)
    
    # Step 1: Prepare data splits
    print("\n[1/4] Preparing federated data splits...")
    prep_cmd = [
        sys.executable,
        "scripts/prepare_sweet_federated.py",
        "--config",
        args.config,
    ]
    
    result = subprocess.run(prep_cmd, capture_output=False)
    if result.returncode != 0:
        print("❌ Failed to prepare data splits")
        sys.exit(1)
    
    # Load manifest to get run directory
    config_path = Path(args.config)
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml
        config_data = yaml.safe_load(config_path.read_text())
    else:
        config_data = json.loads(config_path.read_text())
    
    federation = config_data.get("federation", {})
    output_dir = federation.get("output_dir", "federated_runs/sweet")
    run_name = federation.get("run_name", f"run_{config_data.get('split', {}).get('seed', 42)}")
    run_dir = Path(output_dir) / run_name
    
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        sys.exit(1)
    
    manifest = json.loads(manifest_path.read_text())
    input_dim = manifest["num_features"]
    
    # Step 2: Start MQTT broker check
    print("\n[2/4] Checking MQTT broker...")
    print(f"  Expected broker: {args.mqtt_broker}:{args.mqtt_port}")
    print("  Make sure MQTT broker is running (e.g., mosquitto)")
    time.sleep(2)
    
    # Step 3: Start server
    print("\n[3/4] Starting SWEET federated server...")
    
    # Use first node for test evaluation
    first_node = list(manifest["nodes"].keys())[0]
    
    server_cmd = [
        sys.executable,
        "-m",
        "flower_basic.servers.sweet",
        "--run-dir",
        str(run_dir),
        "--test-node-id",
        first_node,
        "--input-dim",
        str(input_dim),
        "--mqtt-broker",
        args.mqtt_broker,
        "--mqtt-port",
        str(args.mqtt_port),
        "--min-clients",
        str(len(manifest["nodes"])),
    ]
    
    if args.enable_telemetry:
        server_cmd.append("--enable-telemetry")
    if args.enable_prometheus:
        server_cmd.append("--enable-prometheus")
    
    print(f"  Command: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(server_cmd)
    time.sleep(3)
    
    # Step 4: Start clients
    print("\n[4/4] Starting SWEET federated clients...")
    client_procs = []
    
    for node_id in manifest["nodes"].keys():
        client_cmd = [
            sys.executable,
            "-m",
            "flower_basic.clients.sweet",
            "--client-id",
            f"client_{node_id}",
            "--run-dir",
            str(run_dir),
            "--node-id",
            node_id,
            "--input-dim",
            str(input_dim),
            "--mqtt-broker",
            args.mqtt_broker,
            "--mqtt-port",
            str(args.mqtt_port),
        ]
        
        if args.enable_telemetry:
            client_cmd.append("--enable-telemetry")
        if args.enable_prometheus:
            client_cmd.append("--enable-prometheus")
        
        print(f"  Starting client for {node_id}...")
        proc = subprocess.Popen(client_cmd)
        client_procs.append(proc)
        time.sleep(1)
    
    print("\n" + "=" * 80)
    print("✓ SWEET Federated Learning Demo Running")
    print("=" * 80)
    print(f"  Run directory: {run_dir}")
    print(f"  Nodes: {len(manifest['nodes'])}")
    print(f"  Features: {input_dim}")
    print(f"  Target rounds: {args.num_rounds}")
    print("\nPress Ctrl+C to stop...")
    
    try:
        # Wait for completion
        server_proc.wait()
    except KeyboardInterrupt:
        print("\n\nStopping federated learning...")
        server_proc.terminate()
        for proc in client_procs:
            proc.terminate()
        
        time.sleep(2)
        
        server_proc.kill()
        for proc in client_procs:
            proc.kill()
        
        print("✓ All processes stopped")


if __name__ == "__main__":
    main()
