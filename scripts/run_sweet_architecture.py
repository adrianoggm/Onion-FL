#!/usr/bin/env python3
"""
Run SWEET federated learning from architecture configuration.

Similar to SWELL's run_architecture_from_config.py but tailored for SWEET dataset.

Usage:
    python scripts/run_sweet_architecture.py \\
        --config configs/sweet_architecture_5nodes.yaml \\
        --dispatch-config \\
        --launch

Flags:
    --config PATH: Path to architecture YAML configuration
    --dispatch-config: Generate manifest and federated splits
    --launch: Launch server and clients
    --manifest PATH: (Optional) Use existing manifest instead of generating
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
from typing import Any

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dispatch_config(config: dict[str, Any]) -> Path:
    """Generate federated splits from configuration.
    
    Returns:
        Path to generated manifest.json
    """
    print("\n" + "=" * 80)
    print("STEP 1: Dispatch Configuration & Generate Federated Splits")
    print("=" * 80)
    
    arch = config["federated_architecture"]
    dataset_cfg = arch["dataset"]
    
    # Create temporary YAML config for sweet_federated module
    fed_config = {
        "dataset": {
            "data_dir": dataset_cfg["data_dir"],
            "label_strategy": dataset_cfg.get("label_strategy", "ordinal_3class"),
            "elevated_threshold": dataset_cfg.get("elevated_threshold", 2.0),
            "min_samples_per_subject": dataset_cfg.get("min_samples_per_subject", 5),
        },
        "split": dataset_cfg["split"],
        "federation": dataset_cfg["federation"],
    }
    
    # Add pretrained model if configured
    if arch.get("pretrained", {}).get("enabled"):
        fed_config["transfer_learning"] = {
            "pretrained_model_path": arch["pretrained"].get("model_path"),
            "pretrained_scaler_path": arch["pretrained"].get("scaler_path"),
            "freeze_initial_weights": arch["pretrained"].get("freeze_initial_weights", False),
            "fine_tune_lr_multiplier": arch["pretrained"].get("fine_tune_lr_multiplier", 0.1),
        }
    
    # Write temporary config
    temp_config_path = Path("configs/.temp_sweet_federated.yaml")
    with open(temp_config_path, "w") as f:
        yaml.dump(fed_config, f, default_flow_style=False)
    
    print(f"[INFO] Temporary config written to: {temp_config_path}")
    print(f"[INFO] Generating federated splits for {dataset_cfg['federation']['num_fog_nodes']} fog nodes...")
    
    # Import and run materialization
    from flower_basic.datasets.sweet_federated import plan_and_materialize_sweet_federated
    
    try:
        manifest = plan_and_materialize_sweet_federated(str(temp_config_path))
        manifest_path = Path(manifest["output_dir"]) / "manifest.json"
        
        print(f"\n[SUCCESS] Manifest generated: {manifest_path}")
        print(f"  Nodes: {len(manifest['nodes'])}")
        print(f"  Features: {manifest['num_features']}")
        print(f"  Total subjects: {len(manifest.get('subjects_train', []))} train, "
              f"{len(manifest.get('subjects_val', []))} val, "
              f"{len(manifest.get('subjects_test', []))} test")
        
        # Clean up temp config
        temp_config_path.unlink()
        
        return manifest_path
        
    except Exception as e:
        print(f"\n[ERROR] Failed to generate federated splits: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def launch_federated_system(
    config: dict[str, Any],
    manifest_path: Path,
    mqtt_check: bool = True,
) -> list[subprocess.Popen]:
    """Launch SWEET federated system with 3-layer hierarchical architecture.
    
    Architecture (following SWELL pattern):
        1. Central Server (Flower) - Aggregates from fog bridges
        2. Fog Brokers (5) - Regional aggregators, one per fog node
        3. Fog Bridges (5) - Flower clients connecting brokers to server
        4. SWEET Clients (5) - Local training, publish to fog brokers
    
    Returns:
        List of running processes.
    """
    print("\n" + "=" * 80)
    print("STEP 2: Launch Federated System (3-Layer Architecture)")
    print("=" * 80)
    
    arch = config["federated_architecture"]
    orchestrator = arch["orchestrator"]
    mqtt_cfg = orchestrator["mqtt"]
    model_cfg = arch["model"]
    
    # Load manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    run_dir = Path(manifest["output_dir"])
    input_dim = manifest["num_features"]
    num_classes = model_cfg.get("num_classes", 3)
    hidden_dims = model_cfg.get("hidden_dims", [64, 32])
    rounds = orchestrator.get("rounds", 10)
    
    # Check MQTT broker
    if mqtt_check:
        print(f"\n[CHECK] MQTT Broker: {mqtt_cfg['broker']}:{mqtt_cfg['port']}")
        print("  ⚠️  IMPORTANT: MQTT broker MUST be running!")
        print("  Start mosquitto with:")
        print("    docker run -d -p 1883:1883 eclipse-mosquitto")
        print("\n  Waiting 5 seconds before starting... (Ctrl+C to cancel)")
        
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n\n[CANCELLED] Startup cancelled by user")
            sys.exit(0)
    
    procs = []
    
    # ========================================================================
    # LAYER 1: Central Server (Flower)
    # ========================================================================
    print(f"\n[LAUNCH] Layer 1: Starting SWEET Central Server...")
    
    server_cmd = [
        sys.executable,
        "-m",
        "flower_basic.servers.sweet",
        "--input-dim",
        str(input_dim),
        "--num-classes",
        str(num_classes),
        "--rounds",
        str(rounds),
        "--mqtt-broker",
        mqtt_cfg["broker"],
        "--mqtt-port",
        str(mqtt_cfg["port"]),
        "--manifest",
        str(manifest_path),
        "--min-fit-clients",
        str(len(manifest["nodes"])),
        "--server-addr",
        "0.0.0.0:8080",
    ]
    
    # Add hidden dims
    server_cmd.extend(["--hidden-dims"] + [str(d) for d in hidden_dims])
    
    print(f"  Server address: 0.0.0.0:8080")
    print(f"  Min fog bridges: {len(manifest['nodes'])}")
    server_proc = subprocess.Popen(server_cmd)
    procs.append(server_proc)
    time.sleep(3)
    
    # ========================================================================
    # LAYER 2A: Fog Brokers (Regional Aggregators)
    # ========================================================================
    print(f"\n[LAUNCH] Layer 2A: Starting {len(manifest['nodes'])} Fog Brokers...")
    
    # K=1 because we have 1 client per fog node
    k_map = {node_id: 1 for node_id in manifest["nodes"].keys()}
    k_map_json = json.dumps(k_map)
    
    broker_cmd = [
        sys.executable,
        "-m",
        "flower_basic.brokers.sweet_fog",
        "--mqtt-broker",
        mqtt_cfg["broker"],
        "--mqtt-port",
        str(mqtt_cfg["port"]),
        "--k",
        "1",  # Default K
        "--k-map",
        k_map_json,  # Per-region K values
    ]
    
    print(f"  Broker K map: {k_map}")
    print(f"  Topics: fl/updates (in), fl/partial (out)")
    broker_proc = subprocess.Popen(broker_cmd)
    procs.append(broker_proc)
    time.sleep(2)
    
    # ========================================================================
    # LAYER 2B: Fog Bridges (Flower Clients → Server)
    # ========================================================================
    print(f"\n[LAUNCH] Layer 2B: Starting {len(manifest['nodes'])} Fog Bridges...")
    
    for node_id in manifest["nodes"].keys():
        bridge_cmd = [
            sys.executable,
            "-m",
            "flower_basic.clients.fog_bridge_sweet",
            "--server",
            "localhost:8080",
            "--region",
            node_id,
            "--input-dim",
            str(input_dim),
            "--num-classes",
            str(num_classes),
            "--mqtt-broker",
            mqtt_cfg["broker"],
            "--mqtt-port",
            str(mqtt_cfg["port"]),
        ]
        
        # Add hidden dims
        bridge_cmd.extend(["--hidden-dims"] + [str(d) for d in hidden_dims])
        
        print(f"  Starting fog bridge for {node_id}...")
        bridge_proc = subprocess.Popen(bridge_cmd)
        procs.append(bridge_proc)
        time.sleep(0.5)
    
    time.sleep(2)  # Give bridges time to connect to server
    
    # ========================================================================
    # LAYER 3: SWEET Clients (Local Training)
    # ========================================================================
    print(f"\n[LAUNCH] Layer 3: Starting {len(manifest['nodes'])} SWEET Clients...")
    
    for node_id, node_info in manifest["nodes"].items():
        node_dir = run_dir / node_id
        
        client_cmd = [
            sys.executable,
            "-m",
            "flower_basic.clients.sweet",
            "--node-dir",
            str(node_dir),
            "--region",
            node_id,
            "--input-dim",
            str(input_dim),
            "--num-classes",
            str(num_classes),
            "--lr",
            str(model_cfg.get("lr", 0.001)),
            "--batch-size",
            str(model_cfg.get("batch_size", 32)),
            "--local-epochs",
            str(model_cfg.get("local_epochs", 5)),
            "--rounds",
            str(rounds),
            "--mqtt-broker",
            mqtt_cfg["broker"],
            "--mqtt-port",
            str(mqtt_cfg["port"]),
        ]
        
        # Add hidden dims
        client_cmd.extend(["--hidden-dims"] + [str(d) for d in hidden_dims])
        
        print(f"  Starting client for {node_id}...")
        client_proc = subprocess.Popen(client_cmd)
        procs.append(client_proc)
        time.sleep(0.5)
    
    print(f"\n[RUNNING] Federated system running with {len(procs)} processes:")
    print(f"  - 1 Central Server")
    print(f"  - 1 Fog Broker (aggregates {len(manifest['nodes'])} regions)")
    print(f"  - {len(manifest['nodes'])} Fog Bridges")
    print(f"  - {len(manifest['nodes'])} SWEET Clients")
    print(f"\n  Architecture: Clients → Fog Broker → Fog Bridges → Central Server")
    print("  Press Ctrl+C to stop all processes")
    
    return procs


def main():
    parser = argparse.ArgumentParser(
        description="Run SWEET federated learning from architecture configuration"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to architecture YAML configuration",
    )
    parser.add_argument(
        "--manifest",
        help="Path to existing manifest.json (skip data preparation)",
    )
    parser.add_argument(
        "--dispatch-config",
        action="store_true",
        help="Generate manifest and federated splits from config",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch server and clients",
    )
    parser.add_argument(
        "--skip-mqtt-check",
        action="store_true",
        help="Skip MQTT broker check",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SWEET Federated Learning - Architecture-based Execution")
    print("=" * 80)
    
    # Load configuration
    config = load_config(args.config)
    print(f"\n[CONFIG] Loaded: {args.config}")
    
    arch = config["federated_architecture"]
    print(f"  Workflow: {arch['workflow']}")
    print(f"  Fog nodes: {arch['dataset']['federation']['num_fog_nodes']}")
    print(f"  Rounds: {arch['orchestrator']['rounds']}")
    
    manifest_path = None
    
    # Step 1: Dispatch configuration (generate splits)
    if args.dispatch_config:
        manifest_path = dispatch_config(config)
    elif args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"\n[ERROR] Manifest not found: {manifest_path}")
            sys.exit(1)
        print(f"\n[INFO] Using existing manifest: {manifest_path}")
    else:
        # Try to find manifest from config
        dataset_cfg = arch["dataset"]
        run_name = dataset_cfg.get("run_name", "auto_5nodes")
        output_dir = dataset_cfg.get("output_dir", "federated_runs/sweet")
        manifest_path = Path(output_dir) / run_name / "manifest.json"
        
        if not manifest_path.exists():
            print(f"\n[ERROR] No manifest found at {manifest_path}")
            print("  Use --dispatch-config to generate splits or --manifest to specify path")
            sys.exit(1)
        
        print(f"\n[INFO] Using manifest: {manifest_path}")
    
    # Step 2: Launch federated system
    if args.launch:
        if manifest_path is None:
            print("\n[ERROR] Cannot launch without manifest. Use --dispatch-config or --manifest")
            sys.exit(1)
        
        procs = launch_federated_system(
            config,
            manifest_path,
            mqtt_check=not args.skip_mqtt_check,
        )
        
        # Wait for processes
        try:
            while True:
                time.sleep(1)
                # Check if any process died
                for p in procs:
                    if p.poll() is not None:
                        print(f"\n[WARNING] Process {p.pid} exited with code {p.returncode}")
        except KeyboardInterrupt:
            print("\n\n[SHUTDOWN] Stopping all processes...")
            for p in procs:
                try:
                    if p.poll() is None:
                        p.terminate()
                        p.wait(timeout=5)
                except Exception:
                    pass
            print("[SHUTDOWN] All processes stopped")
    
    print("\n" + "=" * 80)
    print("✅ Done")
    print("=" * 80)


if __name__ == "__main__":
    main()
