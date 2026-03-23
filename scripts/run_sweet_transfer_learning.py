#!/usr/bin/env python3
"""
Complete transfer learning workflow for SWEET:
1. Train baseline model on selection1 (102 subjects)
2. Prepare federated splits from selection2 (140 subjects)
3. Run federated fine-tuning with pre-trained model

Usage:
    python scripts/run_sweet_transfer_learning.py [--skip-baseline] [--skip-prep]

Options:
    --skip-baseline: Skip baseline training (use existing model)
    --skip-prep: Skip data preparation (use existing splits)
    --rounds N: Number of federated rounds (default: 10)
    --mqtt-broker HOST: MQTT broker address (default: localhost)
    --mqtt-port PORT: MQTT broker port (default: 1883)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'=' * 80}")
    print(description)
    print(f"{'=' * 80}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False

    print(f"\n✅ Success: {description}")
    return True


def check_prerequisites() -> bool:
    """Check if required data directories exist."""
    selection1_dir = Path("data/SWEET/selection1/users")
    selection2_dir = Path("data/SWEET/selection2/users")

    if not selection1_dir.exists():
        print(f"❌ Missing selection1 data: {selection1_dir}")
        print("   Please extract selection1 first")
        return False

    if not selection2_dir.exists():
        print(f"❌ Missing selection2 data: {selection2_dir}")
        print("   Run: python scripts/extract_sweet_selection2.py")
        return False

    print("✅ Data directories found")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="SWEET Transfer Learning: Selection1 → Selection2 Federated Fine-tuning"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline training on selection1",
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip federated data preparation",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated fine-tuning rounds",
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
        help="Enable OpenTelemetry tracing",
    )
    parser.add_argument(
        "--enable-prometheus",
        action="store_true",
        help="Enable Prometheus metrics",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("SWEET Transfer Learning Workflow")
    print("=" * 80)
    print("\nPhase 1: Pre-training on selection1 (102 subjects)")
    print("Phase 2: Federated fine-tuning on selection2 (140 subjects)")
    print("=" * 80)

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    # Phase 1: Train baseline model on selection1
    if not args.skip_baseline:
        baseline_model_path = Path("baseline_models/sweet/xgboost_tuned_model.json")
        if baseline_model_path.exists():
            print(f"\n⚠️  Baseline model already exists: {baseline_model_path}")
            response = input("Retrain? [y/N]: ").strip().lower()
            if response != "y":
                print("Skipping baseline training")
                args.skip_baseline = True

        if not args.skip_baseline:
            cmd = [sys.executable, "scripts/train_sweet_baseline_selection1.py"]
            if not run_command(cmd, "Phase 1: Train Baseline Model (selection1)"):
                sys.exit(1)
    else:
        print("\n[SKIPPED] Phase 1: Baseline training")
        baseline_model_path = Path("baseline_models/sweet/xgboost_tuned_model.json")
        if not baseline_model_path.exists():
            print(f"❌ Baseline model not found: {baseline_model_path}")
            print("   Remove --skip-baseline to train it")
            sys.exit(1)

    # Phase 2: Prepare federated splits from selection2
    if not args.skip_prep:
        run_dir = Path("federated_runs/sweet/transfer_selection2")
        if run_dir.exists():
            print(f"\n⚠️  Federated run directory already exists: {run_dir}")
            response = input("Recreate splits? [y/N]: ").strip().lower()
            if response != "y":
                print("Skipping data preparation")
                args.skip_prep = True

        if not args.skip_prep:
            cmd = [sys.executable, "scripts/prepare_sweet_federated_transfer.py"]
            if not run_command(cmd, "Phase 2: Prepare Federated Splits (selection2)"):
                sys.exit(1)
    else:
        print("\n[SKIPPED] Phase 2: Federated data preparation")
        manifest_path = Path("federated_runs/sweet/transfer_selection2/manifest.json")
        if not manifest_path.exists():
            print(f"❌ Manifest not found: {manifest_path}")
            print("   Remove --skip-prep to prepare splits")
            sys.exit(1)

    # Phase 3: Run federated fine-tuning
    print("\n" + "=" * 80)
    print("Phase 3: Federated Fine-Tuning (selection2)")
    print("=" * 80)
    print(f"\nFederated rounds: {args.rounds}")
    print(f"MQTT broker: {args.mqtt_broker}:{args.mqtt_port}")
    print(f"Telemetry: {'enabled' if args.enable_telemetry else 'disabled'}")
    print(f"Prometheus: {'enabled' if args.enable_prometheus else 'disabled'}")

    print("\n⚠️  Important: Make sure MQTT broker is running!")
    print("   Example: mosquitto -c mosquitto.conf")
    print("\nStarting in 5 seconds... (Ctrl+C to cancel)")

    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)

    # Run federated demo
    cmd = [
        sys.executable,
        "scripts/run_sweet_federated_demo.py",
        "--config",
        "configs/sweet_federated_transfer.yaml",
        "--num-rounds",
        str(args.rounds),
        "--mqtt-broker",
        args.mqtt_broker,
        "--mqtt-port",
        str(args.mqtt_port),
    ]

    if args.enable_telemetry:
        cmd.append("--enable-telemetry")
    if args.enable_prometheus:
        cmd.append("--enable-prometheus")

    if not run_command(cmd, "Phase 3: Federated Fine-Tuning"):
        sys.exit(1)

    # Success summary
    print("\n" + "=" * 80)
    print("✅ SWEET Transfer Learning Complete!")
    print("=" * 80)
    print("\nResults:")
    print("  Baseline model: baseline_models/sweet/xgboost_tuned_model.json")
    print("  Federated run: federated_runs/sweet/transfer_selection2/")
    print("  Manifest: federated_runs/sweet/transfer_selection2/manifest.json")
    print("\nNext steps:")
    print("  - Check federated_runs/sweet/transfer_selection2/ for results")
    print("  - Compare selection1 baseline vs selection2 fine-tuned performance")
    print("  - Analyze per-node metrics and convergence")
    print("=" * 80)


if __name__ == "__main__":
    main()
