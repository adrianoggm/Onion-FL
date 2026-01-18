#!/usr/bin/env python3
"""
Validate SWELL federated setup before running 10 executions.

Checks:
  1. SWELL dataset exists and is readable
  2. MQTT broker is accessible
  3. All required modules are available
  4. Configuration file is valid
  5. Scripts and servers exist
"""

from __future__ import annotations

import sys
import subprocess
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def check_file_exists(path: str | Path, description: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    p = Path(path)
    if p.exists():
        return True, f"✅ {description}: {p}"
    return False, f"❌ {description}: NOT FOUND - {p}"


def check_directory_exists(path: str | Path, description: str) -> Tuple[bool, str]:
    """Check if a directory exists."""
    p = Path(path)
    if p.is_dir():
        return True, f"✅ {description}: {p}"
    return False, f"❌ {description}: NOT FOUND - {p}"


def check_swell_dataset() -> Tuple[bool, List[str]]:
    """Check if SWELL dataset is accessible."""
    messages = []
    repo_root = Path(__file__).resolve().parent

    # Check base directory
    swell_dir = repo_root / "data" / "SWELL"
    if not swell_dir.exists():
        messages.append(f"❌ SWELL directory not found: {swell_dir}")
        return False, messages

    messages.append(f"✅ SWELL directory found: {swell_dir}")

    # Try to load dataset
    try:
        sys.path.insert(0, str(repo_root / "src"))
        from flower_basic.datasets.swell import load_swell_dataset

        X_train, X_test, y_train, y_test, info = load_swell_dataset(
            data_dir=str(swell_dir), modalities=["computer", "facial", "posture", "physiology"]
        )
        n_subjects = len(np.unique(info["subject_ids"]))
        n_samples = len(X_train) + len(X_test)
        messages.append(f"✅ SWELL dataset loaded successfully")
        messages.append(f"   - Subjects: {n_subjects}")
        messages.append(f"   - Total samples: {n_samples}")
        messages.append(f"   - Feature dimension: {X_train.shape[1]}")
        return True, messages

    except Exception as e:
        messages.append(f"❌ Failed to load SWELL dataset: {e}")
        return False, messages


def check_mqtt_broker(broker: str = "localhost", port: int = 1883) -> Tuple[bool, str]:
    """Check if MQTT broker is running."""
    try:
        import paho.mqtt.client as mqtt

        client = mqtt.Client()
        client.connect(broker, port, keepalive=2)
        client.disconnect()
        return True, f"✅ MQTT broker accessible at {broker}:{port}"

    except Exception as e:
        return (
            False,
            f"❌ MQTT broker not accessible at {broker}:{port}\n"
            f"   Error: {e}\n"
            f"   Start with: mosquitto -c mosquitto.conf",
        )


def check_python_modules() -> Tuple[bool, List[str]]:
    """Check if required Python modules are available."""
    messages = []
    modules = [
        "flower",
        "paho.mqtt.client",
        "numpy",
        "scipy",
        "pyyaml",
        "scikit-learn",
    ]

    for module in modules:
        try:
            __import__(module)
            messages.append(f"✅ {module}")
        except ImportError:
            messages.append(f"❌ {module} - NOT INSTALLED")
            return False, messages

    return True, messages


def check_config_file(config_path: str | Path) -> Tuple[bool, List[str]]:
    """Validate configuration file."""
    messages = []
    p = Path(config_path)

    if not p.exists():
        messages.append(f"❌ Config file not found: {p}")
        return False, messages

    try:
        if p.suffix in [".yaml", ".yml"]:
            import yaml

            config = yaml.safe_load(p.read_text(encoding="utf-8"))
        else:
            config = json.loads(p.read_text(encoding="utf-8"))

        messages.append(f"✅ Config file valid: {p}")

        # Check key sections
        if "dataset" in config:
            messages.append(f"   ✅ Dataset config found")
        if "split" in config:
            messages.append(f"   ✅ Split config found")
        if "federation" in config:
            fed = config["federation"]
            messages.append(f"   ✅ Federation config found")
            if "manual_assignments" in fed:
                num_nodes = len(fed["manual_assignments"])
                total_subjects = sum(
                    len(v) for v in fed["manual_assignments"].values()
                )
                messages.append(
                    f"      - {num_nodes} Fog nodes with {total_subjects} subjects total"
                )

        return True, messages

    except Exception as e:
        messages.append(f"❌ Config validation failed: {e}")
        return False, messages


def check_scripts_exist(repo_root: Path) -> Tuple[bool, List[str]]:
    """Check if required scripts exist."""
    messages = []
    scripts = [
        "src/flower_basic/servers/swell.py",
        "src/flower_basic/clients/swell.py",
        "src/flower_basic/brokers/fog.py",
        "src/flower_basic/clients/fog_bridge_swell.py",
        "scripts/prepare_swell_federated.py",
    ]

    all_exist = True
    for script in scripts:
        p = repo_root / script
        if p.exists():
            messages.append(f"✅ {script}")
        else:
            messages.append(f"❌ {script} - NOT FOUND")
            all_exist = False

    return all_exist, messages


def main() -> None:
    repo_root = Path(__file__).resolve().parent

    print("=" * 70)
    print("SWELL FEDERATED SETUP VALIDATION")
    print("=" * 70)

    checks = []

    # 1. Check repository structure
    print("\n[1/6] Checking repository structure...")
    scripts_ok, scripts_msgs = check_scripts_exist(repo_root)
    for msg in scripts_msgs:
        print(f"      {msg}")
    checks.append(scripts_ok)

    # 2. Check SWELL dataset
    print("\n[2/6] Checking SWELL dataset...")
    dataset_ok, dataset_msgs = check_swell_dataset()
    for msg in dataset_msgs:
        print(f"      {msg}")
    checks.append(dataset_ok)

    # 3. Check Python modules
    print("\n[3/6] Checking Python modules...")
    modules_ok, modules_msgs = check_python_modules()
    for msg in modules_msgs:
        print(f"      {msg}")
    checks.append(modules_ok)

    # 4. Check configuration
    print("\n[4/6] Checking configuration file...")
    config_path = repo_root / "configs/swell_federated_10runs.yaml"
    config_ok, config_msgs = check_config_file(config_path)
    for msg in config_msgs:
        print(f"      {msg}")
    checks.append(config_ok)

    # 5. Check MQTT broker
    print("\n[5/6] Checking MQTT broker...")
    mqtt_ok, mqtt_msg = check_mqtt_broker()
    print(f"      {mqtt_msg}")
    checks.append(mqtt_ok)

    # 6. Check Mosquitto availability
    print("\n[6/6] Checking Mosquitto availability...")
    try:
        result = subprocess.run(
            ["which", "mosquitto"], capture_output=True, timeout=5
        )
        if result.returncode == 0:
            print(f"      ✅ Mosquitto command available")
            checks.append(True)
        else:
            print(f"      ⚠️  Mosquitto command not in PATH (may still work if running)")
            checks.append(True)  # Not critical
    except Exception as e:
        print(f"      ⚠️  Could not check Mosquitto: {e}")
        checks.append(True)  # Not critical

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    critical_checks = checks[:5]  # First 5 are critical
    if all(critical_checks):
        print("✅ All critical checks passed!")
        print("\nYou can now run:")
        print(f"  python run_swell_10_executions.py --config configs/swell_federated_10runs.yaml")
        return 0
    else:
        print("❌ Some critical checks failed!")
        print("\nFix the issues above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
