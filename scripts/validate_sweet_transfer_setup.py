#!/usr/bin/env python3
"""
Quick validation script to check SWEET transfer learning setup.

Verifies:
- Data directories exist
- Configuration files are valid
- Python dependencies are available
- Module imports work correctly

Usage:
    python scripts/validate_sweet_transfer_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def check_mark(passed: bool) -> str:
    return "✅" if passed else "❌"


def main():
    print("=" * 80)
    print("SWEET Transfer Learning Setup Validation")
    print("=" * 80)
    
    all_checks_passed = True
    
    # 1. Check data directories
    print("\n[1/6] Checking data directories...")
    
    selection1_dir = Path("data/SWEET/selection1/users")
    selection1_exists = selection1_dir.exists()
    print(f"  {check_mark(selection1_exists)} Selection1 data: {selection1_dir}")
    if selection1_exists:
        selection1_users = list(selection1_dir.glob("user*"))
        print(f"      Found {len(selection1_users)} users")
    all_checks_passed &= selection1_exists
    
    selection2_dir = Path("data/SWEET/selection2/users")
    selection2_exists = selection2_dir.exists()
    print(f"  {check_mark(selection2_exists)} Selection2 data: {selection2_dir}")
    if selection2_exists:
        selection2_users = list(selection2_dir.glob("user*"))
        print(f"      Found {len(selection2_users)} users")
    all_checks_passed &= selection2_exists
    
    # 2. Check configuration files
    print("\n[2/6] Checking configuration files...")
    
    config_file = Path("configs/sweet_federated_transfer.yaml")
    config_exists = config_file.exists()
    print(f"  {check_mark(config_exists)} Config file: {config_file}")
    all_checks_passed &= config_exists
    
    # 3. Check Python dependencies
    print("\n[3/6] Checking Python dependencies...")
    
    required_packages = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("sklearn", "sklearn"),
        ("xgboost", "xgb"),
        ("torch", "torch"),
        ("yaml", "yaml"),
    ]
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} (not installed)")
            all_checks_passed = False
    
    # 4. Check module imports
    print("\n[4/6] Checking module imports...")
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    
    modules_to_check = [
        "flower_basic.datasets.sweet_samples",
        "flower_basic.datasets.sweet_federated",
        "flower_basic.clients.sweet",
        "flower_basic.servers.sweet",
    ]
    
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except Exception as e:
            print(f"  ❌ {module}: {e}")
            all_checks_passed = False
    
    # 5. Check scripts
    print("\n[5/6] Checking scripts...")
    
    scripts_to_check = [
        "scripts/extract_sweet_selection2.py",
        "scripts/train_sweet_baseline_selection1.py",
        "scripts/prepare_sweet_federated_transfer.py",
        "scripts/run_sweet_federated_demo.py",
        "scripts/run_sweet_transfer_learning.py",
    ]
    
    for script_path in scripts_to_check:
        script = Path(script_path)
        script_exists = script.exists()
        print(f"  {check_mark(script_exists)} {script_path}")
        all_checks_passed &= script_exists
    
    # 6. Check baseline model (optional)
    print("\n[6/6] Checking baseline model (optional)...")
    
    baseline_model = Path("baseline_models/sweet/xgboost_tuned_model.json")
    baseline_scaler = Path("baseline_models/sweet/scaler.json")
    
    baseline_exists = baseline_model.exists()
    scaler_exists = baseline_scaler.exists()
    
    print(f"  {check_mark(baseline_exists)} Baseline model: {baseline_model}")
    print(f"  {check_mark(scaler_exists)} Baseline scaler: {baseline_scaler}")
    
    if not baseline_exists or not scaler_exists:
        print("      Note: Run 'python scripts/train_sweet_baseline_selection1.py' to create")
    
    # Summary
    print("\n" + "=" * 80)
    if all_checks_passed:
        print("✅ All critical checks passed!")
        print("\nNext steps:")
        if not baseline_exists:
            print("  1. Train baseline model: python scripts/train_sweet_baseline_selection1.py")
            print("  2. Run transfer learning: python scripts/run_sweet_transfer_learning.py")
        else:
            print("  1. Run transfer learning: python scripts/run_sweet_transfer_learning.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        if not selection2_exists:
            print("  - Extract selection2: python scripts/extract_sweet_selection2.py")
        print("  - Install dependencies: pip install -r requirements.txt")
    
    print("=" * 80)
    
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
