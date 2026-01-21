#!/usr/bin/env python3
"""Quick test of SWEET federated setup with 70/20/10 splits.

This script verifies that the SWEET federated infrastructure is working correctly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from flower_basic.datasets.sweet_federated import (
    FederatedConfigSWEET,
    plan_and_materialize_sweet_federated,
)
from flower_basic.datasets.sweet_samples import load_sweet_sample_dataset


def test_sweet_loading():
    """Test loading SWEET dataset with 70/20/10 split."""
    print("=" * 80)
    print("Testing SWEET Dataset Loading (70/20/10 split)")
    print("=" * 80)

    try:
        dataset = load_sweet_sample_dataset(
            data_dir="data/SWEET/sample_subjects",
            label_strategy="binary",
            elevated_threshold=2.0,
            train_fraction=0.7,
            val_fraction=0.2,
            random_state=42,
        )

        train_pct = len(dataset.train.y) / (
            len(dataset.train.y) + len(dataset.val.y) + len(dataset.test.y)
        )
        val_pct = len(dataset.val.y) / (
            len(dataset.train.y) + len(dataset.val.y) + len(dataset.test.y)
        )
        test_pct = len(dataset.test.y) / (
            len(dataset.train.y) + len(dataset.val.y) + len(dataset.test.y)
        )

        print(f"✓ Dataset loaded successfully")
        print(
            f"  Train: {len(dataset.train_subjects)} subjects, {len(dataset.train.y)} samples ({train_pct:.1%})"
        )
        print(
            f"  Val:   {len(dataset.val_subjects)} subjects, {len(dataset.val.y)} samples ({val_pct:.1%})"
        )
        print(
            f"  Test:  {len(dataset.test_subjects)} subjects, {len(dataset.test.y)} samples ({test_pct:.1%})"
        )
        print(f"  Features: {len(dataset.feature_names)}")
        print(f"  Label strategy: {dataset.label_strategy}")

        return True
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False


def test_federated_config():
    """Test federated configuration defaults."""
    print("\n" + "=" * 80)
    print("Testing SWEET Federated Configuration (70/20/10 defaults)")
    print("=" * 80)

    config = FederatedConfigSWEET()

    print(f"✓ Default configuration:")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Split train: {config.split_train} (70%)")
    print(f"  Split val: {config.split_val} (20%)")
    print(f"  Split test: {config.split_test} (10%)")
    print(f"  Scaler: {config.scaler}")
    print(f"  Split strategy: {config.split_strategy}")
    print(f"  Label strategy: {config.label_strategy}")

    total = config.split_train + config.split_val + config.split_test
    if abs(total - 1.0) < 1e-6:
        print(f"✓ Split percentages sum to 1.0")
        return True
    else:
        print(f"❌ Split percentages sum to {total}, expected 1.0")
        return False


def test_config_file():
    """Test example config file."""
    print("\n" + "=" * 80)
    print("Testing SWEET Example Config File")
    print("=" * 80)

    config_path = Path("configs/sweet_federated.example.yaml")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False

    try:
        import yaml

        config_data = yaml.safe_load(config_path.read_text())

        split_config = config_data.get("split", {})
        train = split_config.get("train", 0.0)
        val = split_config.get("val", 0.0)
        test = split_config.get("test", 0.0)

        print(f"✓ Config file loaded:")
        print(f"  Train: {train} (70%)")
        print(f"  Val: {val} (20%)")
        print(f"  Test: {test} (10%)")

        if abs(train - 0.7) < 1e-6 and abs(val - 0.2) < 1e-6 and abs(test - 0.1) < 1e-6:
            print(f"✓ Config file has correct 70/20/10 split")
            return True
        else:
            print(f"❌ Config file split does not match 70/20/10")
            return False

    except Exception as e:
        print(f"❌ Failed to load config file: {e}")
        return False


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "SWEET FEDERATED SYSTEM TEST" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    results = []

    # Test 1: Dataset loading
    results.append(("Dataset Loading", test_sweet_loading()))

    # Test 2: Federated config
    results.append(("Federated Config", test_federated_config()))

    # Test 3: Config file
    results.append(("Config File", test_config_file()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print(
            "\nThe SWEET federated system is correctly configured with 70/20/10 splits."
        )
        print("\nNext steps:")
        print("  1. Prepare baseline model:")
        print(
            "     python scripts/prepare_sweet_baseline.py --data-dir data/SWEET/selection1"
        )
        print("\n  2. Prepare federated splits:")
        print(
            "     python scripts/prepare_sweet_federated.py --config configs/sweet_federated.example.yaml"
        )
        print("\n  3. Run federated demo:")
        print(
            "     python scripts/run_sweet_federated_demo.py --config configs/sweet_federated.example.yaml"
        )
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease check the errors above and fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
