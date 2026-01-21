#!/usr/bin/env python3
"""
Prepare SWEET selection2 federated splits for transfer learning.

This script materializes federated data splits from selection2 (140 subjects)
to be used for fine-tuning the pre-trained model from selection1.

Usage:
    python scripts/prepare_sweet_federated_transfer.py

Prerequisites:
    1. Extract selection2: python scripts/extract_sweet_selection2.py
    2. Train baseline model: python scripts/train_sweet_baseline_selection1.py

Output:
    federated_runs/sweet/transfer_selection2/
      ├── manifest.json
      ├── scaler_global.json
      ├── pretrained_model.json (copied from baseline)
      ├── pretrained_scaler.json (copied from baseline)
      ├── fog_0/
      │   ├── train.npz, val.npz, test.npz
      │   └── subject_*/train.npz, val.npz, test.npz
      ├── fog_1/...
      └── fog_2/...
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flower_basic.datasets.sweet_federated import plan_and_materialize_sweet_federated


def main():
    print("=" * 80)
    print("SWEET Selection2 Federated Splits Preparation (Transfer Learning)")
    print("=" * 80)

    config_path = "configs/sweet_federated_transfer.yaml"

    print(f"\n[INFO] Reading configuration from: {config_path}")
    print("[INFO] This will:")
    print("  1. Load SWEET selection2 (140 subjects)")
    print("  2. Create subject-level train/val/test splits (60/20/20)")
    print("  3. Distribute subjects across 3 fog nodes")
    print("  4. Materialize NPZ files for federated training")
    print("  5. Copy pre-trained model and scaler from baseline")

    try:
        manifest = plan_and_materialize_sweet_federated(config_path)

        print("\n" + "=" * 80)
        print("✅ SWEET Selection2 Federated Splits Ready!")
        print(f"   Output: {manifest['output_dir']}")
        print(f"   Nodes: {len(manifest['nodes'])}")
        print(f"   Ready for federated fine-tuning")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print(
            "  1. selection2 data extracted: python scripts/extract_sweet_selection2.py"
        )
        print(
            "  2. Baseline model trained: python scripts/train_sweet_baseline_selection1.py"
        )
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
