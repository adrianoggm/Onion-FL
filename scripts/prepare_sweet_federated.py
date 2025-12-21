#!/usr/bin/env python3
"""Prepare SWEET federated splits from configuration file.

Similar to prepare_swell_federated.py but for SWEET dataset.

Usage:
    python scripts/prepare_sweet_federated.py \\
        --config configs/sweet_federated.example.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from flower_basic.datasets.sweet_federated import plan_and_materialize_sweet_federated


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SWEET federated splits from config file"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML/JSON config file",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SWEET Federated Data Preparation")
    print("=" * 80)
    
    try:
        manifest = plan_and_materialize_sweet_federated(args.config)
        
        print("\n" + "=" * 80)
        print("✓ SUCCESS")
        print("=" * 80)
        print(f"Output directory: {manifest['output_dir']}")
        print(f"Number of nodes: {len(manifest['nodes'])}")
        print(f"Number of features: {manifest['meta']['n_features']}")
        print(f"Total subjects: {manifest['meta']['n_subjects']}")
        print(f"Total clients: {sum(len(clients) for clients in manifest['clients'].values())}")
        
        print("\nNode Details:")
        for node_id, subjects_list in manifest['nodes'].items():
            num_clients = len(manifest['clients'][node_id])
            print(f"  {node_id}:")
            print(f"    Subjects: {len(subjects_list)}")
            print(f"    Clients: {num_clients}")
            print(f"    Sample subjects: {', '.join(subjects_list[:3])}...")
        
        print("\nNext steps:")
        print("  1. Start MQTT broker (if not running)")
        print("  2. Run architecture script:")
        print(f"     python scripts/run_sweet_architecture.py --config {args.config} --launch")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
