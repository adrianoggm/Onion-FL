#!/usr/bin/env python3
"""
Prepare SWELL federated partitions from a JSON/YAML config.

Usage:
  python scripts/prepare_swell_federated.py --config configs/swell_fed.example.json

Outputs a directory with per-node train/val/test npz files and a manifest.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure 'src' is on sys.path when running from repo root (no install needed)
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from flower_basic.datasets.swell_federated import plan_and_materialize_swell_federated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SWELL federated datasets from config"
    )
    parser.add_argument(
        "--config", required=True, help="Path to JSON or YAML config file"
    )
    args = parser.parse_args()

    result = plan_and_materialize_swell_federated(args.config)

    out = result["output_dir"]
    print("\n=== SWELL Federated Partitions Prepared ===")
    print(f"Output directory: {out}")
    print(f"Manifest: {Path(out) / 'manifest.json'}")
    nodes = result["manifest"]["nodes"]
    print("Nodes:")
    for n, subs in nodes.items():
        print(f"  - {n}: {len(subs)} subjects")


if __name__ == "__main__":
    main()
