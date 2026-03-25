#!/usr/bin/env python3
"""
Prepare SWELL federated partitions from a JSON/YAML config.

Usage:
  python scripts/prepare_swell_federated.py --config configs/swell_fed.example.json

Outputs a directory with per-node train/val/test npz files and a manifest.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure 'src' is on sys.path when running from repo root (no install needed)
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from flower_basic.datasets.swell_federated import plan_and_materialize_swell_federated


def ensure_swell_features() -> bool:
    """Check if SWELL feature CSV files exist, if not process RRI data."""
    data_dir = repo_root / "data" / "SWELL"
    required_files = [
        "A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv",
        "B - Facial expressions features (FaceReaderAllData_final (NaN is 999))-sheet_1.csv",
        "C - Body posture features (Kinect C (position - per minute))- Sheet_1.csv",
        "D - Physiology features (HR_HRV_SCL - final).csv",
    ]

    # Check if all files exist
    if all((data_dir / f).exists() for f in required_files):
        print("✅ SWELL feature files found")
        return True

    # Check if RRI data exists
    rri_dir = data_dir / "data" / "raw" / "rri"
    if rri_dir.exists() and any(rri_dir.glob("p*.txt")):
        print("🔄 SWELL feature files missing, processing RRI data...")
        try:
            import subprocess

            result = subprocess.run(
                [sys.executable, str(repo_root / "process_swell_rri.py")],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                print(result.stdout)
                return True
            else:
                print(f"⚠️  RRI processing failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"⚠️  Could not process RRI data: {e}")
            return False

    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SWELL federated datasets from config"
    )
    parser.add_argument(
        "--config", required=True, help="Path to JSON or YAML config file"
    )
    args = parser.parse_args()

    # Ensure feature files exist
    if not ensure_swell_features():
        print("❌ SWELL feature files not available")
        sys.exit(1)

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
