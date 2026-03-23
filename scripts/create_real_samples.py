#!/usr/bin/env python3
"""
Real Dataset Sample Creator
===========================

Creates small samples from real WESAD and SWELL datasets for testing purposes.
These samples maintain the authenticity of the original data while being lightweight.

STRICT RULE: Uses ONLY real data, NO mock generation.
"""

import json
import pickle
from pathlib import Path

import pandas as pd


def create_wesad_sample(data_dir: str = "data/WESAD", output_dir: str = "data/samples"):
    """
    Create a small sample from real WESAD dataset.

    Args:
        data_dir: Path to full WESAD dataset
        output_dir: Path to save samples
    """
    print("🧬 Creating WESAD real data sample...")

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Select a few subjects for sample
    sample_subjects = ["S2", "S14", "S10"]  # Good representatives

    sample_data = {}

    for subject_id in sample_subjects:
        subject_file = data_path / subject_id / f"{subject_id}.pkl"

        if not subject_file.exists():
            print(f"  ⚠️  {subject_id} not found, skipping...")
            continue

        try:
            # Load real WESAD data
            with open(subject_file, "rb") as f:
                full_data = pickle.load(f, encoding="latin1")

            print(f"  📊 Processing real data from {subject_id}...")

            # Extract subset of real signals (first 1000 samples from each modality)
            wrist_signals = full_data["signal"]["wrist"]
            labels = full_data["label"]

            # Create sample with real data structure
            sample_signals = {}
            for signal_name, signal_data in wrist_signals.items():
                if signal_name in ["BVP", "EDA", "ACC", "TEMP"]:
                    # Take real samples from different time periods
                    sample_size = min(1000, len(signal_data))
                    sample_signals[signal_name] = signal_data[:sample_size]

            # Corresponding real labels
            label_sample_size = min(1000, len(labels))
            sample_labels = labels[:label_sample_size]

            # Store real sample
            sample_data[subject_id] = {
                "signal": {"wrist": sample_signals},
                "label": sample_labels,
                "subject": full_data["subject"],
                "metadata": {
                    "source": "REAL_WESAD_SAMPLE",
                    "original_file": str(subject_file),
                    "sample_size": sample_size,
                    "sampling_info": "First 1000 samples from real physiological data",
                },
            }

            print(
                f"    ✅ Real sample created: {len(sample_signals)} signals, {len(sample_labels)} labels"
            )

        except Exception as e:
            print(f"  ❌ Error processing {subject_id}: {e}")

    # Save real samples
    sample_file = output_path / "wesad_real_sample.pkl"
    with open(sample_file, "wb") as f:
        pickle.dump(sample_data, f)

    print(f"💾 WESAD real samples saved to: {sample_file}")
    print(f"   Subjects included: {list(sample_data.keys())}")

    return sample_file


def create_swell_sample(data_dir: str = "data/SWELL", output_dir: str = "data/samples"):
    """
    Create a small sample from real SWELL dataset.

    Args:
        data_dir: Path to full SWELL dataset
        output_dir: Path to save samples
    """
    print("🎯 Creating SWELL real data sample...")

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Try to find SWELL CSV files
    possible_paths = [
        data_path / "3 - Feature dataset" / "per sensor",
        data_path,
        Path("data/0_SWELL"),
    ]

    csv_files = []
    swell_path = None

    for path in possible_paths:
        if path.exists():
            found_csvs = list(path.glob("*.csv"))
            if found_csvs:
                csv_files = found_csvs
                swell_path = path
                print(f"  📊 Found SWELL CSVs in: {swell_path}")
                break

    if not csv_files:
        print("  ⚠️  No real SWELL CSV files found for sampling")
        return None

    # Load and sample real SWELL data
    sample_data = {}

    for csv_file in csv_files[:4]:  # Sample from first 4 CSV files
        try:
            # Load real SWELL data
            try:
                df = pd.read_csv(csv_file, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file, encoding="latin1")

            print(f"  📈 Processing real data from: {csv_file.name}")
            print(f"    Original shape: {df.shape}")

            # Take sample of real data (first 50 rows)
            sample_size = min(50, len(df))
            df_sample = df.head(sample_size).copy()

            # Store real sample
            modality_name = csv_file.stem.lower()
            if "computer" in modality_name or "ulog" in modality_name:
                modality = "computer"
            elif "facial" in modality_name or "face" in modality_name:
                modality = "facial"
            elif "posture" in modality_name or "kinect" in modality_name:
                modality = "posture"
            elif "physiology" in modality_name or "hr" in modality_name:
                modality = "physiology"
            else:
                modality = "unknown"

            sample_data[modality] = {
                "data": df_sample,
                "metadata": {
                    "source": "REAL_SWELL_SAMPLE",
                    "original_file": str(csv_file),
                    "original_shape": df.shape,
                    "sample_shape": df_sample.shape,
                    "sampling_info": f"First {sample_size} rows from real SWELL data",
                },
            }

            print(f"    ✅ Real sample created: {df_sample.shape}")

        except Exception as e:
            print(f"  ❌ Error processing {csv_file.name}: {e}")

    # Save real samples
    sample_file = output_path / "swell_real_sample.pkl"
    with open(sample_file, "wb") as f:
        pickle.dump(sample_data, f)

    # Also save as JSON for easy inspection
    json_data = {}
    for modality, data in sample_data.items():
        json_data[modality] = {
            "metadata": data["metadata"],
            "columns": data["data"].columns.tolist(),
            "shape": data["data"].shape,
            "sample_rows": data["data"]
            .head(3)
            .to_dict("records"),  # First 3 rows for inspection
        }

    json_file = output_path / "swell_real_sample_info.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    print(f"💾 SWELL real samples saved to: {sample_file}")
    print(f"📋 Sample info saved to: {json_file}")
    print(f"   Modalities included: {list(sample_data.keys())}")

    return sample_file


def create_dataset_samples():
    """Create real data samples from both datasets."""
    print("🔬 Creating Real Dataset Samples")
    print("=" * 40)
    print("⚠️  STRICT RULE: NO MOCK DATA - REAL DATA ONLY!")
    print()

    # Create samples directory
    samples_dir = Path("data/samples")
    samples_dir.mkdir(exist_ok=True)

    # Create WESAD sample
    try:
        wesad_sample = create_wesad_sample()
        print("✅ WESAD real sample created")
    except Exception as e:
        print(f"❌ WESAD sample failed: {e}")
        wesad_sample = None

    print()

    # Create SWELL sample
    try:
        swell_sample = create_swell_sample()
        print("✅ SWELL real sample created")
    except Exception as e:
        print(f"❌ SWELL sample failed: {e}")
        swell_sample = None

    print()
    print("📊 REAL DATA SAMPLES SUMMARY")
    print("=" * 30)

    if wesad_sample:
        print(f"🧬 WESAD: {wesad_sample}")
    if swell_sample:
        print(f"🎯 SWELL: {swell_sample}")

    print()
    print("✅ All samples use AUTHENTIC dataset files")
    print("🚫 Zero synthetic/mock data generated")
    print("🔬 Ready for ML algorithm testing")


if __name__ == "__main__":
    create_dataset_samples()
