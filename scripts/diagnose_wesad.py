#!/usr/bin/env python3
"""
WESAD Dataset Diagnostic Tool
Analyzes WESAD data structure to understand why few windows are extracted
"""

import pickle
from pathlib import Path

import numpy as np


def diagnose_wesad_subject(subject_path):
    """Diagnose a single WESAD subject"""
    try:
        with open(subject_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        print(f"\n{'='*60}")
        print(f"Subject: {subject_path.stem}")
        print(f"{'='*60}")

        # Check data structure
        print("Data keys:", list(data.keys()))

        if "signal" in data:
            signals = data["signal"]
            print("Signal modalities:", list(signals.keys()))

            # Analyze each signal
            for modality, signal_data in signals.items():
                if isinstance(signal_data, np.ndarray):
                    print(
                        f"  {modality}: shape {signal_data.shape}, dtype {signal_data.dtype}"
                    )
                else:
                    print(f"  {modality}: {type(signal_data)}")

        if "label" in data:
            labels = data["label"]
            print(f"Labels: shape {labels.shape}, dtype {labels.dtype}")

            # Analyze label distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            print("Label distribution:")
            for label, count in zip(unique_labels, counts):
                percentage = count / len(labels) * 100
                print(f"  Label {label}: {count:,} samples ({percentage:.1f}%)")

                if label != 0:  # Non-transient labels
                    # Find longest contiguous regions
                    label_indices = np.where(labels == label)[0]
                    if len(label_indices) > 0:
                        diffs = np.diff(label_indices)
                        splits = np.where(diffs > 1)[0] + 1
                        regions = np.split(label_indices, splits)
                        region_lengths = [len(region) for region in regions]
                        max_region = max(region_lengths) if region_lengths else 0
                        print(f"    Longest contiguous region: {max_region:,} samples")
                        print(f"    Number of regions: {len(regions)}")

                        # Check if long enough for windows
                        window_size = 1920  # 30 seconds at 64Hz
                        usable_regions = [r for r in regions if len(r) >= window_size]
                        print(
                            f"    Regions >= {window_size} samples: {len(usable_regions)}"
                        )

        # Check if subject has usable data
        if "signal" in data and "label" in data:
            signals = data["signal"]
            labels = data["label"]

            # Check length alignment
            if "BVP" in signals:
                bvp_len = len(signals["BVP"])
                label_len = len(labels)
                print("\nSignal-Label alignment:")
                print(f"  BVP length: {bvp_len:,}")
                print(f"  Label length: {label_len:,}")
                print(f"  Length match: {bvp_len == label_len}")

                # Calculate potential windows
                window_size = 1920
                step_size = 960
                min_length = min(bvp_len, label_len)

                # Count windows with non-transient labels
                valid_windows = 0
                for start in range(0, min_length - window_size, step_size):
                    end = start + window_size
                    label_window = labels[start:end]
                    unique_labels, counts = np.unique(label_window, return_counts=True)
                    dominant_label = unique_labels[np.argmax(counts)]

                    # Check consistency and non-transient
                    if (
                        np.max(counts) / len(label_window) >= 0.5
                        and dominant_label != 0
                    ):
                        valid_windows += 1

                print(f"  Potential valid windows: {valid_windows}")

    except Exception as e:
        print(f"Error processing {subject_path}: {e}")


def main():
    """Main diagnostic function"""
    print("🔍 WESAD Dataset Diagnostic")
    print("=" * 60)

    # Find WESAD data directory
    data_dir = Path("data/WESAD")
    if not data_dir.exists():
        print(f"WESAD directory not found: {data_dir}")
        return

    # Process each subject
    subject_files = sorted(data_dir.glob("S*/S*.pkl"))
    print(f"Found {len(subject_files)} subject files")

    for subject_file in subject_files:
        diagnose_wesad_subject(subject_file)

    print(f"\n{'='*60}")
    print("Diagnostic complete!")


if __name__ == "__main__":
    main()
