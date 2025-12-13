#!/usr/bin/env python3
"""Extract SWEET selection1 user ZIP files.

This script extracts individual user ZIP files from selection1_zip folder
to make them compatible with the SWEET dataset loader.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def extract_user_zips(source_dir: Path, target_dir: Path, force: bool = False):
    """Extract all user ZIP files from source to target directory.
    
    Args:
        source_dir: Directory containing user*.zip files
        target_dir: Directory where to extract (will create user folders)
        force: If True, overwrite existing extracted folders
    """
    
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    # Find all user ZIP files
    zip_files = sorted(source_dir.glob("user*.zip"))
    
    if not zip_files:
        print(f"❌ No user*.zip files found in {source_dir}")
        return False
    
    print(f"Found {len(zip_files)} user ZIP files")
    print(f"Extracting to: {target_dir}")
    print("=" * 60)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    skipped_count = 0
    
    for zip_path in zip_files:
        user_name = zip_path.stem  # e.g., "user0091"
        user_dir = target_dir / user_name
        
        if user_dir.exists() and not force:
            print(f"⊙ Skipping {user_name} (already exists)")
            skipped_count += 1
            continue
        
        try:
            print(f"→ Extracting {user_name}...", end=" ")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(user_dir)
            
            print("✓")
            extracted_count += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("=" * 60)
    print(f"✓ Extraction complete")
    print(f"  Extracted: {extracted_count} users")
    print(f"  Skipped: {skipped_count} users (already existed)")
    print(f"  Target: {target_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract SWEET selection1 user ZIP files"
    )
    parser.add_argument(
        "--source",
        default="data/SWEET/selection1/selection1_zip",
        help="Directory containing user*.zip files",
    )
    parser.add_argument(
        "--target",
        default="data/SWEET/selection1/users",
        help="Directory to extract user folders to",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if folders exist",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SWEET Selection1 User ZIP Extraction")
    print("=" * 80)
    
    source = Path(args.source)
    target = Path(args.target)
    
    success = extract_user_zips(source, target, args.force)
    
    if success:
        print("\n✓ Ready to use with prepare_sweet_baseline.py:")
        print(f"  python scripts/prepare_sweet_baseline.py --data-dir {target}")
    else:
        print("\n❌ Extraction failed")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
