#!/usr/bin/env python3
"""Simple ECG5000 deprecation checker for CI/CD integration.

This is a simplified version that works reliably in Windows environments
and CI/CD pipelines without Unicode encoding issues.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def check_ecg5000_usage(project_root: Path) -> tuple[int, list[str]]:
    """Check for ECG5000 usage in the project."""
    violations = []
    count = 0

    # Patterns to detect
    patterns = [
        r"ecg5000",
        r"ECG5000",
        r"load_ecg5000",
    ]

    # Files to check
    src_dirs = ["src", "tests"]

    for src_dir in src_dirs:
        src_path = project_root / src_dir
        if not src_path.exists():
            continue

        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    line_clean = line.strip()

                    # Skip comments and certain files
                    if line_clean.startswith("#"):
                        continue
                    if "deprecated" in str(py_file).lower():
                        continue

                    for pattern in patterns:
                        if re.search(pattern, line_clean, re.IGNORECASE):
                            relative_path = py_file.relative_to(project_root)
                            violations.append(
                                f"{relative_path}:{line_num}: {line_clean[:80]}"
                            )
                            count += 1
                            break

            except Exception:
                continue

    return count, violations


def main() -> None:
    """Main function."""
    project_root = Path.cwd()

    count, violations = check_ecg5000_usage(project_root)

    if count == 0:
        print("SUCCESS: No deprecated ECG5000 usage detected")
        sys.exit(0)
    else:
        print(f"ERROR: Found {count} deprecated ECG5000 usages")
        print("\nViolations found:")
        for violation in violations[:20]:  # Show first 20
            print(f"  {violation}")

        if len(violations) > 20:
            print(f"  ... and {len(violations) - 20} more")

        print("\nMigration required: Replace ECG5000 with WESAD/SWELL datasets")
        print("See docs/RULES.md for complete migration guide")
        sys.exit(1)


if __name__ == "__main__":
    main()
