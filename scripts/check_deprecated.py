#!/usr/bin/env python3
"""Check for deprecated ECG5000 dataset usage in codebase.

This script automatically detects usage of the deprecated ECG5000 dataset
and provides migration suggestions. It's integrated into the CI/CD pipeline
to prevent new code from using deprecated functionality.

Usage:
    python scripts/check_deprecated.py
    python scripts/check_deprecated.py --fix  # Auto-suggest fixes
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class DeprecationViolation(NamedTuple):
    """Represents a deprecated code usage violation."""

    file_path: Path
    line_number: int
    line_content: str
    violation_type: str
    suggestion: str


class ECG5000DeprecationChecker:
    """Checker for deprecated ECG5000 dataset usage."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path.cwd()
        self.violations: list[DeprecationViolation] = []

        # Patterns to detect ECG5000 usage
        self.deprecated_patterns = {
            "import_ecg5000": {
                "pattern": r"from.*ecg5000|import.*ecg5000",
                "suggestion": "Replace with WESAD dataset: from flower_basic.datasets import load_wesad_dataset",
            },
            "load_ecg5000_openml": {
                "pattern": r"load_ecg5000_openml",
                "suggestion": "Replace with load_wesad_dataset() for proper subject-based federated learning",
            },
            "ecg5000_dataset": {
                "pattern": r"ecg5000|ECG5000",
                "suggestion": "Migrate to WESAD dataset for realistic federated learning scenarios",
            },
            "ecg_model_140": {
                "pattern": r"140.*features?|140.*time.*points?",
                "suggestion": "Update model architecture for flexible input dimensions (WESAD/SWELL)",
            },
        }

        # Approved replacements
        self.approved_patterns = [
            r"load_wesad_dataset",
            r"load_swell_dataset",
            r"WESAD",
            r"SWELL",
        ]

    def check_file(self, file_path: Path) -> list[DeprecationViolation]:
        """Check a single file for deprecated ECG5000 usage."""
        violations = []

        if not file_path.exists() or file_path.suffix != ".py":
            return violations

        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line_clean = line.strip()

                # Skip comments and docstrings (basic check)
                if line_clean.startswith("#") or '"""' in line_clean:
                    continue

                # Check each deprecation pattern
                for violation_type, pattern_info in self.deprecated_patterns.items():
                    pattern = pattern_info["pattern"]
                    suggestion = pattern_info["suggestion"]

                    if re.search(pattern, line_clean, re.IGNORECASE):
                        # Check if it's a legitimate usage (e.g., in migration docs)
                        if self._is_legitimate_usage(line_clean, file_path):
                            continue

                        violations.append(
                            DeprecationViolation(
                                file_path=file_path,
                                line_number=line_num,
                                line_content=line_clean,
                                violation_type=violation_type,
                                suggestion=suggestion,
                            )
                        )

        except Exception as e:
            logger.warning(f"Could not check {file_path}: {e}")

        return violations

    def _is_legitimate_usage(self, line: str, file_path: Path) -> bool:
        """Check if ECG5000 usage is legitimate (e.g., in tests, docs, migration code)."""
        # Allow in test files that explicitly test deprecated functionality
        if "test_" in file_path.name and (
            "deprecated" in line.lower() or "legacy" in line.lower()
        ):
            return True

        # Allow in migration scripts or documentation
        if any(
            term in str(file_path).lower()
            for term in ["migration", "deprecated", "legacy", "docs"]
        ):
            return True

        # Allow if line contains migration context
        if any(
            term in line.lower()
            for term in ["deprecated", "migrate", "old", "legacy", "replace"]
        ):
            return True

        return False

    def check_project(self) -> list[DeprecationViolation]:
        """Check entire project for deprecated ECG5000 usage."""
        all_violations = []

        # Check source code
        src_paths = [
            self.project_root / "src",
            self.project_root / "tests",
            self.project_root / "examples",  # If exists
        ]

        for src_path in src_paths:
            if not src_path.exists():
                continue

            # Find all Python files
            python_files = list(src_path.rglob("*.py"))

            for py_file in python_files:
                violations = self.check_file(py_file)
                all_violations.extend(violations)

        self.violations = all_violations
        return all_violations

    def generate_report(self) -> str:
        """Generate a comprehensive deprecation report."""
        if not self.violations:
            return "SUCCESS: No deprecated ECG5000 usage detected. Good job!"

        report = ["ERROR: DEPRECATED ECG5000 USAGE DETECTED", "=" * 50, ""]

        # Group violations by type
        violations_by_type = {}
        for violation in self.violations:
            vtype = violation.violation_type
            if vtype not in violations_by_type:
                violations_by_type[vtype] = []
            violations_by_type[vtype].append(violation)

        # Generate report for each type
        for vtype, type_violations in violations_by_type.items():
            report.append(f"[{vtype.upper()}] {len(type_violations)} violations")
            report.append("-" * 40)

            for violation in type_violations:
                relative_path = violation.file_path.relative_to(self.project_root)
                report.extend(
                    [
                        f"  File: {relative_path}",
                        f"  Line {violation.line_number}: {violation.line_content}",
                        f"  Suggestion: {violation.suggestion}",
                        "",
                    ]
                )

        # Add summary and migration guide
        report.extend(
            [
                "=" * 50,
                f"SUMMARY: {len(self.violations)} total violations detected",
                "",
                "MIGRATION GUIDE:",
                "1. Replace ECG5000 dataset with WESAD dataset",
                "2. Update model architecture for flexible input dimensions",
                "3. Use subject-based data partitioning",
                "4. Implement proper cross-validation",
                "",
                "Documentation: See docs/RULES.md for complete migration guide",
                "",
            ]
        )

        return "\n".join(report)

    def suggest_fixes(self) -> dict[Path, list[str]]:
        """Generate automatic fix suggestions for each file."""
        fixes_by_file = {}

        for violation in self.violations:
            file_path = violation.file_path
            if file_path not in fixes_by_file:
                fixes_by_file[file_path] = []

            fix_suggestion = self._generate_specific_fix(violation)
            fixes_by_file[file_path].append(fix_suggestion)

        return fixes_by_file

    def _generate_specific_fix(self, violation: DeprecationViolation) -> str:
        """Generate a specific fix suggestion for a violation."""
        line = violation.line_content
        vtype = violation.violation_type

        if vtype == "import_ecg5000":
            return f"Replace: {line}\nWith: from flower_basic.datasets import load_wesad_dataset"

        elif vtype == "load_ecg5000_openml":
            return f"Replace: {line}\nWith: X_train, X_test, y_train, y_test = load_wesad_dataset()"

        elif vtype == "ecg5000_dataset":
            return f"Update: {line}\nTo use WESAD dataset with proper subject-based splitting"

        elif vtype == "ecg_model_140":
            return f"Update model architecture: {line}\nMake input dimension flexible for multi-modal data"

        return violation.suggestion


def main() -> None:
    """Main entry point for deprecation checker."""
    # Set UTF-8 encoding for Windows compatibility
    import sys

    if sys.platform == "win32":
        import os

        os.environ["PYTHONIOENCODING"] = "utf-8"

    parser = argparse.ArgumentParser(
        description="Check for deprecated ECG5000 dataset usage"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Generate automatic fix suggestions"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Run checker
    checker = ECG5000DeprecationChecker(args.project_root)
    violations = checker.check_project()

    # Generate and print report
    report = checker.generate_report()
    print(report)

    # Generate fixes if requested
    if args.fix and violations:
        print("\nAUTOMATIC FIX SUGGESTIONS:")
        print("=" * 50)

        fixes = checker.suggest_fixes()
        for file_path, file_fixes in fixes.items():
            relative_path = file_path.relative_to(args.project_root)
            print(f"\n{relative_path}:")
            for fix in file_fixes:
                print(f"  {fix}")

    # Exit with error code if violations found (for CI/CD)
    exit_code = len(violations)
    if exit_code > 0:
        print(
            f"\nFound {exit_code} violations. Please migrate to WESAD/SWELL datasets."
        )
    else:
        print("\nAll good! No deprecated ECG5000 usage found.")

    exit(min(exit_code, 1))  # Cap exit code at 1 for shell compatibility


if __name__ == "__main__":
    main()
