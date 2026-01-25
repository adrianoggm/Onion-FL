#!/usr/bin/env python3
"""Script to validate GitHub Actions workflows."""

import sys
from pathlib import Path
from typing import Dict, List

import yaml


def validate_workflow_file(workflow_path: Path) -> Dict[str, List[str]]:
    """Validate a single workflow file."""
    errors = []
    warnings = []

    try:
        with open(workflow_path, encoding="utf-8") as f:
            content = f.read()
        workflow = yaml.safe_load(content)
    except UnicodeDecodeError:
        try:
            with open(workflow_path, encoding="latin-1") as f:
                content = f.read()
            workflow = yaml.safe_load(content)
        except Exception as e:
            errors.append(f"File encoding error: {e}")
            return {"errors": errors, "warnings": warnings}
    except yaml.YAMLError as e:
        errors.append(f"YAML parsing error: {e}")
        return {"errors": errors, "warnings": warnings}
    except Exception as e:
        errors.append(f"File reading error: {e}")
        return {"errors": errors, "warnings": warnings}

    # Check required fields
    if "name" not in workflow:
        errors.append("Missing 'name' field")

    # Check for trigger field (can be 'on' or True in YAML)
    if "on" not in workflow and True not in workflow:
        errors.append("Missing trigger field")

    if "jobs" not in workflow:
        errors.append("Missing 'jobs' field")

    # Check jobs structure
    if "jobs" in workflow:
        jobs = workflow["jobs"]
        if not isinstance(jobs, dict):
            errors.append("'jobs' should be a dictionary")
        else:
            for job_name, job_config in jobs.items():
                if "runs-on" not in job_config:
                    errors.append(f"Job '{job_name}' missing 'runs-on' field")

                if "steps" not in job_config:
                    warnings.append(f"Job '{job_name}' has no steps")

    # Check for Python setup in CI workflows
    workflow_name = workflow.get("name", "").lower()
    if any(keyword in workflow_name for keyword in ["ci", "test", "pr"]):
        has_python_setup = False
        if "jobs" in workflow:
            for job_config in workflow["jobs"].values():
                if "steps" in job_config:
                    for step in job_config["steps"]:
                        if isinstance(step, dict) and "uses" in step:
                            if "setup-python" in step["uses"]:
                                has_python_setup = True
                                break

        if not has_python_setup:
            warnings.append("CI workflow should include Python setup")

    return {"errors": errors, "warnings": warnings}


def validate_all_workflows() -> bool:
    """Validate all workflow files."""
    workflows_dir = Path(".github/workflows")

    if not workflows_dir.exists():
        print("❌ No .github/workflows directory found")
        return False

    workflow_files = list(workflows_dir.glob("*.yml")) + list(
        workflows_dir.glob("*.yaml")
    )

    if not workflow_files:
        print("❌ No workflow files found")
        return False

    all_valid = True
    total_errors = 0
    total_warnings = 0

    print("🔍 Validating GitHub Actions workflows")
    print("=" * 50)

    for workflow_file in sorted(workflow_files):
        print(f"\n📄 Validating {workflow_file.name}...")

        result = validate_workflow_file(workflow_file)
        errors = result["errors"]
        warnings = result["warnings"]

        if errors:
            print(f"❌ Errors in {workflow_file.name}:")
            for error in errors:
                print(f"   • {error}")
            all_valid = False
            total_errors += len(errors)

        if warnings:
            print(f"⚠️ Warnings in {workflow_file.name}:")
            for warning in warnings:
                print(f"   • {warning}")
            total_warnings += len(warnings)

        if not errors and not warnings:
            print(f"✅ {workflow_file.name} is valid")

    print("\n" + "=" * 50)
    print("📊 Validation Summary:")
    print(f"   • Files checked: {len(workflow_files)}")
    print(f"   • Total errors: {total_errors}")
    print(f"   • Total warnings: {total_warnings}")

    if all_valid:
        print("\n🎉 All workflows are valid!")
    else:
        print("\n❌ Some workflows have errors and need fixing")

    return all_valid


def check_required_dependencies():
    """Check if all required dependencies are in requirements.txt."""
    requirements_file = Path("requirements.txt")

    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False

    with open(requirements_file) as f:
        requirements = f.read().lower()

    required_packages = ["torch", "flwr", "paho-mqtt", "numpy", "scikit-learn"]

    missing_packages = []
    for package in required_packages:
        if package not in requirements:
            missing_packages.append(package)

    if missing_packages:
        print(f"⚠️ Missing packages in requirements.txt: {missing_packages}")
        return False

    print("✅ All required packages found in requirements.txt")
    return True


def check_project_structure():
    """Check if all required files exist."""
    required_files = [
        "model.py",
        "utils.py",
        "broker_fog.py",
        "server.py",
        "client.py",
        "requirements.txt",
        "pytest.ini",
        "tests/",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False

    print("✅ All required project files exist")
    return True


def main():
    """Main validation function."""
    print("🚀 GitHub Actions Workflow Validation")
    print("=" * 50)

    all_checks_passed = True

    # Check project structure
    if not check_project_structure():
        all_checks_passed = False

    # Check dependencies
    if not check_required_dependencies():
        all_checks_passed = False

    # Validate workflows
    if not validate_all_workflows():
        all_checks_passed = False

    # Final summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 All validations passed! Workflows are ready for GitHub.")
        print("\n🚀 Next steps:")
        print("   1. Commit and push workflows to GitHub")
        print("   2. Configure branch protection rules")
        print("   3. Set up required status checks")
        print("   4. Configure Codecov token (optional)")
        return 0
    else:
        print("❌ Some validations failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
