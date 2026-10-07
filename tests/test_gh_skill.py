"""The GitHub helper of the task workflow never prints a credential (QA1, #173).

It runs gh.py in a scratch git repository; no request reaches GitHub, because
the origin URL is refused before any credential is asked for.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

GH = Path(".claude/skills/tarea-github/scripts/gh.py").resolve()


def test_a_bad_origin_url_is_reported_without_its_credential(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    secret = "ghp_EXAMPLEtoken1234567890"
    url = f"https://someone:{secret}@github.com/owner/repo/"  # trailing slash: refused
    subprocess.run(
        ["git", "-C", str(tmp_path), "remote", "add", "origin", url], check=True
    )

    done = subprocess.run(
        [sys.executable, str(GH), "whoami"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert done.returncode != 0
    assert secret not in done.stdout + done.stderr
    assert "github.com/owner/repo" in done.stdout + done.stderr
