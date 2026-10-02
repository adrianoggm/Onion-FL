from __future__ import annotations

"""Identifiers that sign a result (spec §10.1).

``topology_id`` lives on ``Topology``. The rest are here: what was asked for
(``config_id``), with which data (``data_id``) and code (``code_version``),
which execution (``run_id``) and what it produced (``run_hash``, in
``observability.run``).
"""

import hashlib
import json
import os
import socket
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def config_id(config: Mapping[str, Any]) -> str:
    """SHA-256 of the resolved config, independent of key order."""
    return sha256(_canonical(config))


def data_id(digests: Mapping[str, str]) -> str:
    """SHA-256 of the cache digests in use (dataset -> digest), independent of order."""
    return sha256(_canonical(dict(sorted(digests.items()))))


def code_version() -> dict[str, Any]:
    """Git commit of the code that runs, and whether it had uncommitted changes."""
    here = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


def new_run_id(config: str, seed: int) -> str:
    """``<UTC date>-<sha12>``: sorts by date, unique even for parallel runs of one config."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    entropy = (
        f"{config}|{seed}|{time.time_ns()}|{socket.gethostname()}|{os.getpid()}|"
        f"{os.urandom(16).hex()}"
    )
    return f"{stamp}-{sha256(entropy)[:12]}"
