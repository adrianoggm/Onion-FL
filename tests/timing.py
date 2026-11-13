"""Wall-clock bounds for tests, stretched on a slow machine.

``ONIONFL_TEST_SLACK`` (default 1) multiplies every bound, so a loaded CI runner
can widen them without the tests losing their point: they still catch a hang.
"""

from __future__ import annotations

import os


def slack(seconds: float) -> float:
    """``seconds`` times ``ONIONFL_TEST_SLACK``."""
    return seconds * float(os.environ.get("ONIONFL_TEST_SLACK", "1"))
