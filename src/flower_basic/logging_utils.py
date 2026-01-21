from __future__ import annotations

import builtins
import datetime as _dt
import os
import sys


def enable_timestamped_print() -> None:
    """Prefix all print output with a timestamp for easier log tracing."""
    if getattr(builtins, "_flower_timestamp_print", False):
        return

    use_utc = os.getenv("FLOWER_LOG_UTC", "0").lower() in {"1", "true", "yes"}
    original_print = builtins.print

    def _ts_print(*args, **kwargs):
        sep = kwargs.pop("sep", " ")
        end = kwargs.pop("end", "\n")
        file = kwargs.pop("file", sys.stdout)
        flush = kwargs.pop("flush", False)

        ts = _dt.datetime.utcnow() if use_utc else _dt.datetime.now()
        stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
        message = sep.join(str(a) for a in args)
        lines = message.splitlines() if message else [""]

        for idx, line in enumerate(lines):
            line_end = end if idx == len(lines) - 1 else "\n"
            original_print(
                f"[{stamp}] {line}",
                end=line_end,
                file=file,
                flush=flush if idx == len(lines) - 1 else False,
            )

    builtins.print = _ts_print
    builtins._flower_timestamp_print = True
