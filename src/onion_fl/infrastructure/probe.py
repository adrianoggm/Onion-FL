from __future__ import annotations

"""What a machine says about itself before a distributed run (I14).

The launcher sends this module's source over SSH and runs it with the
machine's Python (``python -c <source> <request>``), so it uses the standard
library only and imports nothing of ``onion_fl`` but the package it checks.
It prints one JSON line with the facts the request asks for:

- ``python``, ``machine`` and ``system``;
- ``onion_fl``: its version and ``source``, the digest of its ``.py`` files
  with their line endings normalised (a Windows checkout and a Linux one of
  the same commit give the same digest);
- ``pins``: the version of each numeric library, ``None`` when missing;
- ``files``: the SHA-256 of each file asked for, ``None`` when absent;
- ``bind``: whether the port its gRPC server takes is free; if it is, the
  probe keeps listening on it until the session ends, so that the other
  machines can check they reach it;
- ``ntp``: the offset of its clock from an NTP server, when asked.

Then it answers the launcher, one line per line: ``time`` gets its
``time.time()`` (the launcher measures the clock offset as NTP does, over the
SSH session it already holds, and keeps the sample with the shortest round
trip), and ``reach ["host:port", ...]`` gets whether each accepts a TCP
connection. The session ends when the launcher closes it.
"""

import hashlib
import json
import platform
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

NTP_EPOCH = 2208988800  # seconds from 1900-01-01 to 1970-01-01


def source_digest(package: str | Path) -> str:
    """SHA-256 of the ``.py`` files under ``package``, LF line endings."""
    root = Path(package)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if "__pycache__" in rel:
            continue
        digest.update(rel.encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def sntp_offset(server: str, timeout: float = 2.0) -> float:
    """Offset of this clock from ``server`` (``host`` or ``host:port``), in s."""
    host, _, port = server.partition(":")
    packet = b"\x1b" + 47 * b"\0"  # LI 0, version 3, mode 3 (client)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        t0 = time.time()
        sock.sendto(packet, (host, int(port or 123)))
        data, _ = sock.recvfrom(1024)
        t3 = time.time()
    if len(data) < 48:
        raise ValueError(f"{server}: short NTP answer")
    t1 = _ntp_time(data[32:40])  # receive timestamp
    t2 = _ntp_time(data[40:48])  # transmit timestamp
    return ((t1 - t0) + (t2 - t3)) / 2


def _ntp_time(raw: bytes) -> float:
    seconds, fraction = struct.unpack("!II", raw)
    return seconds - NTP_EPOCH + fraction / 2**32


def _reachable(target: str, timeout: float = 2.0) -> bool:
    host, _, port = target.rpartition(":")
    host = host.removeprefix("[").removesuffix("]")  # a bracketed IPv6
    try:
        socket.create_connection((host, int(port)), timeout).close()
    except OSError:
        return False
    return True


def listen(host: str, port: int) -> socket.socket | None:
    """Hold ``port`` open, accepting and closing connections; None if taken."""
    import threading

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.listen(16)
    except OSError:
        sock.close()
        return None

    def accept() -> None:
        while True:
            try:
                conn, _ = sock.accept()
            except OSError:  # closed: the session is over
                return
            conn.close()

    threading.Thread(target=accept, daemon=True).start()
    return sock


def facts(request: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
    }
    try:
        import onion_fl

        out["onion_fl"] = getattr(onion_fl, "__version__", None)
        out["source"] = source_digest(Path(onion_fl.__file__).parent)
    except Exception as exc:  # what is wrong is the answer
        out["onion_fl"], out["source"] = None, None
        out["error"] = f"import onion_fl: {exc!r}"
    from importlib.metadata import PackageNotFoundError, version

    pins: dict[str, str | None] = {}
    for name in request.get("pins", []):
        try:
            pins[name] = version(name).split("+")[0]
        except PackageNotFoundError:
            pins[name] = None
    out["pins"] = pins
    out["files"] = {
        f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
        if Path(f).is_file()
        else None
        for f in request.get("files", [])
    }
    if request.get("ntp"):
        try:
            out["ntp"] = sntp_offset(request["ntp"])
        except (OSError, ValueError) as exc:
            out["ntp"], out["ntp_error"] = None, repr(exc)
    return out


def main(argv: list[str]) -> None:
    request = json.loads(argv[1]) if len(argv) > 1 else {}
    out = facts(request)
    held = None
    if request.get("bind"):
        host, port = request["bind"]
        held = listen(host, int(port))
        out["bind"] = held is not None
    print(json.dumps(out), flush=True)
    for line in sys.stdin:
        command, _, rest = line.strip().partition(" ")
        if command == "time":
            print(repr(time.time()), flush=True)
        elif command == "reach":
            targets = json.loads(rest or "[]")
            print(json.dumps({t: _reachable(t) for t in targets}), flush=True)
        else:
            print(json.dumps({"error": f"unknown command {command!r}"}), flush=True)
    if held is not None:
        held.close()


if __name__ == "__main__":
    main(sys.argv)
