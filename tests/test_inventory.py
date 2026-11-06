"""The inventory of real machines and the probe that checks them (issue #199).

The probe runs here as a child process over pipes, as the launcher runs it
over SSH on a machine.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from inspect import getsource
from pathlib import Path

import pytest

import onion_fl
from onion_fl.infrastructure import probe
from onion_fl.infrastructure.inventory import load_inventory, parse_inventory
from onion_fl.infrastructure.model import InfrastructureError

SRC = Path(onion_fl.__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- the inventory ---------------------------------------------------------------------


def test_an_inventory_loads_from_yaml_and_expands_the_key_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ONIONFL_TEST_KEY", str(tmp_path / "id_test"))
    path = tmp_path / "lab.yaml"
    path.write_text(
        "machines:\n"
        "  cloud-1: {address: 10.0.0.1, user: onion, key: '${ONIONFL_TEST_KEY}'}\n"
        "  field-pi-1: {address: 10.0.0.2, port: 2222, advertise: pi.lan}\n"
        "clock: {reference: ntp, server: ntp.example, max_offset: 0.02}\n",
        encoding="utf-8",
    )

    inv = load_inventory(path)

    cloud, pi = inv.machine("cloud-1"), inv.machine("field-pi-1")
    assert cloud.key_path() == str(tmp_path / "id_test")
    assert (pi.port, pi.peer_address, cloud.peer_address) == (
        2222,
        "pi.lan",
        "10.0.0.1",
    )
    assert (pi.python, pi.root, pi.grpc_port, pi.listen) == (
        "python3",
        "onionfl",
        50051,
        "0.0.0.0",
    )
    assert inv.clock.reference == "ntp" and inv.clock.max_offset == 0.02
    with pytest.raises(InfrastructureError, match="no machine for host 'fog-1'"):
        inv.machine("fog-1")


@pytest.mark.parametrize(
    "machine, where",
    [
        ({"address": "a", "password": "hunter2"}, "password"),
        ({"address": "a", "key": "-----BEGIN OPENSSH PRIVATE KEY-----"}, "path"),
        ({"address": "a", "env": {"1BAD": "x"}}, "environment variable"),
        ({"address": ""}, "address"),
    ],
)
def test_an_inventory_holds_no_secrets_and_says_what_is_wrong(machine, where) -> None:
    with pytest.raises(InfrastructureError, match=where):
        parse_inventory({"machines": {"cloud-1": machine}})


def test_a_key_path_with_an_unset_variable_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ONIONFL_NO_SUCH_KEY", raising=False)
    inv = parse_inventory(
        {"machines": {"a": {"address": "a", "key": "$ONIONFL_NO_SUCH_KEY"}}}
    )
    with pytest.raises(InfrastructureError, match="not set"):
        inv.machine("a").key_path()


# --- the probe ----------------------------------------------------------------------


def test_the_probe_answers_facts_time_and_reach_over_its_pipes(tmp_path: Path) -> None:
    data = tmp_path / "meta.json"
    data.write_bytes(b"{}")
    listening, port = socket.socket(), free_port()
    listening.bind(("127.0.0.1", 0))
    listening.listen()
    held = free_port()
    request = {
        "pins": ["numpy", "no-such-lib"],
        "files": [str(data), str(tmp_path / "absent")],
        "bind": ["127.0.0.1", held],
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            getsource(probe),
            json.dumps(request),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=dict(os.environ) | {"PYTHONPATH": str(SRC)},
    )
    try:
        said = json.loads(process.stdout.readline())
        process.stdin.write("time\n")
        process.stdin.flush()
        remote = float(process.stdout.readline())
        targets = [f"127.0.0.1:{listening.getsockname()[1]}", f"127.0.0.1:{port}"]
        targets.append(f"127.0.0.1:{held}")  # the probe's own gRPC port, held open
        process.stdin.write("reach " + json.dumps(targets) + "\n")
        process.stdin.flush()
        reach = json.loads(process.stdout.readline())
    finally:
        process.stdin.close()
        process.wait(timeout=10)
        listening.close()

    assert said["source"] == probe.source_digest(SRC / "onion_fl")
    assert said["pins"]["numpy"] and said["pins"]["no-such-lib"] is None
    assert said["files"][str(data)] and said["files"][str(tmp_path / "absent")] is None
    assert said["bind"] is True
    assert abs(remote - time.time()) < 5
    assert reach == {targets[0]: True, targets[1]: False, targets[2]: True}


def test_the_source_digest_ignores_line_endings(tmp_path: Path) -> None:
    for name, ending in (("lf", b"\n"), ("crlf", b"\r\n")):
        (tmp_path / name / "pkg").mkdir(parents=True)
        (tmp_path / name / "pkg" / "a.py").write_bytes(b"x = 1" + ending + b"y = 2")
    assert probe.source_digest(tmp_path / "lf") == probe.source_digest(
        tmp_path / "crlf"
    )


def test_sntp_reads_the_offset_of_a_server() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    ahead = 1.5

    def answer() -> None:
        _, sender = server.recvfrom(48)
        now = time.time() + ahead + probe.NTP_EPOCH
        stamp = struct.pack("!II", int(now), int((now % 1) * 2**32))
        server.sendto(b"\x1c" + 31 * b"\0" + stamp + stamp, sender)

    thread = threading.Thread(target=answer, daemon=True)
    thread.start()
    try:
        offset = probe.sntp_offset(f"127.0.0.1:{server.getsockname()[1]}")
    finally:
        thread.join(5)
        server.close()

    assert offset == pytest.approx(ahead, abs=0.1)
