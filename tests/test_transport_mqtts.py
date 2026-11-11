"""The MQTT transport over TLS, against a Mosquitto in Docker (issue #227).

The broker's certificates are made at test time (``tests.certs``), so the
broker runs here, with two TLS listeners: one authenticates clients by
certificate, the other by password. ``ONIONFL_MQTTS_PORT`` fixes the first
published port (the second is the next one); otherwise Docker picks them.
Skips without Docker or ``cryptography``.
"""

from __future__ import annotations

import os
import shutil
import socket
import ssl
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from onion_fl.transports import transports  # noqa: E402
from onion_fl.transports.mqtt import MqttTransport  # noqa: E402
from tests.certs import (  # noqa: E402
    BROKER_TLS,
    CERT_PORT,
    PASSWORD_PORT,
    Pki,
    mosquitto_tls,
    wait_for_mosquitto,
)

NAME = os.environ.get("ONIONFL_MQTTS_PREFIX", "onionfl-mqtts-test")
PORT = os.environ.get("ONIONFL_MQTTS_PORT")
USER, PASSWORD = "onion", f"pw-{uuid.uuid4().hex}"


def docker_up() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        done = subprocess.run(["docker", "info"], capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def _docker(*args: str) -> str:
    done = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=120
    )
    if done.returncode != 0:
        raise RuntimeError(f"docker {args[0]}: {done.stderr.strip()}")
    return done.stdout.strip()


@pytest.fixture(scope="module")
def broker(tmp_path_factory: pytest.TempPathFactory):
    """A TLS-only Mosquitto; yields its PKI and ``container port -> host:port``."""
    if not docker_up():
        pytest.skip("no Docker")
    folder = tmp_path_factory.mktemp("mqtts")
    pki = Pki(folder / "pki")
    command = mosquitto_tls(
        pki, folder / "broker", ["localhost", "127.0.0.1"], USER, PASSWORD
    )
    ports = ["", ""] if PORT is None else [PORT, str(int(PORT) + 1)]
    mount = f"type=bind,source={(folder / 'broker').as_posix()},target={BROKER_TLS}"
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True, timeout=60)
    _docker(
        "run", "-d", "--name", NAME, "--mount", mount,
        "-p", f"127.0.0.1:{ports[0]}:{CERT_PORT}",
        "-p", f"127.0.0.1:{ports[1]}:{PASSWORD_PORT}",
        "eclipse-mosquitto:2", *command,
    )  # fmt: skip
    try:
        address = {
            port: "127.0.0.1:"
            + _docker("port", NAME, str(port)).splitlines()[0].rsplit(":")[-1]
            for port in (CERT_PORT, PASSWORD_PORT)
        }
        wait_for_mosquitto(NAME)
        yield pki, address
    finally:
        subprocess.run(["docker", "rm", "-f", NAME], capture_output=True, timeout=60)


def tls(pki: Pki, client: str | None = None) -> dict:
    out = {"ca": str(pki.ca)}
    if client is not None:
        cert, key = pki.issue(client, [client])
        out |= {"cert": str(cert), "key": str(key)}
    return out


def roundtrip(**params) -> list[bytes]:
    inbox = f"onionfl/test-{uuid.uuid4().hex}/node/inbox"
    got, done = [], threading.Event()
    transport = MqttTransport(qos=1, connect_timeout=10, **params)
    transport.on_receive(inbox, lambda data: (got.append(data), done.set()))
    transport.start()
    try:
        transport.send(inbox, b"sealed")
        assert done.wait(10)
    finally:
        transport.stop()
    return got


def refused(**params) -> str:
    transport = MqttTransport(qos=1, connect_timeout=3, reconnect_min=0.2, **params)
    transport.on_receive("onionfl/never/x/inbox", lambda data: None)
    with pytest.raises(ConnectionError) as error:
        transport.start()
    return str(error.value)


def handshake(address: str, tls: dict, name: str | None = None) -> str | None:
    """The TLS error with the transport's own context, or ``None`` if it connects.

    paho retries a failed handshake quietly, so the transport only says the
    broker "did not answer"; this tells a TLS refusal from a broker that is down.
    """
    context = MqttTransport(broker=address, tls=tls)._tls_context()
    host, _, port = address.rpartition(":")
    with socket.create_connection((host, int(port)), timeout=10) as raw:
        try:
            with context.wrap_socket(raw, server_hostname=name or host) as secure:
                secure.settimeout(0.5)
                try:  # a TLS 1.3 server rejects the client after the handshake
                    secure.recv(1)
                except TimeoutError:
                    pass
        except ssl.SSLError as exc:
            return str(exc)
    return None


def test_mqtts_with_a_client_certificate_delivers(broker) -> None:
    pki, address = broker
    assert roundtrip(broker=address[CERT_PORT], tls=tls(pki, "edge-1")) == [b"sealed"]


def test_mqtts_with_a_password_from_a_file_or_the_environment_delivers(
    broker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pki, address = broker
    secret = tmp_path / "mqtt-password"
    secret.write_text(PASSWORD + "\n", encoding="utf-8")
    common = {"broker": address[PASSWORD_PORT], "tls": tls(pki), "username": USER}
    assert roundtrip(**common, password_file=str(secret)) == [b"sealed"]
    monkeypatch.setenv("ONIONFL_TEST_MQTT_PASSWORD", PASSWORD)
    got = roundtrip(**common, password_env="ONIONFL_TEST_MQTT_PASSWORD")
    assert got == [b"sealed"]


def test_mqtts_rejects_clients_without_credentials(
    broker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pki, address = broker
    by_cert, by_password = address[CERT_PORT], address[PASSWORD_PORT]
    stranger = Pki(tmp_path / "other", name="another-ca")
    cert, key = stranger.issue("stranger", ["stranger"])
    monkeypatch.setenv("ONIONFL_TEST_WRONG", "not-the-password")

    # No certificate, or one another CA signed: the TLS handshake fails, while
    # one this CA signed gets through the same listener.
    assert handshake(by_cert, tls(pki, "edge-2")) is None
    assert handshake(by_cert, tls(pki)) is not None
    assert "did not answer" in refused(broker=by_cert, tls=tls(pki))
    foreign = tls(pki) | {"cert": str(cert), "key": str(key)}
    assert handshake(by_cert, foreign) is not None
    assert "did not answer" in refused(broker=by_cert, tls=foreign)
    # No password, or a wrong one: the broker says no, and start fails at once.
    started = time.monotonic()
    assert "refused" in refused(broker=by_password, tls=tls(pki))
    assert time.monotonic() - started < 3
    wrong = {"username": USER, "password_env": "ONIONFL_TEST_WRONG"}
    assert "refused" in refused(broker=by_password, tls=tls(pki), **wrong)
    # Plain MQTT on a TLS port never connects, though TLS on it does.
    assert handshake(by_password, tls(pki)) is None
    assert "did not answer" in refused(broker=by_password)


def test_mqtts_checks_the_brokers_certificate(broker, tmp_path: Path) -> None:
    pki, address = broker
    stranger = Pki(tmp_path / "other", name="another-ca")
    assert handshake(address[PASSWORD_PORT], tls(pki)) is None
    error = handshake(address[PASSWORD_PORT], {"ca": str(stranger.ca)})
    assert "certificate verify failed" in str(error)
    assert "did not answer" in refused(
        broker=address[PASSWORD_PORT], tls={"ca": str(stranger.ca)}
    )


@pytest.mark.parametrize(
    "params",
    [
        {"password_env": "X"},  # a password needs a username
        {"username": "u", "password_env": "X", "password_file": "p"},
        {"username": "u", "password_env": "hunter 2"},  # a value, not a name
        {"tls": {"ca": "-----BEGIN CERTIFICATE-----\nMII"}},
        {"tls": {"ca": "ca.pem", "key": "k.key"}},
        {"password": "secret"},  # there is no field for the secret itself
    ],
)
def test_mqtt_takes_names_and_paths_never_secrets(params) -> None:
    with pytest.raises(ValueError):
        transports.create("mqtt", params)


def test_a_missing_password_variable_is_named(monkeypatch) -> None:
    monkeypatch.delenv("ONIONFL_TEST_UNSET", raising=False)
    transport = MqttTransport(username="u", password_env="ONIONFL_TEST_UNSET")
    with pytest.raises(ConnectionError, match="ONIONFL_TEST_UNSET is not set"):
        transport.start()
