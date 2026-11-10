"""Throwaway certificates for the TLS tests (issue #227).

Made at test time with ``cryptography`` in a temporary folder: no private key
is ever committed. One CA signs a certificate per name; each certificate is
valid for TLS servers and clients alike, for the names and addresses given.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import subprocess
import time
from pathlib import Path

import pytest

x509 = pytest.importorskip("cryptography.x509")

from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID  # noqa: E402


def _key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


def _write_key(key: ec.EllipticCurvePrivateKey, path: Path) -> None:
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


def _name(common: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common)])


def _san(names: list[str]) -> x509.SubjectAlternativeName:
    entries: list[x509.GeneralName] = []
    for name in names:
        try:
            entries.append(x509.IPAddress(ipaddress.ip_address(name)))
        except ValueError:
            entries.append(x509.DNSName(name))
    return x509.SubjectAlternativeName(entries)


class Pki:
    """A CA in ``folder`` that issues certificates on demand."""

    def __init__(self, folder: Path, name: str = "onionfl-test-ca") -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        now = dt.datetime.now(dt.UTC)
        self._key = _key()
        self._name = _name(name)
        self._cert = (
            x509.CertificateBuilder()
            .subject_name(self._name)
            .issuer_name(self._name)
            .public_key(self._key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                True,
            )
            .sign(self._key, hashes.SHA256())
        )
        self.ca = folder / "ca.pem"
        self.ca.write_bytes(self._cert.public_bytes(serialization.Encoding.PEM))

    def issue(
        self, common: str, names: list[str], folder: Path | None = None
    ) -> tuple[Path, Path]:
        """``(cert, key)`` for ``common``, valid for ``names``, in ``folder``."""
        folder = folder or self.folder
        folder.mkdir(parents=True, exist_ok=True)
        now = dt.datetime.now(dt.UTC)
        key = _key()
        cert = (
            x509.CertificateBuilder()
            .subject_name(_name(common))
            .issuer_name(self._name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
            .add_extension(_san(names), False)
            .add_extension(
                x509.ExtendedKeyUsage(
                    [ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]
                ),
                False,
            )
            .sign(self._key, hashes.SHA256())
        )
        cert_path, key_path = folder / f"{common}.pem", folder / f"{common}.key"
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        _write_key(key, key_path)
        return cert_path, key_path


# --- a Mosquitto that speaks only TLS ----------------------------------------------------

CERT_PORT, PASSWORD_PORT = 8883, 8884  # inside the broker's container
BROKER_TLS = "/mosquitto/tls"


def mosquitto_tls(
    pki: Pki, folder: Path, names: list[str], user: str, password: str
) -> list[str]:
    """Config, certificate and command of a Mosquitto with two TLS listeners.

    ``CERT_PORT`` takes clients that show a certificate ``pki`` signed (their
    name is their user); ``PASSWORD_PORT`` takes ``user`` with ``password``.
    Neither takes anonymous clients. ``folder`` is mounted at ``BROKER_TLS``;
    the password file is made inside the container, never in the repository.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ca.pem").write_bytes(pki.ca.read_bytes())
    pki.issue("broker", names, folder)
    tls = "\n".join(
        [
            f"cafile {BROKER_TLS}/ca.pem",
            f"certfile {BROKER_TLS}/broker.pem",
            f"keyfile {BROKER_TLS}/broker.key",
            "tls_version tlsv1.2",
        ]
    )
    config = (
        "per_listener_settings true\n"
        f"listener {CERT_PORT}\n{tls}\n"
        "require_certificate true\nuse_identity_as_username true\n"
        "allow_anonymous false\n\n"
        f"listener {PASSWORD_PORT}\n{tls}\n"
        "allow_anonymous false\npassword_file /tmp/passwd\n"
    )
    (folder / "mosquitto.conf").write_text(config, encoding="utf-8", newline="\n")
    script = (
        'mosquitto_passwd -b -c /tmp/passwd "$1" "$2" && '
        "chown mosquitto:mosquitto /tmp/passwd && chmod 0600 /tmp/passwd && "
        f"exec mosquitto -c {BROKER_TLS}/mosquitto.conf"
    )
    return ["sh", "-c", script, "broker", user, password]


def wait_for_mosquitto(name: str, timeout: float = 30) -> None:
    """Mosquitto listens once its log says so, not when the container starts."""
    deadline = time.monotonic() + timeout
    while True:
        logs = subprocess.run(
            ["docker", "logs", name], capture_output=True, text=True, timeout=30
        )
        if "running" in logs.stdout + logs.stderr:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"mosquitto did not start: {logs.stderr}")
        time.sleep(0.3)
