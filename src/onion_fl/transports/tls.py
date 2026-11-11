from __future__ import annotations

"""TLS settings shared by the MQTT and gRPC transports (issue #227).

A link names files and environment variables, never their contents: the
link is part of the config, so whatever it holds ends up in the
``config_id``, the ``topology_id`` and ``run.json``. A path is read when the
transport starts, on the machine that opens the connection, with ``~`` and
``$VARS`` expanded there; so one YAML serves machines that keep their own
certificate at the same path, or under a path the machine's environment
names.
"""

import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def path_only(value: str | None) -> str | None:
    """Refuse what looks like a PEM or a secret pasted where a path goes."""
    if value is None:
        return None
    if "-----BEGIN" in value or "\n" in value or "\r" in value:
        raise ValueError("give the path of the file, not its contents")
    if not value.strip():
        raise ValueError("the path is empty")
    return value


def env_name(value: str | None) -> str | None:
    """An environment variable's name, not its value."""
    if value is not None and not ENV_NAME.match(value):
        raise ValueError(
            f"{value!r} is not an environment variable name; "
            "name the variable that holds the secret, not the secret"
        )
    return value


def expand(path: str) -> Path:
    """``~`` and ``$VARS`` expanded on the machine that reads the file."""
    return Path(os.path.expanduser(os.path.expandvars(path)))


def read(path: str, what: str) -> bytes:
    """The file's bytes; a ``ConnectionError`` naming the setting otherwise."""
    target = expand(path)
    try:
        return target.read_bytes()
    except OSError as exc:
        raise ConnectionError(f"TLS {what} {path!r} ({target}): {exc}") from None


class TlsParams(BaseModel):
    """The certificates of one end of a TLS link, as paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ca: str = Field(
        description="Ruta del certificado de la CA (PEM) que firma al otro extremo"
    )
    cert: str | None = Field(
        None, description="Ruta del certificado propio (PEM), para TLS mutuo"
    )
    key: str | None = Field(
        None, description="Ruta de la clave privada propia (PEM, sin contraseña)"
    )

    _paths = field_validator("ca", "cert", "key")(lambda v: path_only(v))

    @model_validator(mode="after")
    def _pair(self) -> TlsParams:
        if (self.cert is None) != (self.key is None):
            raise ValueError("tls: give cert and key together")
        return self
