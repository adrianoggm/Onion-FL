from __future__ import annotations

"""Operator orders from outside the tree (continuum spec §11, #203).

An operator may promote a version, roll one back, and pause or resume a
simulation. Each has an id and a secret token, in a YAML file that
``ONIONFL_OPERATORS`` names (never in the config, so no secret reaches a
``config_id``)::

    alice: "<a long random token>"

- **Signed orders.** An order is the ``meta`` of a ``control`` message, signed
  with the operator's token: an HMAC-SHA256 over the operator, the run it is
  for (``audience``, its ``run_id``), a single-use ``nonce`` and the order.
  The coordinator checks it whatever the transport carried it, so a sender that
  only says it is an operator, a forged order, a replayed one or one meant for
  another run is refused.
- **The inbox.** The Studio API appends signed orders to
  ``runs/<run_id>/control.jsonl``. A simulation reads them as it runs
  (``LiveControl``): it pauses and resumes itself, and hands promotions and
  rollbacks to the coordinator at the virtual time asked (``at``) or at once.
  ``control.json`` says whether it listens, whether it is paused, and its clock.

Without operators nothing listens, and every order is refused: it fails closed.
"""

import hashlib
import heapq
import hmac
import json
import math
import os
import re
import secrets
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from onion_fl.core.message import Message

ENV = "ONIONFL_OPERATORS"
VERSION_ORDERS = ("promote", "rollback")  # the coordinator's
RUN_ORDERS = ("pause", "resume")  # the simulation's
INBOX, STATE = "control.jsonl", "control.json"
NODE = "control"  # the node the simulation's control events are recorded at
MIN_TOKEN = 16
_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


class OperatorError(ValueError):
    """An operators file or an order that cannot be used."""


def load_operators(path: str | Path | None = None) -> dict[str, str] | None:
    """The operators in ``path``, or in the file ``ONIONFL_OPERATORS`` names;
    None when neither is set."""
    source = path or os.environ.get(ENV)
    if not source:
        return None
    raw = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise OperatorError(f"{ENV}: {source} must map operator ids to tokens")
    out: dict[str, str] = {}
    for operator, token in raw.items():
        if not isinstance(operator, str) or not _ID.fullmatch(operator):
            raise OperatorError(f"{ENV}: bad operator id {operator!r}")
        if not isinstance(token, str) or len(token) < MIN_TOKEN:
            raise OperatorError(
                f"{ENV}: the token of {operator} must have {MIN_TOKEN} characters "
                "or more"
            )
        out[operator] = token
    return out


def authenticate(operators: Mapping[str, str] | None, token: str | None) -> str | None:
    """The operator whose token ``token`` is, compared in constant time."""
    found = None
    for operator, secret in (operators or {}).items():
        if token and hmac.compare_digest(secret.encode(), token.encode()):
            found = operator
    return found


def _mac(token: str, operator: str, audience: str, nonce: str, order: Any) -> str:
    body = json.dumps(
        {"operator": operator, "audience": audience, "nonce": nonce, "order": order},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hmac.new(token.encode(), body.encode(), hashlib.sha256).hexdigest()


def _unsigned(order: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in order.items() if k != "auth"}


def sign(
    order: Mapping[str, Any],
    *,
    operator: str,
    token: str,
    audience: str,
    nonce: str | None = None,
) -> dict[str, Any]:
    """``order`` with its ``auth``: a fresh nonce and the operator's signature."""
    nonce = nonce or secrets.token_hex(16)
    plain = _unsigned(order)
    return plain | {
        "auth": {"nonce": nonce, "mac": _mac(token, operator, audience, nonce, plain)}
    }


def refusal(
    order: Mapping[str, Any],
    *,
    operator: str,
    operators: Mapping[str, str],
    audience: str,
) -> str | None:
    """Why ``order`` from ``operator`` is refused; None when they signed it for
    ``audience``. The nonce is the caller's to keep single-use."""
    token = operators.get(operator)
    if token is None:
        return f"{operator} is not an operator"
    auth = order.get("auth")
    if not (
        isinstance(auth, Mapping)
        and isinstance(auth.get("nonce"), str)
        and isinstance(auth.get("mac"), str)
    ):
        return "unsigned order"
    try:
        expected = _mac(token, operator, audience, auth["nonce"], _unsigned(order))
    except (TypeError, ValueError):
        return "unsigned order"
    if not hmac.compare_digest(expected, auth["mac"]):
        return "bad signature"
    return None


def parse_order(body: Mapping[str, Any]) -> dict[str, Any]:
    """The order of an API request ``{order, version?, at?}``, in the coordinator's
    form: ``{"promote": v}``, ``{"rollback": v or None}``, ``{"pause": True}`` or
    ``{"resume": True}``, with ``at`` (virtual seconds) when it is scheduled."""
    name = body.get("order")
    if name not in VERSION_ORDERS + RUN_ORDERS:
        raise OperatorError(
            f"order must be one of {list(VERSION_ORDERS + RUN_ORDERS)}, got {name!r}"
        )
    version = body.get("version")
    is_version = isinstance(version, int) and not isinstance(version, bool)
    if name == "promote" and not (is_version and version >= 0):
        raise OperatorError("promote needs the version to promote")
    if name == "rollback" and not (version is None or (is_version and version >= 0)):
        raise OperatorError("rollback takes a version, or none for the previous one")
    if name in RUN_ORDERS and version is not None:
        raise OperatorError(f"{name} takes no version")
    order: dict[str, Any] = {name: version if name in VERSION_ORDERS else True}
    at = body.get("at")
    if at is not None:
        if name == "resume":
            raise OperatorError("resume is at once: a paused run's clock stands still")
        if (
            isinstance(at, bool)
            or not isinstance(at, int | float)
            or not math.isfinite(at)
            or at < 0
        ):
            raise OperatorError(f"at must be a virtual time >= 0, got {at!r}")
        order["at"] = float(at)
    return order


def name_of(order: Mapping[str, Any]) -> str | None:
    return next((k for k in VERSION_ORDERS + RUN_ORDERS if k in order), None)


def append_order(folder: str | Path, operator: str, order: Mapping[str, Any]) -> None:
    """Add a signed order to a run's inbox."""
    line = {
        "operator": operator,
        "order": dict(order),
        "issued_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with (Path(folder) / INBOX).open("a", encoding="utf-8") as out:
        out.write(json.dumps(line, sort_keys=True, allow_nan=False) + "\n")


def read_inbox(folder: str | Path) -> list[dict[str, Any]]:
    """The orders in a run's inbox, without their signatures."""
    path = Path(folder) / INBOX
    out = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        try:
            entry = json.loads(line)
            order = entry.get("order") if isinstance(entry, Mapping) else None
        except json.JSONDecodeError:
            continue
        if isinstance(order, Mapping):
            out.append(entry | {"order": _unsigned(order)})
    return out


def read_state(folder: str | Path) -> dict[str, Any] | None:
    """What ``control.json`` says, or None when the run never listened."""
    path = Path(folder) / STATE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class LiveControl:
    """Runs a ``SimRuntime`` while it reads the run's inbox (see the module).

    Every line is checked as the coordinator would (signature, audience and a
    single-use nonce). Pauses and resumes apply here; promotions and rollbacks
    reach the coordinator as ``control`` messages from the operator, which
    checks them again. An order with ``at`` applies when the virtual clock gets
    there, so a schedule is reproducible; one whose time has passed applies at
    once (``late``); a promotion or rollback is handled at once, paused or not.
    The inbox is read every ``poll`` wall seconds, and while paused, when
    ``idle`` is called first (to write out the buffered events, ``Run.sync``).
    """

    def __init__(
        self,
        folder: str | Path,
        operators: Mapping[str, str],
        audience: str,
        coordinator: str,
        *,
        paused: bool = False,
        poll: float = 0.2,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        idle: Callable[[], None] = lambda: None,
    ) -> None:
        self.folder, self.operators = Path(folder), dict(operators)
        self._idle = idle
        self.audience, self.coordinator = audience, coordinator
        self.paused, self.poll = paused, poll
        self._sleep, self._clock = sleep, clock
        self._offset = 0  # bytes of the inbox read so far
        self._read_lines = 0
        self._nonces: set[str] = set()
        self._due: list[tuple[float, int, str, dict[str, Any]]] = []
        self._next_read = -math.inf

    def run(self, runtime: Any) -> None:
        self._write(runtime, listening=True)
        if self.paused:
            runtime.record(NODE, "control.paused", None, operator=None)
        try:
            while True:
                self._read(runtime)
                if self.paused:
                    self._idle()
                    self._sleep(self.poll)
                    continue
                upcoming = runtime.next_time()
                if upcoming is None:
                    break  # the run is over; what is still scheduled never applies
                if self._due and self._due[0][0] <= upcoming:
                    at, _, operator, order = heapq.heappop(self._due)
                    runtime.run(until=max(at, runtime.now))
                    self._apply(runtime, operator, order, late=False)
                    continue
                runtime.run(until=upcoming)
        finally:
            self._write(runtime, listening=False)

    # --- reading the inbox --------------------------------------------------------

    def _read(self, runtime: Any) -> None:
        if not self.paused and self._clock() < self._next_read:
            return
        self._next_read = self._clock() + self.poll
        path = self.folder / INBOX
        if not path.is_file():
            return
        with path.open("rb") as inbox:
            inbox.seek(self._offset)
            chunk = inbox.read()
        complete = chunk[: chunk.rfind(b"\n") + 1]  # a line half-written waits
        self._offset += len(complete)
        lines = complete.decode("utf-8", errors="replace").splitlines()
        for line in lines:
            self._read_lines += 1
            self._take(runtime, line)
        if lines:
            self._write(runtime, listening=True)

    def _take(self, runtime: Any, line: str) -> None:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            entry = None
        operator = entry.get("operator") if isinstance(entry, Mapping) else None
        order = entry.get("order") if isinstance(entry, Mapping) else None
        if not isinstance(operator, str) or not isinstance(order, Mapping):
            self._refuse(runtime, None, "not an order")
            return
        why = refusal(
            order, operator=operator, operators=self.operators, audience=self.audience
        )
        name = name_of(order)
        if why is None and order["auth"]["nonce"] in self._nonces:
            why = "replayed order"
        if why is None and name is None:
            why = "not an order"
        at = order.get("at")
        if why is None and at is not None and not _finite(at):
            why = "bad time"
        if why is not None:
            self._refuse(runtime, operator, why)
            return
        self._nonces.add(order["auth"]["nonce"])
        plain = _unsigned(order)
        runtime.record(NODE, "control.received", None, operator=operator, order=plain)
        if at is not None and at > runtime.now:
            heapq.heappush(self._due, (float(at), self._read_lines, operator, order))
        else:
            self._apply(runtime, operator, order, late=at is not None)

    def _refuse(self, runtime: Any, operator: str | None, reason: str) -> None:
        runtime.record(NODE, "control.refused", None, operator=operator, reason=reason)

    # --- applying an order --------------------------------------------------------

    def _apply(
        self, runtime: Any, operator: str, order: Mapping[str, Any], late: bool
    ) -> None:
        name, tags = name_of(order), {"operator": operator, "late": late}
        if name == "pause":
            if not self.paused:
                self.paused = True
                runtime.record(NODE, "control.paused", None, **tags)
        elif name == "resume":
            if self.paused:
                self.paused = False
                runtime.record(NODE, "control.resumed", None, **tags)
            else:
                self._refuse(runtime, operator, "not paused")
        else:
            runtime.inject(
                Message(kind="control", src=operator, dst=self.coordinator, meta=order)
            )
            # handled now, so a paused run shows what it did; the clock stays
            runtime.run(until=runtime.now)
        self._write(runtime, listening=True)

    def _write(self, runtime: Any, listening: bool) -> None:
        state = {
            "listening": listening,
            "paused": self.paused and listening,
            "t": runtime.now,
            "read": self._read_lines,
            "scheduled": [
                {"at": at, "operator": operator, "order": _unsigned(order)}
                for at, _, operator, order in sorted(self._due)
            ],
        }
        path, temporary = self.folder / STATE, self.folder / f".{STATE}.tmp"
        temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        for attempt in range(20):  # a reader may hold it open for a moment (Windows)
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                time.sleep(0.01 * (attempt + 1))


def _finite(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
