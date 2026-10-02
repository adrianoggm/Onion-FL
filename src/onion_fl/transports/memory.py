from __future__ import annotations

"""In-process transport: a named bus shared by every transport of the process.

For tests and for running the real runtime on one machine without a broker.
"""

import threading
from collections.abc import Callable
from typing import ClassVar

from pydantic import BaseModel, Field

from onion_fl.transports import transports

Callback = Callable[[bytes], None]


class MemoryParams(BaseModel):
    bus: str = Field(
        "default", description="Transports with the same bus see each other"
    )


@transports.register(
    "memory",
    title="En memoria",
    description="Bus dentro del proceso; para tests y ejecuciones locales sin broker.",
    params=MemoryParams,
)
class MemoryTransport:
    _buses: ClassVar[dict[str, dict[str, list[Callback]]]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, bus: str = "default") -> None:
        self.bus = bus
        self.stats = {"sent": 0, "bytes_sent": 0, "received": 0, "bytes_received": 0}

    def _subscribers(self) -> dict[str, list[Callback]]:
        return MemoryTransport._buses.setdefault(self.bus, {})

    def on_receive(self, address: str, callback: Callback) -> None:
        def counted(data: bytes) -> None:
            self.stats["received"] += 1
            self.stats["bytes_received"] += len(data)
            callback(data)

        with self._lock:
            self._subscribers().setdefault(address, []).append(counted)

    def start(self) -> None:
        pass

    def send(self, address: str, data: bytes) -> None:
        self.stats["sent"] += 1
        self.stats["bytes_sent"] += len(data)
        with self._lock:
            listeners = list(self._subscribers().get(address, []))
        for callback in listeners:
            callback(data)

    def stop(self) -> None:
        pass
