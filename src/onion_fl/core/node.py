from __future__ import annotations

"""Base class for every role: coordinator, aggregator, edge (spec §5.1, §6)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onion_fl.core.context import Context
    from onion_fl.core.message import Message


class Node:
    """A state machine driven by messages and timers.

    A node has no threads, sockets or clock of its own: everything it does to
    the outside world goes through the ``Context`` the runtime passes in, which
    is what lets the same node run in simulation and over real transports.
    Subclasses override only the handlers they need.
    """

    def __init__(self, node_id: str) -> None:
        self.id = node_id

    def on_start(self, ctx: Context) -> None:
        """Called once when the runtime starts the node."""

    def on_message(self, msg: Message, ctx: Context) -> None:
        """Called for every message addressed to this node."""

    def on_timer(self, name: str, ctx: Context) -> None:
        """Called when a timer armed with ``ctx.set_timer(delay, name)`` fires."""
