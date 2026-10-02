"""Transports that carry encoded messages between processes (spec §5.1, §9.2).

A transport delivers bytes to an address (a node's inbox) and calls back
whoever listens on it::

    transport.on_receive("onionfl/<run_id>/<node>/inbox", callback)
    transport.start()
    transport.send(address, data)
    transport.stop()

In simulation the link layer of ``runtime.sim`` plays this part.
"""

from onion_fl.core.registry import Registry

transports = Registry("transport")

from onion_fl.transports import (  # noqa: E402, F401  (register the built-ins)
    memory,
    mqtt,
)
