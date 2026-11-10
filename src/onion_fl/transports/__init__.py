"""Transports that carry encoded messages between processes (spec §5.1, §9.2).

The built-ins: ``memory`` (one process), ``mqtt`` (through a broker),
``grpc`` (direct, without a broker; needs the ``grpc`` extra) and ``flower``
(the same links, each frame in a Flower ``Message``; the ``flower`` extra).

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
    flower,
    grpc,
    memory,
    mqtt,
)
