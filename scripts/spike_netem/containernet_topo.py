"""I9 spike, option B: the same 1 cloud, 2 fogs and 4 edges on Containernet.

Runs inside the official ``containernet/containernet`` image (see README.md).
Each physical link in ``links.json`` is a direct ``TCLink`` (a veth pair whose
two ends Mininet shapes with HTB + netem) between two Docker hosts, named
``mn.<host>`` so that ``measure.py --option containernet`` can drive them.
With ``cpu`` instead of a shaping set it starts three unlinked hosts limited
to 0.5, 1 and 2 CPUs (Docker CFS quota, as Containernet sets it), for
``cpubench.py``. The network stays up until ``/tmp/stop`` exists.

No ``from __future__ import annotations``: the image ships Python 3.6.
"""

import json
import os
import sys
import time

# isort: off
# mininet.link and mininet.node import each other: mininet.net must come first.
from mininet.net import Containernet
from mininet.link import TCLink
from mininet.log import setLogLevel

# isort: on

IMAGE = "onionfl-spike-netem:latest"
CPUS = (0.5, 1.0, 2.0)
HOSTS = ("cloud", "fog1", "fog2", "e1", "e2", "e3", "e4")


def main(set_: str) -> None:
    links = json.load(open(os.path.join(os.path.dirname(__file__), "links.json")))
    net = Containernet(controller=None)
    if set_ == "cpu":
        for cpus in CPUS:
            net.addDocker(
                f"cpu{int(cpus * 100)}",
                ip=None,
                dimage=IMAGE,
                cpu_period=100000,
                cpu_quota=int(cpus * 100000),
            )
        net.start()
        return serve(net)
    nodes = {h: net.addDocker(h, ip=None, dimage=IMAGE) for h in HOSTS}
    for link in links["links"].values():
        params = {"delay": f"{link['latency_ms']}ms", "bw": link["bw_mbit"]}
        if set_ == "loss":
            params["loss"] = link["loss_pct"]
        net.addLink(
            nodes[link["lower"]],
            nodes[link["upper"]],
            cls=TCLink,
            params1={"ip": link["lower_ip"] + "/24"},
            params2={"ip": link["upper_ip"] + "/24"},
            **params,
        )
    net.start()
    for fog in ("fog1", "fog2"):
        nodes[fog].cmd("sysctl -w net.ipv4.ip_forward=1")
    serve(net)


def serve(net: Containernet) -> None:
    print("READY", flush=True)
    while not os.path.exists("/tmp/stop"):
        time.sleep(1)
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    main(sys.argv[1])
