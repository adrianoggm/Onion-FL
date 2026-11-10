from __future__ import annotations

"""The generator of the emulated backend: Docker Compose + netem (I1 §11, I10).

The I9 spike chose it (``docs/decisions/2026-11-06-i9-emulator.md``).
``compose_project`` turns an infrastructure and the hosts a run uses into a
Compose project and one setup script per container:

- **one container per host**, with the ``cpus`` and ``memory`` of its profile
  as Docker's CPU quota and memory limit;
- **one router per site** (when the run spans several), and one Docker
  network per site LAN and per physical link between two sites in use. A
  link's netem runs on the egress of the router at each end, so every host of
  a site shares it, as with ``network.bandwidth: shared`` in the simulation;
  the ``up`` rate on the ``between[0]`` end, the ``down`` rate on the other;
- **the LAN** of a site with several hosts is shaped on each host's egress, for
  destinations inside the site only: a path between sites crosses only the
  links, as in ``onion_fl.infrastructure.paths``;
- **routes** along the path of least latency (``paths.site_path``): each host
  reaches the other sites through its router, and each router the other
  sites' LANs through the next router on the path;
- **the MQTT broker** as a service in the network namespace of the host that
  ``services.mqtt_broker`` names, so it is behind that host's shaping.

What the shaping means:

- ``bandwidth`` is the rate netem shapes, Ethernet frames included; TCP gets
  about 95.6 % of it as goodput (the spike).
- netem's queue holds what is in flight on the link (its bandwidth-delay
  product, a segment without bandwidth counting as 1 Gbit/s) plus netem's
  default 1000 packets, so a burst queues rather than drops, as the simulator's
  unbounded queues do.
- ``loss`` is per packet: under MQTT over TCP it becomes retransmissions and
  delay, never a lost message as in the simulation.

The module only writes text: it never calls Docker (the backend does), so
every command it generates can be checked without it.
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv4Network
from itertools import combinations
from typing import Any

from onion_fl.infrastructure.model import (
    Infrastructure,
    InfrastructureError,
    Link,
    LinkFields,
)
from onion_fl.infrastructure.paths import site_path

MQTT_PORT = 1883
QUEUE = 1000  # netem's default queue (packets), kept on top of what is in flight
FRAME = 1514  # bytes of an Ethernet frame at a 1500-byte MTU
UNLIMITED = 1e9  # bit/s at which the queue of a segment without bandwidth is sized
SETUP = "/onionfl/setup"  # where every container sees its setup script
READY = "/tmp/onionfl-ready"  # a container's setup is done when this exists


def refusals(infra: Infrastructure) -> list[str]:
    """What the emulation cannot reproduce of ``infra``; empty when it can."""
    out = []
    if infra.network.schedule:
        out.append("network.schedule: link changes over time are not emulated yet")
    if infra.network.bandwidth == "dedicated":
        out.append(
            "network.bandwidth: dedicated cannot be emulated; a physical link is one "
            "netem queue, so its logical links share it (shared)"
        )
    segments = [(f"sites.{s}.lan", spec.lan) for s, spec in infra.sites.items()]
    segments += [(f"network.links[{i}]", x) for i, x in enumerate(infra.network.links)]
    for where, fields in segments:
        if fields.distribution == "lognormal" and fields.jitter > 0:
            out.append(f"{where}: netem has no lognormal delay; use normal or fixed")
    return out


def _jittered(fields: LinkFields) -> bool:
    """Whether the delay varies: as the simulator's ``LinkProfile`` decides."""
    return fields.distribution != "fixed" and fields.jitter > 0 and fields.latency > 0


def queue_limit(fields: LinkFields, rate: float | None) -> int:
    """netem's ``limit`` in packets: what is in flight, plus netem's default.

    In flight is the rate times the longest likely delay: the latency plus
    three standard deviations of its jitter, when netem applies one.
    """
    jitter = fields.jitter if _jittered(fields) else 0.0
    delay = fields.latency + 3 * jitter
    return QUEUE + math.ceil((rate or UNLIMITED) * delay / (8 * FRAME))


def netem(fields: LinkFields, rate: float | None) -> str:
    """The netem qdisc of one direction of a segment, ``rate`` in bit/s."""
    parts = ["netem", f"limit {queue_limit(fields, rate)}"]
    if fields.latency > 0:
        parts.append(f"delay {round(fields.latency * 1e6)}us")
        if _jittered(fields):
            parts.append(f"{round(fields.jitter * 1e6)}us")
            parts.append(f"distribution {fields.distribution}")
    if rate is not None:
        parts.append(f"rate {round(rate)}bit")
    if fields.loss > 0:
        parts.append(f"loss {fields.loss * 100:.6g}%")
    return " ".join(parts)


def slug(name: str) -> str:
    """A name Compose accepts for a service or a network."""
    return re.sub(r"[^a-z0-9.-]+", "-", name.lower()).strip("-.") or "x"


@dataclass(frozen=True)
class Layout:
    """Containers, networks and addresses of an emulation, before any Compose."""

    infra: Infrastructure
    subnet: IPv4Network  # every network of the emulation is a /24 of it
    hosts: dict[str, str]  # host -> its site, for every host container
    broker: str | None  # the host whose network namespace runs the broker
    sites: list[str]  # sites with a LAN: those on a path between the hosts
    links: list[Link]  # the physical links on those paths
    lans: dict[str, IPv4Network]  # site -> its LAN
    nets: dict[tuple[str, str], IPv4Network]  # a link's between -> its network
    addresses: dict[str, IPv4Address]  # host -> its LAN address
    routers: dict[str, IPv4Address]  # site -> its router's LAN address
    services: dict[str, str]  # host or site -> its Compose service

    @property
    def containers(self) -> int:
        return len(self.hosts) + len(self.routers) + (self.broker is not None)

    def summary(self) -> dict[str, Any]:
        return {
            "containers": self.containers,
            "hosts": len(self.hosts),
            "routers": len(self.routers),
            "broker": self.broker,
            "networks": len(self.lans) + len(self.nets),
        }


def layout(
    infra: Infrastructure,
    hosts: Sequence[str],
    broker: str | None = None,
    subnet: str = "10.195.0.0/16",
) -> Layout:
    """Where every container and network of an emulation goes.

    ``hosts`` are the hosts that run logical nodes; ``broker`` also gets a
    container when it is not one of them. Each network takes a /24 of
    ``subnet``: in a LAN, ``.1`` is Docker's bridge, ``.2`` the site's router
    and the hosts follow in order; in a link, ``.2`` is the router of
    ``between[0]`` and ``.3`` that of ``between[1]``. A run on one site needs
    no router.
    """
    known = infra.hosts()
    wanted = sorted(set(hosts) | ({broker} if broker else set()))
    for host in wanted:
        if host not in known:
            raise InfrastructureError(f"emulation: unknown host {host!r}")
    at = {host: known[host][0] for host in wanted}
    used = sorted(set(at.values()))
    sites, pairs = set(used), set()
    for a, b in combinations(used, 2):
        path = site_path(infra, a, b)
        sites |= set(path)
        pairs |= {frozenset(p) for p in zip(path, path[1:], strict=False)}
    links = [x for x in infra.network.links if frozenset(x.between) in pairs]
    blocks = IPv4Network(subnet).subnets(new_prefix=24)
    try:
        lans = {site: next(blocks) for site in sorted(sites)}
        nets = {x.between: next(blocks) for x in links}
    except StopIteration:
        raise InfrastructureError(
            f"emulation: {len(sites) + len(links)} networks do not fit in {subnet}"
        ) from None
    addresses = {}
    for site in sorted(sites):
        mine = [h for h in wanted if at[h] == site]
        if len(mine) > 252:
            raise InfrastructureError(
                f"emulation: site {site!r} has {len(mine)} hosts; a /24 takes 252"
            )
        addresses |= {h: lans[site][3 + i] for i, h in enumerate(mine)}
    routers = {s: lans[s][2] for s in sorted(sites)} if len(sites) > 1 else {}
    services = {h: f"host-{slug(h)}" for h in wanted}
    services |= {s: f"router-{slug(s)}" for s in routers}
    names = [*services.values(), *(["broker"] if broker else [])]
    if len(set(names)) != len(names):
        raise InfrastructureError(
            "emulation: two hosts or sites differ only in case or punctuation: "
            f"{sorted({n for n in names if names.count(n) > 1})}"
        )
    return Layout(
        infra,
        IPv4Network(subnet),
        at,
        broker,
        sorted(sites),
        links,
        lans,
        nets,
        addresses,
        routers,
        services,
    )


@dataclass(frozen=True)
class Emulation:
    """A Compose project and the setup script of each of its containers."""

    project: str
    layout: Layout
    compose: dict[str, Any]
    scripts: dict[str, str]  # service -> its setup script
    shaping: dict[str, list[str]]  # service -> its tc commands, as they run
    broker: str | None  # host:port where the nodes find the broker

    def container(self, host: str) -> str:
        """The container of ``host``, for ``docker exec``."""
        return f"{self.project}-{self.layout.services[host]}"

    @property
    def containers(self) -> list[str]:
        return [s["container_name"] for s in self.compose["services"].values()]


def _lan(site: str) -> str:
    return f"lan-{slug(site)}"


def _net(between: tuple[str, str]) -> str:
    return f"link-{slug(between[0])}-{slug(between[1])}"


def _end(lay: Layout, between: tuple[str, str], site: str) -> IPv4Address:
    """The address of ``site``'s router on the link ``between``."""
    return lay.nets[between][2 if between[0] == site else 3]


def _shape_lan(lay: Layout, host: str) -> list[str]:
    """tc for the LAN on a host's egress, for destinations inside its site only."""
    site = lay.hosts[host]
    if not any(s == site and h != host for h, s in lay.hosts.items()):
        return []  # nobody to reach on the LAN
    lan = lay.infra.sites[site].lan
    dev = f"$(dev {lay.addresses[host]})"
    return [
        f"tc qdisc add dev {dev} root handle 1: prio bands 2 priomap "
        + " ".join(["0"] * 16),
        f"tc qdisc add dev {dev} parent 1:2 handle 20: {netem(lan, lan.bandwidth_up)}",
        f"tc filter add dev {dev} parent 1: protocol ip prio 1 u32 "
        f"match ip dst {lay.lans[site]} flowid 1:2",
    ]


def _shape_links(lay: Layout, site: str) -> list[str]:
    """tc on a router's end of each of its links: the rate leaving that end."""
    out = []
    for link in lay.links:
        if site in link.between:
            ip = _end(lay, link.between, site)
            rate = link.bandwidth_up if link.between[0] == site else link.bandwidth_down
            out.append(f"tc qdisc add dev $(dev {ip}) root {netem(link, rate)}")
    return out


def _routes(lay: Layout, site: str) -> list[str]:
    """A router's route to every other LAN, through the next router on the path."""
    out = []
    for other in lay.sites:
        if other != site:
            nxt = site_path(lay.infra, site, other)[1]
            link = next(x for x in lay.links if set(x.between) == {site, nxt})
            via = _end(lay, link.between, nxt)
            out.append(f"ip route add {lay.lans[other]} via {via}")
    return out


def _script(title: str, commands: Sequence[str]) -> str:
    return "\n".join(
        [
            "#!/bin/sh",
            f"# {title}",
            "set -eu",
            'dev() { set -- $(ip -4 -o addr show to "$1/32"); echo "$2"; }',
            *commands,
            f"touch {READY}",
            "exec sleep infinity",
            "",
        ]
    )


def compose_project(
    lay: Layout,
    *,
    project: str,
    image: str,
    broker_image: str = "eclipse-mosquitto:2",
    volumes: Sequence[Mapping[str, Any]] = (),
    setup: str = ".",
) -> Emulation:
    """The Compose project of a layout.

    ``volumes`` (Compose long syntax) are mounted in every host container, and
    ``setup``, the folder where the caller writes the scripts, at ``SETUP`` in
    every container. A container runs its script, which shapes its
    interfaces, adds its routes, creates ``READY`` and waits; the backend then
    starts the node processes in the host containers with ``docker exec``.
    """
    mount = {"type": "bind", "source": setup, "target": SETUP, "read_only": True}
    base = {"image": image, "cap_add": ["NET_ADMIN"], "init": True}
    services: dict[str, Any] = {}
    scripts, shaping = {}, {}
    for site, ip in lay.routers.items():
        name = lay.services[site]
        attached = {_lan(site): ip} | {
            _net(x.between): _end(lay, x.between, site)
            for x in lay.links
            if site in x.between
        }
        shaping[name] = _shape_links(lay, site)
        scripts[name] = _script(
            f"router of site {site}", shaping[name] + _routes(lay, site)
        )
        services[name] = base | {
            "container_name": f"{project}-{name}",
            "sysctls": {"net.ipv4.ip_forward": 1},
            "networks": {n: {"ipv4_address": str(a)} for n, a in attached.items()},
            "volumes": [mount],
            "command": ["sh", f"{SETUP}/{name}.sh"],
        }
    for host, site in sorted(lay.hosts.items()):
        name = lay.services[host]
        profile = lay.infra.profile_of(host)
        limits: dict[str, Any] = {}
        if profile.cpus is not None:
            limits["cpus"] = profile.cpus
        if profile.memory is not None:
            limits["mem_limit"] = int(profile.memory)
        routes = []
        if site in lay.routers:  # the other sites, through the site's router
            routes.append(f"ip route add {lay.subnet} via {lay.routers[site]}")
        shaping[name] = _shape_lan(lay, host)
        scripts[name] = _script(f"host {host} of site {site}", shaping[name] + routes)
        services[name] = (
            base
            | {
                "container_name": f"{project}-{name}",
                "networks": {_lan(site): {"ipv4_address": str(lay.addresses[host])}},
                "volumes": [mount, *volumes],
                "command": ["sh", f"{SETUP}/{name}.sh"],
            }
            | limits
        )
    broker = None
    if lay.broker is not None:
        owner = lay.services[lay.broker]
        services["broker"] = {
            "image": broker_image,
            "container_name": f"{project}-broker",
            "network_mode": f"service:{owner}",
            "depends_on": [owner],
            "command": ["mosquitto", "-c", "/mosquitto-no-auth.conf"],
        }
        broker = f"{lay.addresses[lay.broker]}:{MQTT_PORT}"
    networks = {_lan(s): lay.lans[s] for s in lay.sites}
    networks |= {_net(between): net for between, net in lay.nets.items()}
    compose = {
        "name": project,
        "services": services,
        "networks": {
            name: {"ipam": {"config": [{"subnet": str(net), "gateway": str(net[1])}]}}
            for name, net in networks.items()
        },
    }
    return Emulation(project, lay, compose, scripts, shaping, broker)
