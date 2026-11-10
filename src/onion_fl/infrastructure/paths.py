from __future__ import annotations

"""Physical paths and the effective profile of a logical link (I1 §6, §8).

Two hosts on the same machine talk for free; on the same site, through the
site's LAN; across sites, along the path of least latency. A path's profile adds
the latencies, combines the jitters as independent (√Σσ²), takes the narrowest
bandwidth in each direction and composes the losses as 1 − ∏(1 − pᵢ). A
one-segment path keeps that segment's profile exactly. A host whose profile has
a ``bandwidth`` reaches the network through its own access segment
(``host:<host>``), crossed first when it sends and last when it receives.

A path through a broker is two legs, from the sender to the broker and from
the broker to the receiver, with the hop ``via:<host>`` between them. The
broker receives a whole message before it forwards it (store and forward), so
a message is transmitted once per leg: the path's bandwidth is that of its
legs in series, 1 / Σ(1 / bₗ), each leg's ``bₗ`` its narrowest segment. A
message through ``fog -> dc -> fog`` crosses the link twice and takes twice
its transmission time (measured in the calibration of issue #197).
"""

import heapq
import math
from collections.abc import Sequence
from typing import Any

from onion_fl.infrastructure.calibration import link_fields
from onion_fl.infrastructure.model import (
    Infrastructure,
    InfrastructureError,
    LinkFields,
)

LOOPBACK: dict[str, Any] = {
    "latency_s": 0.0,
    "jitter_s": 0.0,
    "distribution": "fixed",
    "bandwidth_up_bps": None,
    "bandwidth_down_bps": None,
    "loss": 0.0,
}


def site_path(infra: Infrastructure, src: str, dst: str) -> list[str]:
    """Sites from ``src`` to ``dst`` with the least total latency (ties by name)."""
    graph: dict[str, list[tuple[str, float]]] = {site: [] for site in infra.sites}
    for link in infra.network.links:
        a, b = link.between
        graph[a].append((b, link.latency))
        graph[b].append((a, link.latency))
    queue: list[tuple[float, list[str]]] = [(0.0, [src])]
    done: set[str] = set()
    while queue:
        cost, path = heapq.heappop(queue)
        here = path[-1]
        if here == dst:
            return path
        if here in done:
            continue
        done.add(here)
        for nxt, latency in graph[here]:
            if nxt not in done:
                heapq.heappush(queue, (cost + latency, [*path, nxt]))
    raise InfrastructureError(f"no physical path between sites {src!r} and {dst!r}")


Hop = tuple[str, bool]  # a segment ("lan:<site>", "link:<a>|<b>" or
# "host:<host>") and whether it is crossed from between[0] to between[1] (for a
# host's access, from the host out); or the broker ("via:<host>")


class _StoreAndForward:
    """The fields of the broker between two legs: nothing but the break."""

    latency = jitter = loss = 0.0
    distribution = "fixed"
    bandwidth_up = bandwidth_down = None


STORE = _StoreAndForward()


def hops(infra: Infrastructure, src: str, dst: str) -> tuple[list[str], list[Hop]]:
    """Sites (or the one host) from host ``src`` to host ``dst``, and the
    segments crossed on the way."""
    hosts = infra.hosts()
    for host in (src, dst):
        if host not in hosts:
            raise InfrastructureError(f"unknown host {host!r}")
    if src == dst:
        return [src], []
    (a, _), (b, _) = hosts[src], hosts[dst]
    first = [(f"host:{src}", True)] if infra.profile_of(src).bandwidth else []
    last = [(f"host:{dst}", False)] if infra.profile_of(dst).bandwidth else []
    if a == b:
        return [a], [*first, (f"lan:{a}", True), *last]
    sites = site_path(infra, a, b)
    links = {frozenset(link.between): link for link in infra.network.links}
    crossed = []
    for x, y in zip(sites, sites[1:], strict=False):
        link = links[frozenset((x, y))]
        crossed.append(
            (f"link:{link.between[0]}|{link.between[1]}", link.between[0] == x)
        )
    return sites, [*first, *crossed, *last]


def segment(infra: Infrastructure, key: str) -> Any:
    """The fields of a segment as the simulation sees them: as the
    infrastructure declares them, or as calibrated for a link that cites a
    calibration."""
    kind, _, name = key.partition(":")
    if kind == "via":
        return STORE
    if kind == "lan":
        return infra.sites[name].lan
    if kind == "host":
        return LinkFields(bandwidth=infra.profile_of(name).bandwidth)
    a, _, b = name.partition("|")
    link = next(link for link in infra.network.links if link.between == (a, b))
    return link_fields(infra, link)


def route(
    infra: Infrastructure, src: str, dst: str, via: str | None = None
) -> tuple[list[str], list[Hop]]:
    """:func:`hops`, through host ``via`` (a broker) when given: its two legs,
    with ``via:<host>`` between them."""
    if via is None:
        return hops(infra, src, dst)
    first, second = hops(infra, src, via), hops(infra, via, dst)
    path = (
        first[0] + second[0][1:]
        if first[0][-1:] == second[0][:1]
        else first[0] + second[0]
    )
    return path, [*first[1], (f"via:{via}", True), *second[1]]


def legs(items: Sequence[tuple[Any, bool]]) -> list[list[tuple[Any, bool]]]:
    """``items`` (hops, or segments with their direction) split at the broker;
    a leg crossing nothing is left out."""
    out: list[list[tuple[Any, bool]]] = [[]]
    for item in items:
        here = item[0]
        if here is STORE or (isinstance(here, str) and here.startswith("via:")):
            out.append([])
        else:
            out[-1].append(item)
    return [leg for leg in out if leg]


def effective(
    infra: Infrastructure, src: str, dst: str, via: str | None = None
) -> tuple[list[str], dict]:
    """Path from host ``src`` to host ``dst`` (through ``via``) and its profile,
    ``up`` = src -> dst. The profile has the fields of the runtime's ``LinkProfile``.
    """
    path, crossed = route(infra, src, dst, via)
    return path, combine([(segment(infra, k), forward) for k, forward in crossed])


def combine(segments: list[tuple[Any, bool]]) -> dict[str, Any]:
    """The profile of segments crossed in a row, with ``STORE`` between the
    legs of a path through a broker; none at all is the loopback."""
    parts = legs(segments)
    segments = [x for leg in parts for x in leg]
    if not segments:
        return dict(LOOPBACK)

    def bandwidth(up: bool) -> float | None:
        narrowest = []
        for leg in parts:
            rates = [
                s.bandwidth_up if forward == up else s.bandwidth_down
                for s, forward in leg
            ]
            rates = [r for r in rates if r is not None]
            if rates:
                narrowest.append(min(rates))
        if len(narrowest) < 2:  # one leg: exactly its narrowest segment
            return narrowest[0] if narrowest else None
        return 1.0 / sum(1.0 / r for r in narrowest)

    if len(segments) == 1:  # exactly the segment: no float drift in the loss
        only = segments[0][0]
        jitter, distribution, loss = only.jitter, only.distribution, only.loss
    else:
        jitter = math.sqrt(sum(s.jitter**2 for s, _ in segments))
        kinds = {s.distribution for s, _ in segments if s.jitter > 0}
        # ponytail: mixed jitter distributions become normal; a sum of lognormals
        # has no closed form, so model it as a distribution only if it matters.
        distribution = (
            "fixed" if not kinds else kinds.pop() if len(kinds) == 1 else "normal"
        )
        loss = 1 - math.prod(1 - s.loss for s, _ in segments)
    return {
        "latency_s": sum(s.latency for s, _ in segments),
        "jitter_s": jitter,
        "distribution": distribution,
        "bandwidth_up_bps": bandwidth(True),
        "bandwidth_down_bps": bandwidth(False),
        "loss": loss,
    }
