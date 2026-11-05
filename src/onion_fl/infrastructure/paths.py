from __future__ import annotations

"""Physical paths and the effective profile of a logical link (I1 §6, §8).

Two hosts on the same machine talk for free; on the same site, through the
site's LAN; across sites, along the path of least latency. A path's profile adds
the latencies, combines the jitters as independent (√Σσ²), takes the narrowest
bandwidth in each direction and composes the losses as 1 − ∏(1 − pᵢ). A
one-segment path keeps that segment's profile exactly.
"""

import heapq
import math
from typing import Any

from onion_fl.infrastructure.model import Infrastructure, InfrastructureError

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


Hop = tuple[str, bool]  # a segment ("lan:<site>" or "link:<a>|<b>") and whether
# it is crossed from between[0] to between[1]


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
    if a == b:
        return [a], [(f"lan:{a}", True)]
    sites = site_path(infra, a, b)
    links = {frozenset(link.between): link for link in infra.network.links}
    crossed = []
    for x, y in zip(sites, sites[1:], strict=False):
        link = links[frozenset((x, y))]
        crossed.append(
            (f"link:{link.between[0]}|{link.between[1]}", link.between[0] == x)
        )
    return sites, crossed


def segment(infra: Infrastructure, key: str) -> Any:
    """The fields of a segment as the infrastructure declares them."""
    kind, _, name = key.partition(":")
    if kind == "lan":
        return infra.sites[name].lan
    a, _, b = name.partition("|")
    return next(link for link in infra.network.links if link.between == (a, b))


def route(
    infra: Infrastructure, src: str, dst: str, via: str | None = None
) -> tuple[list[str], list[Hop]]:
    """:func:`hops`, through host ``via`` (a broker) when given."""
    if via is None:
        return hops(infra, src, dst)
    first, second = hops(infra, src, via), hops(infra, via, dst)
    path = (
        first[0] + second[0][1:]
        if first[0][-1:] == second[0][:1]
        else first[0] + second[0]
    )
    return path, first[1] + second[1]


def effective(
    infra: Infrastructure, src: str, dst: str, via: str | None = None
) -> tuple[list[str], dict]:
    """Path from host ``src`` to host ``dst`` (through ``via``) and its profile,
    ``up`` = src -> dst. The profile has the fields of the runtime's ``LinkProfile``.
    """
    path, crossed = route(infra, src, dst, via)
    return path, combine([(segment(infra, k), forward) for k, forward in crossed])


def combine(segments: list[tuple[Any, bool]]) -> dict[str, Any]:
    """The profile of segments crossed in a row; none at all is the loopback."""
    if not segments:
        return dict(LOOPBACK)

    def bandwidth(up: bool) -> float | None:
        rates = [
            s.bandwidth_up if forward == up else s.bandwidth_down
            for s, forward in segments
        ]
        rates = [r for r in rates if r is not None]
        return min(rates) if rates else None

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
