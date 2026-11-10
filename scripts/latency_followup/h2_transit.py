from __future__ import annotations

"""H2c with the emulated latency measured to the transport's hand-over (no runtime queue).

    python scripts/latency_followup/h2_transit.py TRACED_EMULATED_RUNS SIM_RUNS
"""

import json
import sys
from collections import defaultdict

from onion_fl.experiment.fidelity import _events, load_cells, performance, timings


def cls(n):
    return (
        "cloud"
        if n == "cloud"
        else "eval"
        if n.startswith("test-")
        else "fog"
        if n.startswith("fog")
        else "edge"
    )


def transit(cell):
    sent, ts, td = {}, {}, {}
    for line in open(cell.path / "events.jsonl", encoding="utf-8"):
        e = json.loads(line)
        t = e["tags"] or {}
        if e["name"] == "message.sent":
            sent[t.get("msg_id")] = (e["node"], t["dst"])
        elif e["name"] == "trace.sent":
            ts[t["msg_id"]] = t["wall"][1]
        elif e["name"] == "trace.delivered":
            td[t["msg_id"]] = t["wall"][0]
    acc = defaultdict(list)
    for mid, link in sent.items():
        if mid in ts and mid in td:
            acc[link].append(td[mid] - ts[mid])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main(a, b):
    ca, _ = load_cells(a)
    cb, _ = load_cells(b)
    keys = sorted(ca.keys() & cb.keys())
    print(len(keys), "pairs")
    full = [(timings(_events(ca[k])), timings(_events(cb[k]))) for k in keys]
    tr = [
        ({"link_latency": transit(ca[k])}, {"link_latency": y["link_latency"]})
        for k, (x, y) in zip(keys, full, strict=True)
    ]
    for label, pairs in (
        ("delivered (as pre-registered)", full),
        ("transport hand-over", tr),
    ):
        for name, keep in (
            ("all links", lambda k: True),
            ("no same-host", lambda k: "eval" not in (cls(k[0]), cls(k[1]))),
        ):
            sub = [
                (
                    {
                        "link_latency": {
                            k: v for k, v in x["link_latency"].items() if keep(k)
                        }
                    },
                    {
                        "link_latency": {
                            k: v for k, v in y["link_latency"].items() if keep(k)
                        }
                    },
                )
                for x, y in pairs
            ]
            p = performance(sub)["link_latency"]
            print(
                f"{label:30} {name:13} A={p['a_mean'] * 1000:6.1f} B={p['b_mean'] * 1000:6.1f} e={p['relative_error']:+.3f} CI={p['relative_error_ci']}"
            )


if __name__ == "__main__":
    main(*sys.argv[1:3])
