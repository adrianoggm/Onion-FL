from __future__ import annotations

"""Transit (sent -> the receiver's transport callback) and queue wait, by arm.

    python scripts/latency_followup/transit.py SESSION_DIR [ARM ...]
From the trace events: transit = received - encoded (what the network and the
two MQTT clients and the broker cost, without the receiver's runtime queue);
queue = popped - received; latency = fidelity's (delivered - sent).
Means per run, then mean and [min, max] over runs, per topology / link / kind.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def cls(n):
    if n == "cloud":
        return "cloud"
    if n.startswith("test-"):
        return "eval"
    if n.startswith("fog"):
        return "fog"
    return "edge"


def per_run(run):
    sent, ts, td, dl = {}, {}, {}, {}
    for line in open(run / "events.jsonl", encoding="utf-8"):
        e = json.loads(line)
        t = e["tags"] or {}
        mid = t.get("msg_id")
        n = e["name"]
        if n == "message.sent":
            sent[mid] = (e["node"], t["dst"], t["kind"], e["t_virtual"])
        elif n == "trace.sent":
            ts[mid] = t["wall"]
        elif n == "trace.delivered":
            td[mid] = t["wall"]
        elif n == "message.delivered":
            dl[mid] = e["t_virtual"]
    acc = defaultdict(lambda: defaultdict(list))
    for mid, (s, d, k, t0) in sent.items():
        if mid in ts and mid in td and mid in dl:
            key = (f"{cls(s)}>{cls(d)}", k)
            acc[key]["transit"].append(td[mid][0] - ts[mid][1])
            acc[key]["queue"].append(td[mid][1] - td[mid][0])
            acc[key]["latency"].append(dl[mid] - t0)
    return {k: {q: float(np.mean(v)) for q, v in d.items()} for k, d in acc.items()}


def main(session, arms):
    for arm in arms:
        table = defaultdict(lambda: defaultdict(list))
        for meta in sorted((Path(session) / arm).glob("*/run.json")):
            m = json.loads(meta.read_text(encoding="utf-8"))
            if m.get("status") != "finished":
                continue
            scen = m["scenario"]
            topo = (
                "cloud"
                if "cloud_npz" in scen
                else ("fog_a" if "fog-a" in scen else "fog_b")
            )
            for key, q in per_run(meta.parent).items():
                for name, v in q.items():
                    table[(topo, *key)][name].append(v * 1000)
        print(f"== {arm}")
        for key in sorted(table):
            if key[2] in ("hello",):
                continue
            row = table[key]
            cells = "  ".join(
                f"{q}={np.mean(row[q]):6.1f}[{min(row[q]):6.1f},{max(row[q]):6.1f}]"
                for q in ("transit", "queue", "latency")
            )
            print(
                f"  {key[0]:6} {key[1]:11} {key[2]:13} n={len(row['transit'])} {cells}"
            )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
