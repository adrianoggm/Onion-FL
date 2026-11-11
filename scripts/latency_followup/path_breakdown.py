from __future__ import annotations

"""Break down each message's latency from trace events and the broker's sniff.

    python scripts/latency_followup/path_breakdown.py RUN_DIR [RUN_DIR ...]          # prints per link class / kind
    python scripts/latency_followup/path_breakdown.py --json OUT RUN_DIR ...          # also dumps per-message rows

Stages (wall clock, ms), the latency fidelity measures being decoded - encoded:
  record   message.sent written (JSONL flush)          recorded - encoded
  publish  paho publish() call                         handed - recorded
  to_brk   handed -> last byte at the broker's netns    (paho thread, TCP, netem)
  broker   last byte in -> first byte out of the broker
  to_rcv   first byte out of the broker -> transport callback at the receiver
  queue    callback -> taken off the runtime's queue
  decode   taken off -> decoded (delivered event)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXTRA = [
    "wake",
    "snd_span",
    "path_up",
    "brk_in_span",
    "brk_out_span",
    "path_down",
    "rcv_span",
    "rcv_cb",
]
STAGES = ["record", "publish", "to_brk", "broker", "to_rcv", "queue", "decode"]


def cls(n):
    if n == "cloud":
        return "cloud"
    if n.startswith("test-"):
        return "eval"
    if n.startswith("fog"):
        return "fog"
    return "edge"


def sniff_index(path):
    """msg_id -> {(flow_dir): (first_ts, last_ts)} with dir 'in' (to 1883) / 'out'."""
    segs = defaultdict(list)  # flow -> [(ts, seq, len)]
    starts = []  # (msg_id, flow, ts, end)
    seen = set()
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not r["len"]:
            continue
        key = (r["src"], r["dst"], r["seq"], r["len"])
        if key in seen:  # loopback shows each packet twice
            continue
        seen.add(key)
        flow = (r["src"], r["dst"])
        segs[flow].append((r["ts"], r["seq"], r["len"]))
        for msg_id, end, _ in r.get("pubs", []):
            starts.append((msg_id, flow, r["ts"], end))
    out = defaultdict(dict)
    for msg_id, flow, ts, end in starts:
        last = ts
        if end is not None:
            cands = [
                t
                for t, s, n in segs[flow]
                if (end - 1 - s) % 2**32 < n and t >= ts - 1e-3
            ]
            if cands:
                last = min(cands)
        d = "in" if flow[1].endswith(":1883") else "out"
        prev = out[msg_id].get(d)
        if prev is None or ts < prev[0]:
            out[msg_id][d] = (ts, last)
    return out


def rows_of(run):
    ev = [json.loads(x) for x in open(run / "events.jsonl", encoding="utf-8")]
    sent, tsent, tdel, deliv = {}, {}, {}, {}
    for e in ev:
        t = e["tags"] or {}
        mid = t.get("msg_id")
        if e["name"] == "message.sent":
            sent[mid] = (e["node"], t["dst"], t["kind"], e["value"], e["t_virtual"])
        elif e["name"] == "trace.sent":
            tsent[mid] = t["wall"]
        elif e["name"] == "trace.delivered":
            tdel[mid] = t["wall"]
        elif e["name"] == "message.delivered":
            deliv[mid] = e["t_virtual"]
    brk = sniff_index(run / "emulation" / "sniff-cloud-1.jsonl")
    groups = json.load(open(run / "scenario.json"))["groups"]
    host = {n: h for h, ns in groups.items() for n in ns}
    local = {}  # host -> its sniff index
    for f in (run / "emulation").glob("sniff-*.jsonl"):
        h = f.stem[len("sniff-") :]
        if h != "cloud-1":
            local[h] = sniff_index(f)
    rows = []
    for mid, (src, dst, kind, size, t_s) in sent.items():
        if mid not in tsent or mid not in tdel or mid not in deliv:
            continue
        s0, s1, s2, s3 = tsent[mid]
        rcv, pop, dec, rec, done = tdel[mid]
        b = brk.get(mid, {})
        row = dict(  # noqa: C408
            src=src,
            dst=dst,
            kind=kind,
            size=size,
            link=f"{cls(src)}>{cls(dst)}",
            latency=deliv[mid] - t_s,
            record=s2 - s1,
            publish=s3 - s2,
            queue=pop - rcv,
            decode=dec - pop,
            handle=done - rec,
            sent_wall=s1,
        )
        if "in" in b and "out" in b:
            row["to_brk"] = b["in"][1] - s3
            row["broker"] = b["out"][0] - b["in"][1]
            row["to_rcv"] = rcv - b["out"][0]
            row["brk_in_span"] = b["in"][1] - b["in"][0]
            row["brk_out_span"] = b["out"][1] - b["out"][0]
        hs, hd = host.get(src), host.get(dst)
        if hs in local and "in" in local[hs].get(mid, {}):
            first, last = local[hs][mid]["in"]
            row["wake"] = first - s3  # handed -> first byte leaves the sender host
            row["snd_span"] = last - first  # first -> last byte leave it
            if "in" in b:
                row["path_up"] = b["in"][1] - last  # last byte: sender host -> broker
        if hd in local and "out" in local[hd].get(mid, {}):
            first, last = local[hd][mid]["out"]
            if "out" in b:
                row["path_down"] = last - b["out"][1]
            row["rcv_span"] = last - first
            row["rcv_cb"] = rcv - last  # last byte in -> transport callback
        rows.append(row)
    return rows


def summary(rows, label=""):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["link"], r["kind"])].append(r)
    cols = ["latency"] + STAGES + EXTRA
    print(
        f"{label}{'link':12} {'kind':13} {'n':>5} "
        + " ".join(f"{c[:8]:>8}" for c in cols)
    )
    for k in sorted(groups):
        g = groups[k]
        vals = []
        for c in cols:
            v = [r[c] for r in g if c in r]
            vals.append(f"{np.mean(v) * 1000:8.1f}" if v else f"{'-':>8}")
        print(f"{label}{k[0]:12} {k[1]:13} {len(g):5d} " + " ".join(vals))


def main(argv):
    dump = None
    if argv[0] == "--json":
        dump, argv = argv[1], argv[2:]
    allrows = []
    for d in argv:
        for run in (
            sorted(Path(d).glob("*/run.json"))
            if not (Path(d) / "run.json").exists()
            else [Path(d) / "run.json"]
        ):
            allrows += [r | {"run": run.parent.name} for r in rows_of(run.parent)]
    summary(allrows)
    if dump:
        json.dump(allrows, open(dump, "w"))


if __name__ == "__main__":
    main(sys.argv[1:])
