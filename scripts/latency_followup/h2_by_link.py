from __future__ import annotations

"""H2b/H2c as the harness computes them, plus link latency split by link class.

    python scripts/latency_followup/h2_by_link.py EMULATED_RUNS SIM_RUNS
"""

import sys

from onion_fl.experiment.fidelity import _events, load_cells, performance, timings


def cls(n):
    if n == "cloud":
        return "cloud"
    if n.startswith("test-"):
        return "eval"
    if n.startswith("fog"):
        return "fog"
    return "edge"


def main(a, b):
    ca, _ = load_cells(a)
    cb, _ = load_cells(b)
    keys = sorted(ca.keys() & cb.keys())
    pairs = [(timings(_events(ca[k])), timings(_events(cb[k]))) for k in keys]
    print(f"{len(keys)} pairs")
    perf = performance(pairs)
    for q in ("round_time", "link_latency", "compute_time"):
        p = perf[q]
        ci = p["relative_error_ci"]
        print(
            f"{q:14} A={p['a_mean']:.4f} B={p['b_mean']:.4f} e={p['relative_error']:+.3f}"
            f" CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]"
            if ci
            else f"{q} e={p['relative_error']:+.3f}"
        )
    for name, keep in (
        ("links without same-host", lambda k: "eval" not in (cls(k[0]), cls(k[1]))),
        ("same-host links only", lambda k: "eval" in (cls(k[0]), cls(k[1]))),
        ("downlinks to edges", lambda k: cls(k[1]) == "edge"),
        ("uplinks from edges", lambda k: cls(k[0]) == "edge"),
        ("fog<->cloud", lambda k: {cls(k[0]), cls(k[1])} == {"fog", "cloud"}),
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
        p = performance(sub).get("link_latency")
        if p:
            ci = p["relative_error_ci"]
            print(
                f"  {name:24} A={p['a_mean'] * 1000:6.1f} ms B={p['b_mean'] * 1000:6.1f} ms "
                f"e={p['relative_error']:+.3f} CI=[{ci[0]:+.3f}, {ci[1]:+.3f}] "
                f"diff={(p['a_mean'] - p['b_mean']) * 1000:+.1f} ms"
            )


if __name__ == "__main__":
    main(*sys.argv[1:3])
