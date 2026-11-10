"""How SimRuntime scales with the number of edges: wall time and peak memory.

Each edge holds its own ``modular_mlp`` of FEMNIST's shape (784 pixels, 62
classes) and runs the stub trainer, which learns nothing and needs no data: the
cost measured is the protocol's (messages, aggregation, one model per node), the
part that grows with the number of edges. Training cost comes on top and is
proportional to the samples, whatever their split into edges.

Every size runs in a fresh process, so its peak memory is its own::

    python scripts/sim_scale.py --edges 100 200 500 1000 2000 > scale.jsonl
    python scripts/sim_scale.py --edges 100 200 --codec json
    python scripts/sim_scale.py --edges 500 1000 2000 \
        --diagnostics dataset_conflict drift participation fairness   # no divergence
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time


def peak_rss_bytes() -> int:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        info = kernel32.K32GetProcessMemoryInfo
        info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "K32GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # KiB on Linux


def one(
    n_edges: int, n_fogs: int, rounds: int, codec: str, diagnostics: list | None
) -> dict:
    from onion_fl.core.topology import parse_topology
    from onion_fl.learning.model import (
        DataShape,
        ModularMLP,
        ModularMLPConfig,
        state_arrays,
    )
    from onion_fl.learning.trainers import trainers
    from onion_fl.roles import EdgeSpec, build_federation

    shape = DataShape(dataset="femnist", task="character", n_features=784, n_classes=62)
    config = ModularMLPConfig(adapter_width=128, trunk_hidden=[64], dropout=0.0)
    fogs = [f"fog_{k}" for k in range(n_fogs)]
    settings = {} if diagnostics is None else {"diagnostics": diagnostics}
    topology = parse_topology(
        {
            "name": "scale",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"} | settings,
            "fog": {
                "defaults": {"link_up": {"codec": codec}} | settings,
                "nodes": [{"id": f} for f in fogs],
            },
            "edge": {"link_up": {"codec": codec}},
        }
    )
    base = peak_rss_bytes()
    start = time.perf_counter()
    edges = {f: [] for f in fogs}
    for i in range(n_edges):
        edges[fogs[i % n_fogs]].append(
            EdgeSpec(
                f"e{i}",
                ModularMLP(config, [shape], seed=0),
                trainer=trainers.create("stub", {"noise": 0.01}),
            )
        )
    federation = build_federation(
        topology,
        edges,
        initial_state=state_arrays(ModularMLP(config, [shape], seed=0)),
        rounds=rounds,
    )
    built = time.perf_counter()
    federation.run()
    done = time.perf_counter()
    if not federation.coordinator.finished:
        raise SystemExit(f"{n_edges} edges: the coordinator did not finish")
    return {
        "edges": n_edges,
        "fogs": n_fogs,
        "rounds": rounds,
        "codec": codec,
        "diagnostics": "all" if diagnostics is None else diagnostics,
        "build_s": round(built - start, 2),
        "run_s": round(done - built, 2),
        "s_per_round": round((done - built) / rounds, 2),
        "peak_rss_mib": round(peak_rss_bytes() / 2**20, 1),
        "import_rss_mib": round(base / 2**20, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--edges", type=int, nargs="+", default=[100, 200, 500])
    parser.add_argument("--fogs", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--codec", default="npz", help="Codec of every link")
    parser.add_argument(
        "--diagnostics", nargs="*", help="Diagnostics of every aggregator (all)"
    )
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        print(
            json.dumps(
                one(args.edges[0], args.fogs, args.rounds, args.codec, args.diagnostics)
            )
        )
        return
    print(
        json.dumps(
            {"python": platform.python_version(), "machine": platform.platform()}
        )
    )
    for n in args.edges:
        command = [sys.executable, __file__, "--one", "--edges", str(n)]
        command += ["--fogs", str(args.fogs), "--rounds", str(args.rounds)]
        command += ["--codec", args.codec]
        if args.diagnostics is not None:
            command += ["--diagnostics", *args.diagnostics]
        out = subprocess.run(command, capture_output=True, text=True, check=True)
        print(out.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
