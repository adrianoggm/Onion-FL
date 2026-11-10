"""I9 spike: measure applied latency, bandwidth and loss per link, and summarise.

Runs on the host with ``docker`` on the PATH and drives the containers with
``docker exec``. The targets are in ``links.json`` next to this file.

    python measure.py shape --set rate           # Compose only: apply tc netem
    python measure.py run --option compose --set rate --reps 5 --out raw.jsonl
    python measure.py cpu --reps 3 --out raw.jsonl [--containernet]
    python measure.py summarize raw.jsonl        # Markdown tables on stdout

Two shaping sets, so that no number needs a correction:

- ``rate``: latency and bandwidth, no loss. Measures the RTT (ping) and the TCP
  goodput (iperf3), each alone and with saturating TCP traffic on every other
  link (``cross``), plus the two-hop path e1 -> fog1 -> cloud.
- ``loss``: the same plus loss. Measures one-way loss with UDP at half the link
  rate, alone and with cross traffic.

Containernet shapes its own links (``containernet_topo.py``); for it, ``run``
is called with ``--option containernet`` and no ``shape`` step.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINKS = json.loads((HERE / "links.json").read_text())["links"]
NAMES = {"compose": "onionfl-spike-{}-1", "containernet": "mn.{}"}
PATH_PORT = 5299


def sh(container: str, cmd: str, check: bool = True) -> str:
    out = subprocess.run(
        ["docker", "exec", container, "sh", "-c", cmd],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and out.returncode:
        raise RuntimeError(f"{container}: {cmd}\n{out.stderr}{out.stdout}")
    return out.stdout


def dev_of(container: str, ip: str) -> str:
    out = sh(container, f"ip -4 -o addr show to {ip}/32")
    return out.split()[1].split("@")[0]


def shape(name: str, set_: str) -> None:
    for link in LINKS.values():
        loss = f" loss {link['loss_pct']}%" if set_ == "loss" else ""
        for end in ("lower", "upper"):
            c = name.format(link[end])
            dev = dev_of(c, link[f"{end}_ip"])
            sh(
                c,
                f"tc qdisc replace dev {dev} root netem "
                f"delay {link['latency_ms']}ms rate {link['bw_mbit']}mbit{loss}",
            )


def start_servers(name: str) -> None:
    for i, link in enumerate(LINKS.values()):
        c = name.format(link["upper"])
        for port in (5200 + i, 5300 + i):
            serve(c, port)
    serve(name.format("cloud"), PATH_PORT)


def serve(c: str, port: int) -> None:
    # pgrep -f would match this very sh -c; ask the socket table instead.
    sh(c, f"ss -Hltn 'sport = :{port}' | grep -q . || iperf3 -s -p {port} -D")


def routes(name: str) -> None:
    """e1 reaches the cloud through fog1, and back."""
    sh(name.format("e1"), "ip route replace 10.194.5.0/24 via 10.194.1.3")
    sh(name.format("cloud"), "ip route replace 10.194.1.0/24 via 10.194.5.2")


def ping(c: str, ip: str) -> float:
    out = sh(c, f"ping -q -c 100 -i 0.02 {ip}")
    # rtt min/avg/max/mdev = 10.1/10.3/11.0/0.2 ms
    return float(out.strip().splitlines()[-1].split("=")[1].split("/")[1])


def tcp(c: str, ip: str, port: int) -> float:
    out = json.loads(sh(c, f"iperf3 -J -c {ip} -p {port} -t 5 -O 1"))
    return out["end"]["sum_received"]["bits_per_second"] / 1e6


def udp_loss(c: str, ip: str, port: int, mbit: float) -> float:
    # Now and then iperf3 exits non-zero with no output when its TCP control
    # connection crosses a lossy link; retry, and say so on stderr.
    cmd = f"iperf3 -J -u -c {ip} -p {port} -t 5 -l 1200 -b {mbit / 2}M"
    for attempt in range(3):
        try:
            return json.loads(sh(c, cmd))["end"]["sum"]["lost_percent"]
        except (RuntimeError, ValueError, KeyError) as exc:
            print(f"retry {attempt + 1}: {exc}", file=sys.stderr)
            time.sleep(3)
    raise RuntimeError(f"{c}: {cmd} failed three times")


def cross(name: str, skip: str | None, on: bool) -> None:
    for i, (lid, link) in enumerate(LINKS.items()):
        if lid == skip:
            continue
        c = name.format(link["lower"])
        if on:
            sh(
                c,
                f"iperf3 -c {link['upper_ip']} -p {5300 + i} -t 120 >/dev/null 2>&1 &",
            )
        else:
            sh(c, "pkill -f 'iperf3 -c .* -t 120'", check=False)
    # On: past slow start. Off: let the netem queues (1000 packets, 2.4 s at
    # 5 Mbit/s) drain, or the next "alone" measurement sees them.
    time.sleep(2 if on else 5)


def run(args: argparse.Namespace) -> None:
    name = NAMES[args.option]
    start_servers(name)
    routes(name)
    with open(args.out, "a") as out:

        def rec(link: str, crossed: bool, metric: str, rep: int, value: float):
            row = {
                "option": args.option,
                "set": args.set,
                "link": link,
                "cross": crossed,
                "metric": metric,
                "rep": rep,
                "value": value,
                "t": time.time(),
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(row, file=sys.stderr)

        for rep in range(args.reps):
            for i, (lid, link) in enumerate(LINKS.items()):
                lo, ip = name.format(link["lower"]), link["upper_ip"]
                for crossed in (False, True):
                    if crossed:
                        cross(name, lid, on=True)
                    try:
                        if args.set == "rate":
                            rec(lid, crossed, "rtt_ms", rep, ping(lo, ip))
                            rec(lid, crossed, "bw_mbit", rep, tcp(lo, ip, 5200 + i))
                        else:
                            loss = udp_loss(lo, ip, 5200 + i, link["bw_mbit"])
                            rec(lid, crossed, "loss_pct", rep, loss)
                    finally:
                        if crossed:
                            cross(name, lid, on=False)
            if args.set == "rate":
                e1 = name.format("e1")
                rec("e1_cloud", False, "rtt_ms", rep, ping(e1, "10.194.5.3"))
                rec("e1_cloud", False, "bw_mbit", rep, tcp(e1, "10.194.5.3", PATH_PORT))


def target(link: str, metric: str) -> float:
    if link == "e1_cloud":
        a, b = LINKS["e1_f1"], LINKS["f1_cloud"]
        return {
            "rtt_ms": 2 * (a["latency_ms"] + b["latency_ms"]),
            "bw_mbit": min(a["bw_mbit"], b["bw_mbit"]),
        }[metric]
    lk = LINKS[link]
    return {
        "rtt_ms": 2 * lk["latency_ms"],
        "bw_mbit": lk["bw_mbit"],
        "loss_pct": lk["loss_pct"],
    }[metric]


def margin(metric: str, tgt: float) -> float:
    """The acceptance error fixed before measuring (spec section 11)."""
    return {"rtt_ms": max(0.10 * tgt, 1.0), "bw_mbit": 0.10 * tgt}.get(metric, 1.0)


def passes(metric: str, tgt: float, mean: float) -> bool:
    return abs(mean - tgt) <= margin(metric, tgt)


def summarize(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in open(args.raw)]
    groups: dict[tuple, list[float]] = {}
    cpu_rows = [r for r in rows if r["metric"] == "samples_per_s"]
    rows = [r for r in rows if r["metric"] != "samples_per_s"]
    for r in rows:
        key = (r["option"], r["metric"], r["link"], r["cross"])
        groups.setdefault(key, []).append(r["value"])
    print("| Option | Metric | Link | Cross | Target | Mean | SD | n | Error | OK |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (opt, metric, link, crossed), vals in sorted(groups.items()):
        tgt, mean = target(link, metric), statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if metric == "loss_pct":
            err = f"{mean - tgt:+.2f} pt"
        else:
            err = f"{100 * (mean - tgt) / tgt:+.1f} %"
        ok = "yes" if passes(metric, tgt, mean) else "**no**"
        print(
            f"| {opt} | {metric} | {link} | {'yes' if crossed else 'no'} | "
            f"{tgt:g} | {mean:.2f} | {sd:.2f} | {len(vals)} | {err} | {ok} |"
        )
    print()
    print("Isolation (mean with cross traffic minus mean alone):")
    print()
    print("| Option | Metric | Link | Alone | Cross | Delta | Within margin |")
    print("|---|---|---|---|---|---|---|")
    for (opt, metric, link, crossed), vals in sorted(groups.items()):
        if crossed:
            continue
        other = groups.get((opt, metric, link, True))
        if not other:
            continue
        a, b = statistics.fmean(vals), statistics.fmean(other)
        tgt = target(link, metric)
        ok = abs(b - a) <= margin(metric, tgt)
        print(
            f"| {opt} | {metric} | {link} | {a:.2f} | {b:.2f} | {b - a:+.2f} | "
            f"{'yes' if ok else '**no**'} |"
        )
    if cpu_rows:
        summarize_cpu(cpu_rows)


CPU_LIMITS = (0.5, 1.0, 2.0)
CPU_WORKERS = 4
IMAGE = "onionfl-spike-netem:latest"


def bench(cmd: list[str], workers: int) -> subprocess.Popen:
    """Start cpubench.py with ``cmd`` (a docker run or exec prefix)."""
    proc = subprocess.Popen(
        [*cmd, "python3", "-", str(workers), "10"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    proc.stdin.write((HERE / "cpubench.py").read_text())
    proc.stdin.close()
    return proc


def rate_of(proc: subprocess.Popen) -> float:
    return json.loads(proc.stdout.read())["samples_per_s"]


def cpu(args: argparse.Namespace) -> None:
    """Training throughput under Docker --cpus and Containernet's cpu_quota."""
    docker = ["docker", "run", "--rm", "-i"]
    with open(args.out, "a") as out:

        def rec(option: str, host: str, concurrent: bool, rep: int, value: float):
            row = {
                "option": option,
                "set": "cpu",
                "link": host,
                "cross": concurrent,
                "metric": "samples_per_s",
                "rep": rep,
                "value": value,
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(row, file=sys.stderr)

        if args.control:
            # The same concurrent pair without limits, 1 and 2 workers: a drop
            # here comes from the host (turbo, hybrid cores), not from a quota.
            for rep in range(args.reps):
                for w in (1, 2):
                    alone = rate_of(bench([*docker, IMAGE], w))
                    rec("control", f"workers={w}", False, rep, alone)
                procs = {w: bench([*docker, IMAGE], w) for w in (1, 2)}
                for w, proc in procs.items():
                    rec("control", f"workers={w}", True, rep, rate_of(proc))
            return
        for rep in range(args.reps):
            one = rate_of(bench([*docker, IMAGE], 1))
            rec("docker", "1 worker, no limit", False, rep, one)
            for c in CPU_LIMITS:
                limited = bench([*docker, f"--cpus={c}", IMAGE], CPU_WORKERS)
                rec("compose", f"cpus={c:g}", False, rep, rate_of(limited))
                if args.containernet:
                    host = ["docker", "exec", "-i", f"mn.cpu{int(c * 100)}"]
                    value = rate_of(bench(host, CPU_WORKERS))
                    rec("containernet", f"cpus={c:g}", False, rep, value)
            # Isolation: the 1-CPU and 2-CPU hosts at the same time.
            procs = {
                c: bench([*docker, f"--cpus={c}", IMAGE], CPU_WORKERS)
                for c in (1.0, 2.0)
            }
            for c, proc in procs.items():
                rec("compose", f"cpus={c:g}", True, rep, rate_of(proc))


def summarize_cpu(rows: list[dict]) -> None:
    def values(option: str, host: str, concurrent: bool = False) -> list[float]:
        key = (option, host, concurrent)
        return [r["value"] for r in rows if (r["option"], r["link"], r["cross"]) == key]

    one = statistics.fmean(values("docker", "1 worker, no limit"))
    print()
    print(
        f"CPU: {CPU_WORKERS} workers; expected = min(cpus, workers) x "
        f"{one:.0f} samples/s (1 worker, no limit)."
    )
    print()
    print("| Option | Host | Concurrent | Expected | Mean | SD | n | Ratio |")
    print("|---|---|---|---|---|---|---|---|")
    for option, host, concurrent in sorted(
        {(r["option"], r["link"], r["cross"]) for r in rows}
    ):
        if option == "docker":
            continue
        vals = values(option, host, concurrent)
        exp = min(float(host.split("=")[1]), CPU_WORKERS) * one
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        mean = statistics.fmean(vals)
        print(
            f"| {option} | {host} | {'yes' if concurrent else 'no'} | {exp:.0f} | "
            f"{mean:.0f} | {sd:.0f} | {len(vals)} | {mean / exp:.2f} |"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shape")
    s.add_argument("--set", choices=("rate", "loss"), required=True)
    r = sub.add_parser("run")
    r.add_argument("--option", choices=tuple(NAMES), required=True)
    r.add_argument("--set", choices=("rate", "loss"), required=True)
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--out", required=True)
    c = sub.add_parser("cpu")
    c.add_argument("--reps", type=int, default=3)
    c.add_argument("--out", required=True)
    c.add_argument(
        "--containernet",
        action="store_true",
        help="also run in mn.cpu50/100/200 (containernet_topo.py cpu)",
    )
    c.add_argument(
        "--control",
        action="store_true",
        help="only the concurrent pair without limits (1 and 2 workers)",
    )
    m = sub.add_parser("summarize")
    m.add_argument("raw")
    args = p.parse_args()
    if args.cmd == "shape":
        shape(NAMES["compose"], args.set)
    elif args.cmd == "run":
        run(args)
    elif args.cmd == "cpu":
        cpu(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
