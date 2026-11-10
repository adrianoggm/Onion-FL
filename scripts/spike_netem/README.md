# I9 spike: Docker Compose + netem vs Containernet

These are the scripts behind `docs/decisions/2026-11-06-i9-emulator.md`.

- They build the same lab (1 cloud, 2 fogs, 4 edges) twice, once per option, and measure each physical link against `links.json`.
- Everything runs in throwaway containers. Nothing here is part of the `onion_fl` package.

| File | What it does |
|---|---|
| `Dockerfile` | Measurement host: `iproute2`, `iputils-ping`, `iperf3`, `procps`, `python3-numpy` and `net-tools`, from Debian's apt |
| `links.json` | Targets per link: one-way latency, bandwidth, and loss per direction |
| `compose.yaml` | Option A: one container per host and one Docker network per link |
| `containernet_topo.py` | Option B: the same lab as direct `TCLink`s, or three CPU-limited hosts with `cpu` |
| `containernet_run.sh` | Entrypoint for the Containernet image: starts Open vSwitch, then `containernet_topo.py` |
| `containernet_probe.sh` | Checks that Open vSwitch starts in the Containernet image on this Docker host |
| `measure.py` | Applies tc (Compose only), measures RTT, TCP goodput and UDP loss, and CPU throughput, then prints the tables |
| `cpubench.py` | MLP training throughput (samples/s) under a CPU limit |

## Running it

Run the commands from this folder, with Docker running.

- In Git Bash on Windows, prefix the `docker` lines that contain absolute container paths with `MSYS_NO_PATHCONV=1`.
- `python` is any Python ≥ 3.11 (the repo `.venv` works); it only calls `docker`.

```bash
docker build -t onionfl-spike-netem:latest .

# Option A: Docker Compose + netem
docker compose up -d
python measure.py shape --set rate
python measure.py run --option compose --set rate --reps 5 --out raw.jsonl
python measure.py shape --set loss
python measure.py run --option compose --set loss --reps 5 --out raw.jsonl
docker compose down

# Option B: Containernet, one lab per set: rate, loss, then cpu
docker run -d --name onionfl-spike-cn --privileged --pid=host --network host \
  -v /var/run/docker.sock:/var/run/docker.sock -v /lib/modules:/lib/modules:ro \
  -v "$PWD":/spike --entrypoint /spike/containernet_run.sh \
  containernet/containernet:latest rate
docker logs -f onionfl-spike-cn          # wait for READY
python measure.py run --option containernet --set rate --reps 5 --out raw.jsonl
docker exec onionfl-spike-cn touch /tmp/stop     # Mininet stops the mn.* hosts
docker wait onionfl-spike-cn && docker rm onionfl-spike-cn
# ...the same with "loss", and with "cpu" for:
python measure.py cpu --reps 3 --containernet --out raw.jsonl
python measure.py cpu --reps 3 --control --out raw.jsonl

python measure.py summarize raw.jsonl
```

## Pitfalls

These came up while building the spike:

- **Matching processes.** `pgrep -f` and `pkill -f` inside `docker exec sh -c '…'` match the `sh -c` itself.
- **netem queues.** netem's default queue is 1000 packets (2.4 s at 5 Mbit/s).
  After a bulk transfer stops, wait for the queue to drain before measuring RTT on that link.
- **iperf3 over lossy links.** iperf3 UDP sometimes exits with no result when its TCP control connection crosses a lossy link. `measure.py` retries.
- **Open vSwitch on Docker Desktop.** The Containernet image cannot `modprobe openvswitch` unless the VM's `/lib/modules` is mounted; without it, `ovs-vsctl add-br` hangs.
- **`ifconfig` in host images.** Mininet brings interfaces up with `ifconfig`, so host images need `net-tools`.
- **Import order.** Mininet needs `mininet.net` imported before `mininet.link`, or the import fails.
