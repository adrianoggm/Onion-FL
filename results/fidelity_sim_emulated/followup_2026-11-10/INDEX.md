# Where the emulated link latency goes (follow-up of issue #198, 2026-11-10)

**Status: run on 2026-11-10.** This is a follow-up, not part of the pre-registration: the pre-registered results in [`../INDEX.md`](../INDEX.md) are unchanged. It asks where the ~64 ms per message by which the emulated links exceed the simulated ones came from (H2c −0.483, H2b −0.395). Its method is to instrument the path of each message, test one cause at a time in paired runs, fix what is real, and rerun a reduced comparison.

## Answer

The 64 ms are not one cost of the transport. They have three sources, plus a fixed cost of training that also moves the round time:

1. **The receiving process's queue (head-of-line), about two thirds.** A host runs all its logical nodes in one process with one event loop, here four edges per device. A model for the fourth edge reaches the process with the others, then waits in the queue while the other three train. `message.delivered` is recorded when the loop takes the message, so the wait counts as link latency. The simulator delivers to each node independently: it queues the training on the host's CPU, not the message. The time only changes column (link latency in the emulation, compute time in the simulation). It is a difference in what is measured, not a network delay.
2. **Nagle's algorithm, about a quarter: a real delay, now fixed.** Neither paho nor mosquitto sets `TCP_NODELAY`. When two messages leave the same socket closer together than a round trip, the tail of the second waits for the acknowledgement of the first. This happens to an edge host's updates, a fog's models, and the burst of `stop`s. The microbenchmarks of #197 waited for each message, so they could not see it.
3. **Same-host links.** The simulator gives 0 latency to the links between the coordinator and its evaluators, which share a host. In the emulation they cross the broker and the coordinator's queue (5 to 35 ms). Each such link scores a relative error of −1, and they are 10 of the 50–52 links of a run. They alone explain −0.19 of H2c's −0.483 (below).

With Nagle off, measuring to the transport's hand-over (without the queue) and leaving out same-host links, the simulator's link latency is within **−3.9 % [−8.4 %, +0.4 %]** of the emulation's (4 traced pairs). The calibrated network model holds. What separated the two backends was the measurement and Nagle.

## The path of one message, measured

### How

- **Tracing** (`ONIONFL_TRACE=1`, commit `e1f48a8`, off by default). Each node process records, per message sent, the wall-clock instants when it was stamped, encoded, recorded (the JSONL event written) and handed to paho. Per message handled, it records when paho handed it over, when the loop took it, when it was decoded, recorded and handled. All containers share the Docker VM's clock.
- **Sniffing** (`scripts/latency_followup/sniff_mqtt.py`). An `AF_PACKET` socket with kernel timestamps runs in a container in the network namespace of `cloud-1` (the broker's), of the fog host and of `field-edge_device-1`. It logs every segment to or from port 1883, finds each PUBLISH by its `msg_id`, and finds the segment that carries its last byte.
- **Session s1, 2026-11-10, 09:46–10:06 UTC.** `cloud-1ep` and `fog_a-1ep` × seeds 0, 1 × 4 arms, seed by seed with the arms alternating, 16 runs, every one traced and sniffed. Code: `002e618` plus the tracing, which was committed afterwards as `e1f48a8`, plus a temporary switch of the client and broker `TCP_NODELAY` by environment, not committed. The arms `base` and `nodelay` are what `backend.nodelay: false` and the default now give.

### Breakdown (s1, `base` arm = the pre-registered behaviour; ms, mean over the messages of 2 runs per topology)

| Link, kind | Latency (fidelity's) | Sender: record + publish | Sender host: handed → last byte leaves | Network to the broker | Broker | Broker → receiver's paho | Receiver's queue | Decode |
|---|---|---|---|---|---|---|---|---|
| edge → cloud, update | 78.5 | 3.8 | 11.4 (Nagle) | 55.3 | 0.3 | 0.2 | 6.2 | 1.0 |
| edge → fog, update | 144.8 | 3.1 | 18.2 (Nagle) | 55.2 | 0.8 | 62.0 (of which 17.0 Nagle at the broker) | 2.9 | 1.3 |
| cloud → edge, control (`stop`) | 96.8 | 1.0 | – | 0.3 | **32.6** (Nagle at the broker) | 61.3 | 1.1 | 0.4 |
| cloud → edge, global model | 695.2 | 2.6 | – | 0.5 | 7.3 | 85.7 | **597.6** | 1.6 |
| fog → edge, global model | 838.9 | 3.1 | 31.8 (a 110 kB burst at 10 Mbit/s) | 51.9 | 1.7 | 78.8 | **660.4** | 1.5 |
| fog → cloud, update | 46.5 | 0.9 | 0.1 | 44.2 | 0.2 | 0.4 | 0.1 | 1.0 |

- **Nagle.** For an update, the first and last byte left the edge host 11–37 ms apart, although its egress is not shaped. That is a 5.5 kB message, whose serialisation at 10 Mbit/s takes 4.5 ms. The broker likewise held a `stop` 24–33 ms between its last byte in and its first byte out. With `TCP_NODELAY` both gaps fell to 0–0.5 ms (table below).
- **The queue.** A model for an edge waited 0.35–1 s in its process's queue on this (busy) machine. The queue makes up 79–86 % of the downlinks' latency. For the coordinator's evaluators it was 10–150 ms. The rank of a delivery within its host confirms it on the pre-registered runs: the first edge of a host got its model after 65 ms (the path plus 15 ms), and each later one about 1.3 ms after its neighbour sent its update, that is, as soon as the loop was free.
- **Python, paho and timers are not where the time goes.**
  - Encoding (npz) takes 0.4 ms (median).
  - Writing the `message.sent` event takes 0.6–0.9 ms (median) and 1.6–2.5 ms (mean). One flush to the Windows bind mount stalled once for 0.27 s.
  - paho takes 0.1–0.2 ms to start writing after `publish()` returns, and 0.3–1.8 ms from the last byte in to the callback.
  - Decoding takes 1 ms.
  - No delay is a multiple of 15.6 ms. A QoS 1 acknowledgement holds nothing: the broker forwards in 0.2–0.8 ms.
- **The netem hops are as calibrated.** From the sender host to the broker: 55 ms on a 50 ms path, 44 ms on a 40 ms path, the rest being serialisation.

### One cause at a time (s1, transit = sent → the receiver's paho callback, without the queue; ms, mean [min, max] over 2 runs)

| Link, kind | Nagle on both (`base`) | `TCP_NODELAY` clients and broker | clients only | broker only | Simulation |
|---|---|---|---|---|---|
| edge → cloud, update | 71.2 [63.0, 79.5] | **60.4** [60.2, 60.5] | 59.8 | 73.7 | 56.6 |
| edge → fog, update | 140.6 [136.6, 144.5] | **105.2** [103.6, 106.8] | 120.1 | 128.5 | 101.3 |
| cloud → edge, control | 95.2 [91.4, 99.0] | **52.8** [52.2, 53.3] | 76.7 | 53.5 | 52.7 |
| fog → edge, control | 144.8 [143.3, 146.3] | **94.4** [93.8, 95.0] | 128.1 | 110.9 | 93.6 |
| cloud → edge, global model | 96.1 [93.8, 98.3] | 91.6 [86.7, 96.6] | 100.9 | 82.3 | 92.1 (latency) |
| fog → edge, global model | 177.0 [168.7, 185.2] | 161.1 [151.1, 171.0] | 179.0 | 159.7 | 137.1 (latency) |
| fog ↔ cloud | 42–55 | 42–47 | 43–56 | 46–48 | 41–45 |

- The clients' Nagle delays the updates, and the broker's delays the bursts of control messages. A link through the broker needs both off.
- **The other candidates, ruled out by these numbers.** paho's loop and network thread, the timer resolution, QoS 1, the encoding and the netem hops each take 0.1–2 ms (above).

Transit is little affected by the machine. Everything that depends on the CPU (training, and so the queue) was 2.5–4 times slower than on 2026-11-09, because another agent's containers (`onionfl-a227-*`) were running. That is why the A/B compares paired runs.

### What it means for H2c

The pre-registered runs, split by link class with the harness's own functions (`h2_by_link.py`):

| Links | Emulated (ms) | Sim (ms) | Error |
|---|---|---|---|
| all (H2c) | 149.4 | 84.6 | −0.483 [−0.494, −0.474] |
| without the same-host links | 182.8 | 105.0 | −0.358 [−0.371, −0.347] |
| same-host only | 11.0 | 0.0 | −1.000 |
| downlinks to edges | 241.9 | 121.6 | −0.439 |
| uplinks from edges | 133.0 | 92.8 | −0.291 |
| fog ↔ cloud | 63.3 | 54.3 | −0.140 |

On the traced s1 runs, against the simulation of the same designs and seeds (`h2_transit.py`):

| Emulated latency measured to | Links | Nagle on | Nagle off |
|---|---|---|---|
| delivery (as pre-registered) | all | −0.569 | −0.476 |
| delivery | without same-host | −0.464 | −0.348 |
| the transport's hand-over | all | −0.351 | −0.228 |
| the transport's hand-over | without same-host | −0.193 [−0.250, −0.121] | **−0.039 [−0.084, +0.004]** |

Per link and on average, the 4 traced Nagle-on pairs spend 316.6 ms to delivery, against 95.5 ms of transit and 73.8 ms in the simulation. Of the transit's +21.7 ms over the simulation, 16.1 ms are Nagle (95.5 → 79.4). The rest is the same-host links' 1–7 ms and the fog's burst of models (+24 ms, below). Applied to the pre-registered 149.4 − 84.6 = 64.8 ms, this is an estimate, since those runs have no trace: about 16 ms Nagle, 5 ms other transit, and about 44 ms queue and decoding.

## The fix

Commit `5ac0133`:

- **MQTT transport.** `mqtt` sets `TCP_NODELAY` on both connections, on every socket paho opens (`on_socket_open`, so after a reconnection too). A link's `nodelay: false` keeps Nagle. The default is left out of the config dump.
- **Emulated backend.** `backend.nodelay` (default `true`) runs the broker with `set_tcp_nodelay true` and passes `nodelay` to the node processes in `scenario.json`; `RealRuntime(nodelay=…)` applies it to every MQTT link, as it does the broker's address. `nodelay: false` reproduces the runs before the fix.
- **Calibration.** The microbenchmarks pass their spec's `links.nodelay` to the transport, so a calibration session measures what its spec says, as before (default Nagle on, as the sessions of `b55104b3`).
- **Ids.** The experiment's `config_id`s are unchanged: the six of `experiments/fidelity_sim_emulated.yaml` still come out `073cc75f…`, `a638a5db…` and the others, with the runs folder of the validation.
- **Tests.** `tests/test_runtime_real.py`:
  - both connections get `TCP_NODELAY`, a link with `nodelay: false` keeps Nagle;
  - the launcher's override;
  - tracing off and on.

  `tests/test_transport_mqtt.py` checks the real sockets against a broker. `tests/test_emulated.py` checks the broker's config, `scenario.json`, the default left out of the config, and `ONIONFL_TRACE` through `docker exec`.

## Reduced comparison with the fix

- **Runs.** `cloud-1ep` and `fog_a-1ep` (the pre-registered designs), seeds 0–4, emulated with `backend.nodelay: false` (Nagle, as pre-registered) and with the default (no Nagle). They ran seed by seed with the arms alternating, 2026-11-10 10:13–10:32 UTC. The sim runs of the same designs followed at 10:32. Code: `5ac0133` for every run.
  - The seed-4 runs and the sim runs are marked `dirty` because the untracked `scripts/latency_followup/` files were being written during the session; `src/` was that of `5ac0133`.
- **Analysis.** `onion_fl fidelity … --calibration b55104b3…`, the output is in `fidelity_nagle.json` and `fidelity_nodelay.json`. All 30 runs finished; no run is `estimated` or `outside_calibration`, and the harness raised only its usual warning that macro-F1 and loss tie across these two designs.
- **The machine was busy.** Another agent's containers were running, and the edges trained 2.5 times slower than on 2026-11-09 (compute 0.17 s per edge and round against 0.069 s). Absolute errors are therefore larger than the pre-registered ones, and only the paired difference between the arms speaks to the fix.

| Quantity, (B − A) / A | Nagle on | Nagle off | Pre-registered (Nagle on, 6 designs, quiet machine) |
|---|---|---|---|
| Round time (H2b) | −0.588 [−0.656, −0.519] | −0.525 [−0.623, −0.437] | −0.395 [−0.421, −0.368] |
| Link latency (H2c) | −0.562 [−0.588, −0.530] | **−0.447** [−0.478, −0.421] | −0.483 [−0.494, −0.474] |
| uplinks from edges | −0.316 [−0.342, −0.291] | **−0.076** [−0.105, −0.056] | −0.291 |
| fog ↔ cloud | −0.141 | −0.067 | −0.140 |
| downlinks to edges | −0.608 | −0.561 | −0.439 |
| same-host | −1 | −1 | −1 |
| Compute time per edge (H2a) | −0.836 | −0.836 | −0.095 |

Paired differences of the emulated runs, Nagle off minus on (ms, mean [min, max] over 5 seeds):

| | cloud | fog_a |
|---|---|---|
| uplink latency from edges | −27.9 [−42.2, −12.0] | −35.6 [−41.2, −25.7] |
| fog ↔ cloud | – | −4.2 [−7.0, −1.9] |
| round time | −144 [−708, +427] | −86 [−412, +337] |
| the last update of a round, sent → at the cloud (cloud-1ep) | 109.8 → **59.3** (sim 61.4) | |

- **Uplinks.** The fix removes the uplinks' excess: from +38 ms to +6.5 ms over the simulation.
- **Round time.** The round's critical uplink falls by 50 ms, to the simulation's. Round time improves on average, but the spread of training on the busy machine swamps it within a pair. On the quiet machine of the pre-registered runs, the same 50–70 ms on a 0.72 s round would move H2b by about +0.1, from −0.395 to about −0.3.
- **H2c.** It improves by 0.12 but stays near −0.45. What is left is the queue and the same-host links, neither of which is a network delay.

## What is still unexplained or open

- **The queue is not fixed, because it is not a delay of the emulation.** Where the time is attributed differs between the backends. Two ways to compare like with like, neither done here:
  - The emulated latency could be measured to the transport's hand-over: record in `message.delivered` how long the message waited in the queue, and give the harness a `link_transit` next to `link_latency`. H2c as pre-registered would stay as it is.
  - Or the simulator could block a host's nodes together, as one process does.
- **Same-host links.** H2c counts at −1 the links that the simulation makes free. A secondary criterion should leave them out, or give the simulator the 0.5 ms the calibration measured for `cloud-1` to itself.
- **The fog's burst of models.** 20 models of 5.5 kB (110 kB) leave the fog at once over a 10 Mbit/s, 80 ms link. Their transit is still 24 ms above the simulation without Nagle (161 against 137 ms). The first byte of a model leaves the fog 25–39 ms after `publish()`, behind the previous ones. The likely cause is TCP's congestion window: the slow start after an idle connection that the calibration already found past 14.6 kB. It was not isolated.
- **Round time (H2b).** Most of what remains is the per-training cost: the edge handler takes 40–100 ms (p10–p50) for one epoch, against the simulator's 7 ms. This is the fixed cost of H2a, not studied here.
- **Size of the sample.** The cause-by-cause table has 2 runs per arm and topology, and the reduced comparison 5 seeds × 2 designs. Both ran on a busy machine.

## Files

- `fidelity_nagle.json`, `fidelity_nodelay.json`: the harness's output on the reduced comparison (A = emulated arm, B = sim).
- `runs_nagle/`, `runs_nodelay/`, `runs_sim/`: `run.json` and `summary.json` of each reduced run.
  - `host`, `pid` and `config.paths` are removed, as in `../`, so `verify_run` cannot check these copies.
  - Their `config_id`s differ from the pre-registered ones only in `paths.runs`, and the Nagle arm also in `backend.nodelay`; `design_id` pairs them.
- The traced session s1 (16 runs, 2026-11-10T09:46:56Z–10:05:22Z) is summarised in the tables above. Its events and sniffs (≈ 70 MB) are not committed.
- Scripts in `scripts/latency_followup/`:
  - `traced_session.py`: traced and sniffed paired runs;
  - `sniff_mqtt.py`;
  - `path_breakdown.py`: the stages above;
  - `transit.py`: transit and queue per arm;
  - `h2_by_link.py`, `h2_transit.py`;
  - `run_reduced.py`.

```bash
python scripts/latency_followup/traced_session.py --out runs/a198/s1 --arms nagle nodelay \
  --seeds 0 1 --scenario "assignment=swell_cloud_npz" "'fog_1': 'fog-a'" --sniff
python scripts/latency_followup/path_breakdown.py runs/a198/s1/nagle
python scripts/latency_followup/run_reduced.py runs/a198/reduced && \
  python scripts/latency_followup/run_reduced.py runs/a198/reduced --sim
onion_fl fidelity runs/a198/reduced/nodelay runs/a198/reduced/sim --calibration b55104b3… \
  --metric time --metric bytes --metric macro_f1 --metric loss --series global/swell
```
