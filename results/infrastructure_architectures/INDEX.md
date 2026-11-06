# The same federation over four architectures (new framework, real data)

`experiments/infrastructure_architectures.yaml` was run on 2026-11-06 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows), on the `sim` backend; its logical half also ran on `local`, for the decision fidelity between the two (`fidelity/`). Issue #192, spec I1 §1 (question 2) and §12.

```bash
onion_fl run experiments/infrastructure_architectures_select.yaml --workers 3   # edge learning rate, validation only
onion_fl run experiments/infrastructure_architectures.yaml --workers 5
python scripts/architectures_summary.py runs --out run_metrics.csv
python scripts/run_architectures_fidelity.py --backend sim --broker localhost:18836 --workers 3   # logical_sim/
python scripts/architectures_summary.py runs/architectures_sim --experiment infrastructure_architectures_logical --out logical_metrics.csv
```

## Setup

- **What stays the same.** SWELL + WESAD, the same subject roles (`roles.seed` 0: 28 training edges, 18 SWELL and 10 WESAD; the same 5 SWELL and 3 WESAD test subjects), a modular MLP, SGD at lr 0.1 with 10 local epochs, FedAvg everywhere, 20 rounds, seeds 0–4.
- **What changes is where the nodes run** (`infrastructures/architectures_lab.yaml`: one cloud VM, two fog servers, a field pool of devices; field–fog links of 100 and 50 Mbps, fog–cloud links of 10 Mbps, shared bandwidth). No MQTT broker is placed, so each logical link is the direct physical path between its hosts.

| Arm | Topology | Edges' devices | Dynamics |
|---|---|---|---|
| A, cloud-centralised | `cloud_swell_wesad`: two levels, the 28 edges straight under the cloud | 28 Raspberry Pi (900 samples/s) | none |
| B, edge–fog–cloud | `four_fogs_swell_wesad_physical`: four fogs, two SWELL and two WESAD (α = 0), on `fog-1` and `fog-2` | 28 Raspberry Pi | none |
| C, heterogeneous | as B | 50 % Raspberry Pi, 30 % desktop (6,000/s), 20 % low-power IoT (150/s), `profile_mix` | none |
| D, dynamic | as B | as C | `fog-1` (both SWELL fogs) down from 30 s to 60 s (`architectures_lab_dynamic`), `failover: nearest_alive` after 5 s |

- **The infrastructure numbers are illustrative, not calibrated** (I12, #197). The times below say what this model of the continuum predicts, not what real devices do.
- **The edges' learning rate was chosen on validation** (`validation.csv`): `experiments/infrastructure_architectures_select.yaml` ran B at 0.03, 0.1 and 0.3 on seed 0, scored by the final global model on the validation subjects (`evaluation.global.subjects: val`). 0.1 won (0.634 against 0.629 and 0.558), inside its grid. It is fixed for the four arms, so that only the infrastructure changes. The test subjects played no part. D's failure window (30–60 s) was set from B's selection runs, which took 38 simulated seconds, so that it falls inside every run.
- **Metrics** (`run_metrics.csv`, one row per run; `scripts/architectures_summary.py`):
  - learning: the final global macro-F1 on the test subjects per dataset; each edge's macro-F1 on its own held-out rows at round 20, for the model it received and after its local training;
  - round time: from `round.started` to `round.closed` at the cloud, simulated seconds (rounds that fail their quorum have no close, and are counted apart);
  - time to target: the first round whose pooled global macro-F1 reaches 0.6, a target fixed before the runs; when it never does, the run's end (censored);
  - bytes per level: every `message.sent`, by the two ends of its link. `wan` is the traffic to and from the cloud, which is alone in its site, so every such message crosses a 10 Mbps fog–cloud link. The evaluators' traffic is apart: they sit on their parent's host.

## Identity

| Identity | Value |
|---|---|
| `data_id` | `9fd573754157d68b111f037ecd891bdb62abcbff1d5e5e818da789dde470257c` |
| code | commit `2833a9c`, clean tree (selection: `5790d02`, clean; they differ only in experiment files and the summary script) |
| `topology_id` A | `748154abc258ce8c2460a1fadef805d163efeab51c34564af06e28b175506833` |
| `topology_id` B, C, D | `2a33649e5785c6bf467c30d51ca2ed6015706a3df6b17a6b1268584c7532352f` |
| `infrastructure_id` A, B, C | `90b27e16e12f55272c878d2a409f184f74b7164c83db4067cb720fb1dedc5558` (`architectures_lab`) |
| `infrastructure_id` D | `a28dd1a2b8c4a8169a1f3e950fe9723482cf5f6835f43eb23591a37b7ac7e99b` (`architectures_lab_dynamic`: the schedule is part of the id) |
| `config_id` A | `aec9770cbfdc4f76220a1214907f31e9b71a0326e3a6f87627fbcf820f151b84` |
| `config_id` B | `54d3a0dd72538643801203c0489e2feb47e657eb9075395f150524b8424004f6` |
| `config_id` C | `9e4bb235e14005ad5fa5ec5efbe7f85290f7804615274186bb5fecf68c915b64` |
| `config_id` D | `432ac0377da52920bfcd1f26ba4118f9d8f55e1ad966c91191543d9fab22c8cd` |
| `config_id` selection, lr 0.1 | `4a3cca1291399f1adf2dbf6be6554433a5c18599900153d5fd09fabeedfc5f07` |

Each run's `run_id` and `run_hash` are in `run_metrics.csv` (and `validation.csv` for the selection); the signed `run.json`, `summary.json` and `model.npz` of each are in `runs/` (main), `select/` and `logical_sim/`. The event files (2.7 MB each) are not committed.

## Results (mean ± 95% CI over 5 seeds; `run_metrics.csv`)

| Arm | Global SWELL | Global WESAD | Edge received | Edge trained | Round time (s) | Time to 0.6 (s) | Run (s) | WAN (MB) | Edge–fog (MB) |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.551 ± 0.039 | 0.753 ± 0.013 | 0.583 ± 0.019 | 0.584 ± 0.028 | 5.48 ± 0.00 | 95.5 ± 18 (3 of 5) | 109.8 | 198.4 | — |
| B | 0.552 ± 0.036 | 0.754 ± 0.012 | 0.582 ± 0.021 | 0.583 ± 0.027 | 1.88 ± 0.01 | 32.8 ± 6.1 (3 of 5) | 37.7 | 28.6 | 198.4 |
| C | 0.552 ± 0.036 | 0.754 ± 0.012 | 0.582 ± 0.021 | 0.583 ± 0.027 | 7.56 ± 0.08 | 131.6 ± 24 (3 of 5) | 151.3 | 28.6 | 198.4 |
| D | 0.544 ± 0.042 | 0.751 ± 0.014 | 0.580 ± 0.021 | 0.572 ± 0.018 | 7.72 ± 0.02 | 328.2 ± 130 (3 of 5) | 339.1 ± 120 | 18.2 | 196.0 |

"(3 of 5)": seeds 0, 1 and 4 reach the target, in every arm; seeds 2 and 3 never do, and count with their run's end. D lost 2 cloud rounds in every run and moved all 18 SWELL edges once.

## Reading

- **The learning does not change with the architecture, as it should not.**
  - B and C give the same final model bit for bit in every seed: with quorum 1.0 and no losses, the devices change when updates arrive, never which ones are combined.
  - A and B differ by up to 8·10⁻³ in a weight and by at most 0.007 in a seed's score; the means are within 0.001. With FedAvg weighted by samples at both levels the two-level average equals the flat one (a protocol test checks it with the stub trainer), so the gap is most likely float rounding of the two-stage average carried through 20 rounds of training; that is not tested.
  - D loses 0.008 on SWELL on average, inside the intervals: two lost rounds out of 20.
  - So the experiment cannot rank the architectures on learning: they tie. It ranks them on system metrics, as spec §12 asks when the learning ties.
- **On system metrics the four differ, and every difference has a mechanism in the model:**
  - **A sends 7× more over the WAN than B** (198 against 29 MB): every edge's update crosses the 10 Mbps fog–cloud links, which the 28 edges share. That makes its rounds 2.9× longer (5.5 against 1.9 s). The fogs of B combine 28 updates into 4 before the WAN.
  - **C's rounds take 4× B's** (7.6 against 1.9 s): with quorum 1.0 every round waits for the slowest edges, the low-power IoT devices at 150 samples/s. Its bytes are B's: the devices change time, not traffic.
  - **D shows that the cost of a fog failure here is the round deadline, not the failover.** The failover moves the 18 SWELL edges to the WESAD fogs within 5 s of the failure, and the run goes on with them. But the cloud waits for all four fogs (quorum 1.0): the round open when `fog-1` fails ends at the deadline of whoever is waiting, the cloud's 300 s (seeds 0 and 3, whose round opened just after the failure) or the fogs' 120 s (seeds 1, 2 and 4, whose round was already open). Then the next round fails as well, because the two returned SWELL fogs have no children left. Hence 268 or 446 s instead of C's 151 s, and the wide interval.
  - The deadlines and quorums are the topology's, the same in every arm, and were not tuned; with a cloud quorum below 1.0, or one that counts only live fogs, D would behave differently. That is a design question for the protocol, not a result.
- **The ranking on time depends on the physical network.** Without an infrastructure (`logical_metrics.csv`: the same federations on the topology's implicit LAN links, no device model), A finishes in 0.12 s and B in 0.22 s, because B has one more hop and no WAN to save. With the infrastructure, B is 2.9× faster. Both are the simulator's predictions; neither is a measurement.
- **The learning scores are identical with and without the infrastructure** (`logical_metrics.csv` against the table above, to the fourth decimal), so the infrastructure only moves time and traffic in these runs.

## Decision fidelity between `sim` and `local` (`fidelity/`)

- **What runs.** The `local` backend refuses an infrastructure (I10, I14), so only what tells A from B in the federation itself can run on it. That is `experiments/infrastructure_architectures_logical.yaml`: the two logical federations, with the same subjects, seeds and training. `scripts/run_architectures_fidelity.py` runs them on each backend with the MQTT links pointed at the broker. `onion_fl fidelity` then pairs the runs by configuration and seed, with `sim` as the reference (A) and `local` as the backend under test (B). C and D differ from B only in devices and host failures, which `local` cannot reproduce. They wait for the emulated backend (I10, #195) and the distributed one (I14, #199).

  ```bash
  python scripts/run_architectures_fidelity.py --backend sim --broker localhost:18840 --workers 3
  python scripts/run_architectures_fidelity.py --backend local --broker localhost:18840   # one run at a time
  onion_fl fidelity runs/architectures_sim runs/architectures_local --experiment infrastructure_architectures_logical \
      --metric macro_f1 --metric time --metric bytes --metric time_to_target --series 'global/*' --target macro_f1=0.6 \
      --csv fidelity.csv --html fidelity.html > fidelity.json
  python scripts/architectures_summary.py runs/architectures_<backend> --experiment infrastructure_architectures_logical --out <backend>_metrics.csv
  ```

- **Identity.** Code at commit `76d1bd8`, clean tree, for all 20 runs. The broker is Mosquitto 2.1.2 (`eclipse-mosquitto:2`) in Docker Desktop on Windows, at `localhost:18840`. The broker is part of the topology, so these `topology_id`s differ from those of `logical_sim/` (port 18836). The `sim` runs here and there give the same final models bit for bit.

  | Identity | A (`cloud_swell_wesad`) | B (`four_fogs_swell_wesad_physical`) |
  |---|---|---|
  | `topology_id` | `8b641b981104750b4af0430477f3be0ca12c690edafbbe78648f5e8689487cb6` | `c5478110dfc1…` (full in each `run.json`) |
  | `config_id` sim | `7a5037da10b43bcf5478ff82054c77f3e4d40e38a957e57ae3985456eb5a1d31` | in `sim_metrics.csv` |
  | `config_id` local | `32e9f8e2b5a7c3c4beb1e70864caee873ef9f5e85568c8ab5729ec6181fde17f` | `f8946bbe508c…` (full in `local_metrics.csv`) |
  | `design_id` (the pairing key) | `253dc7ab51cff7de3b2bc7304595b3baacc7ed06f46ac9dff52d49a0576377c4` | `6f1e770e181e662429e97f0cb1d9c65611abe527d2ef201a829696bd1fe1779e` |
  | `local` processes | 1 (the cloud and its 28 edges) | 5 (the cloud, and each fog with its edges) |

  - Each run's `run_id` and `run_hash` are in `sim_metrics.csv`, in `local_metrics.csv` and in the pairs of `fidelity.json`.
  - The signed `run.json`, `summary.json` and `model.npz` of each run are in `fidelity/sim/` and `fidelity/local/`.
  - The event files (3 MB each) are not committed. The functional and performance levels read them, so `fidelity.json` cannot be recomputed from this folder alone.

- **Functional: identical.** All 10 pairs are complete and match on:
  - the final model, bit for bit (`max_abs_diff` 0);
  - every node's sequence of protocol events;
  - the bytes per logical link and round: 229.4 MB per A run and 261.2 MB per B run, the same on both backends.

- **Performance: the times do not compare, by construction.** Without an infrastructure, `sim` has no device model: training takes no simulated time, and only the latency of the implicit LAN links counts. `local` measures this machine's wall clock, which is mostly training. The relative errors `(local − sim) / sim`, with 95 % intervals, are:

  | Quantity | Relative error | 95 % interval |
  |---|---|---|
  | round time | 352× | 254–450 |
  | run duration | 384× | 288–480 |
  | link latency | 392× | 256–527 |
  | time to the target | 432× | 324–541 |

  So the logical half of `sim` does not predict wall-clock time, and it does not claim to.

- **Decision: the same ranking on learning and traffic, the opposite ranking on time.**

  | Metric | `sim` A / B | `local` A / B | τ-b | Top-1 | Regret (relative) | Seeds' ceiling (top-1, regret) |
  |---|---|---|---|---|---|---|
  | final global macro-F1 (pooled) | 0.5940 / 0.5951 | 0.5940 / 0.5951 | 1.0 | yes, p = 1.0 | 0 | 0.4, 0.0018 |
  | bytes per run | 229.5 / 261.2 MB | 229.5 / 261.2 MB | 1.0 | yes, p = 1.0 | 0 | 1.0, 0 |
  | run duration (s) | 0.116 / 0.218 | 62.4 / 50.3 | −1.0 | no, p = 0.0 | 0.103 s (0.89) | 1.0, 0 |
  | time to macro-F1 0.6 (s) | 0.076 / 0.143 | 45.9 / 37.5 | −1.0 | no, p = 0.03 | 0.067 s (0.87) | 1.0, 0 |

  - **On learning, the agreement does not inform.** `local` reproduces `sim` exactly, but A and B differ by 0.001, while the seeds' own ceiling of top-1 agreement is 0.4. They tie, as in the main experiment, and picking a winner between them is a coin toss.
  - **On bytes, the backends agree exactly.** The traffic is the protocol's, and the protocol is the same on both.
  - **On time, the backends disagree, and the mechanism is not the network.**
    - In `sim`, B is slower because it has one more hop and no compute to spread.
    - In `local`, B finishes first (50 against 62 s): its 28 edges train in four fog processes, while A's train one after another in the cloud's single process. In seed 0 the edges' trainings of a round spread over 2.15 s in A, against 1.32 s in B.
    - So `local` here ranks this machine's parallelism, and the logical `sim` ranks link hops. Neither is the time of a deployed continuum.
    - In B, the run time on `local` grew from seed to seed (45 to 62 s), for a reason not looked into (the machine was shared with other work), so its interval is wide (± 9.7 s, against ± 2.8 s for A).
  - **So the decision fidelity of `sim` against `local` holds on the metrics the protocol fixes (model, bytes) and fails on time.**
    - The cause is a model gap, not a defect of either backend: the logical half of `sim` has no compute model.
    - A comparison on time needs the infrastructure on both sides, which is the emulated backend's job (#195).

- **Before the fix, the `local` half stalled.** The runs of A at commit `b56cfeb` delivered about 75 of 270 messages and failed every round at its 300 s deadline. The cause was the MQTT transport behind Docker Desktop's port forwarding on Windows, fixed in `76d1bd8`; the diagnosis follows. Those runs are not committed: none finished.

### Why the `local` runs stalled

Every test below ran on this machine with no other work running, against one Mosquitto container at `localhost:18840` with paho 2.1.0. Messages are 150 KB (one update of this model) unless said otherwise.

| Probe | Result |
|---|---|
| One paho client, publishing to a topic it subscribes to: 5 × 150 KB, 30 × 1 KB or 300 × 15 KB | all arrive in under 0.1 s |
| The same client: 8, 12, 16, 20 or 30 × 150 KB, at QoS 1 or 0, with or without `TCP_NODELAY` | none arrive (one of 30 with `TCP_NODELAY`), then the client drops on keepalive |
| The same, instrumented | paho handed all 1.2 MB to the socket (no partial write, nothing left in its queue). In 10 s it got back 9 bytes, the CONNACK and SUBACK, and not one PUBACK |
| Two clients, one publishing and one subscribing: 30 × 150 KB at QoS 1, or 100 × 150 KB at QoS 0 | all arrive, in 0.02 s and 0.05 s |
| The single-client probe from a container in the broker's own network (no port forwarding) | 8 and 30 × 150 KB arrive in 0.01 s |
| Raw TCP, no MQTT: a Python echo server in a container at a published port, with the client writing and reading at once from two threads | 750 KB is echoed in 0.05 s. For 1.2 MB, 4.5 MB or a single 2 MB write, nothing comes back in 15–20 s. Inside the container's network, 4.5 MB is echoed in 0.05 s |

- **The stall is in Docker Desktop's port forwarding on Windows,** not in paho, in Mosquitto, in the QoS 1 acknowledgements or in Nagle. One TCP connection that carries more than about 1 MB each way at once stops for good.
  - A deadlock in paho's network loop is ruled out. The transport publishes from the runtime's loop thread, never from a paho callback, and two clients on the same broker get through.
  - Mosquitto's inflight and queue limits are ruled out too. The same single client gets through inside the container's network, and raw TCP stalls the same way without MQTT.
- **The federation hit this in every round.** A process that hosts a parent and its children sends the model to its own children, and its one client receives it back at the same time. That is the cloud and its 28 edges in A, and each fog with its edges in B.
- **The fix is in `76d1bd8`.**
  - `MqttTransport` holds two connections, one that listens on the inboxes and one that publishes, so each carries one direction.
  - It starts only once the broker has acknowledged every inbox. With two connections, a message to a node in the same process could otherwise arrive before that node listens.
  - `tests/test_transport_mqtt.py` sends 30 × 150 KB through one transport to its own inbox. Before the fix, 0 of 30 arrived; after it, all 30 arrive, in order.
  - `tests/test_runtime_equivalence.py` still passes.
  - The two connections cost nothing on a broker that is reachable directly.
- **A second limit turned up on the way.** With its default configuration, this Mosquitto image refuses a packet of about 2 MB or more and drops the client: 1,999,900 bytes pass, 2,000,100 do not, and the log says `oversize packet`. This model's 150 KB is far below it; a larger model needs a broker configured for it.
- **Not measured: the latency cost of Nagle (#197).** `TCP_NODELAY` did not cure the stall. With the fix, training time dominates these runs, so Nagle's effect on them was not isolated.

## Limits

- One simulator, illustrative profiles, one placement (α = 0), one model and one training setup. The size of the differences between architectures follows from the profiles chosen, not from measured devices.
- The broker is not placed (each link is its direct path). With a broker in the cloud, every edge–fog message of B would cross the WAN twice and B's advantage would shrink; that variant was not run.
- The target of 0.6 is reached by the same 3 seeds in every arm, so the time to target ranks the arms exactly as the round time does: here, rounds to target × time per round (the factorised predictor of spec §12) is the whole story.
