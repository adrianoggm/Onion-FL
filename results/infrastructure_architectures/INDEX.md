# The same federation over four architectures (new framework, real data)

`experiments/infrastructure_architectures.yaml` was run on 2026-11-06 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows), on the `sim` backend. Issue #192, spec I1 §1 (question 2) and §12.

```bash
onion_fl run experiments/infrastructure_architectures_select.yaml --workers 3   # edge learning rate, validation only
onion_fl run experiments/infrastructure_architectures.yaml --workers 5
python scripts/architectures_summary.py runs --out run_metrics.csv
python scripts/run_architectures_fidelity.py --backend sim --broker localhost:18836 --workers 3
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

## Decision fidelity between `sim` and `local`: not obtained

- **What was planned.** The `local` backend refuses an infrastructure (I10, I14), so only what tells A from B in the federation itself can run on it: `experiments/infrastructure_architectures_logical.yaml`, the two logical federations with the same subjects, seeds and training, run on `sim` and on `local` by `scripts/run_architectures_fidelity.py` (which points the MQTT links at the broker), then `onion_fl fidelity runs/architectures_local runs/architectures_sim`. C and D differ from B only in devices and host failures, which `local` cannot reproduce; they wait for the emulated backend (I10, #195), and the physical one (I14, #199).
- **The `sim` half ran** (`logical_sim/`, `logical_metrics.csv`; commit `2833a9c`, clean): 10 runs, finished.
- **The `local` half did not finish.** Its first attempt exposed a bug: a real run did not host the training edges of a two-level topology, so the coordinator waited for edges no process ran (fixed in `b56cfeb`, with a test). After the fix, the five runs of A (commit `b56cfeb`, clean, five at a time) were stopped after 33 minutes of wall clock:

  | Run | Seed | Root CPU (s) | Messages sent / delivered | Edge trainings | Cloud rounds closed / failed |
  |---|---|---|---|---|---|
  | `20261106T014536Z-c5b4b476d95e` | 0 | 21 | 271 / 75 | 3 | 0 / 6 |
  | `20261106T014536Z-e4977de034f0` | 1 | 22 | 271 / 75 | 3 | 0 / 6 |
  | `20261106T014536Z-2c3de2e44407` | 2 | 26 | 274 / 78 | 6 | 0 / 6 |
  | `20261106T014536Z-03ec41dfa140` | 3 | 20 | 270 / 74 | 2 | 0 / 6 |
  | `20261106T014536Z-a68eeecbfa19` | 4 | 34 | 282 / 86 | 14 | 0 / 6 |

  The processes were idle (20–34 s of CPU in 2,000 s) while most messages were never delivered, and every cloud round failed at its 300 s deadline. A probe outside Onion-FL points at the transport on this machine: one paho client that publishes 30 messages of 150 KB (the size of this model's updates) at QoS 1 to a topic it subscribes to got none back in 120 s through the Mosquitto container, and the broker dropped it for keepalive timeout; 5 such messages, or 30 of 1 KB, went through at once. In the two-level federation, the root's process holds the cloud and its 28 edges behind one client. Why the burst stalls (TCP back-pressure between the client's reading and writing, Docker Desktop's port forwarding, or paho on Windows) is not established. These runs are not committed: none finished.
- **So no decision fidelity between `sim` and `local` is reported here,** and none is claimed. What would be needed: a transport that does not stall on bursts of large messages (or one client per node), then the `local` half of `infrastructure_architectures_logical.yaml` on a machine without other load, for 5 seeds, and `onion_fl fidelity`. On learning metrics the decision would not inform anyway: A and B tie.

## Limits

- One simulator, illustrative profiles, one placement (α = 0), one model and one training setup. The size of the differences between architectures follows from the profiles chosen, not from measured devices.
- The broker is not placed (each link is its direct path). With a broker in the cloud, every edge–fog message of B would cross the WAN twice and B's advantage would shrink; that variant was not run.
- The target of 0.6 is reached by the same 3 seeds in every arm, so the time to target ranks the arms exactly as the round time does: here, rounds to target × time per round (the factorised predictor of spec §12) is the whole story.
