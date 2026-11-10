# Calibration of the emulated lab on pinned sessions, and its check (issue #197)

This replaces the one-session calibration of [`../calibration_emulated/`](../calibration_emulated/INDEX.md), whose check failed: the training rates of three sessions an hour apart did not overlap, and each session's interval said nothing about the next.

What changed in the method, and why:

- **Fixed CPUs.** Torch ran 12 threads under a 1-CPU quota in the first calibration. The spec's `pin` now gives each host fixed CPUs of Docker's VM (`cpuset`), one per profile CPU and never two siblings of a core. In name order, `cloud-1` gets 2 and 4, the devices 6 to 14, `fog-a` 16 and `fog-b` 18, and the routers and the broker share 20-23. Every training now runs 1 thread (`threads` in each row). These are the VM's CPUs, not physical cores: Hyper-V still moves them over the P- and E-cores.
- **Warm-up and load.** The first 2 trainings of each host are left out (`train.warmup`). Before and after each phase, the session records the machine's busy % and performance % (Windows counters) and the VM's load average (`load` in each `session.json`).
- **An interval between sessions.** `calibrate fit` takes several sessions. Each value is the geometric mean of the sessions' values. Its interval is the 95 % prediction interval of the value one more session measures: `exp(m ± t₀.₉₇₅,ₖ₋₁ · s · √(1 + 1/k))`, where `m` and `s` are the mean and standard deviation of the logs. The criterion is not looser than before, it is a different one: the interval of one session's mean is the wrong question for "does it hold in another session", and it failed exactly there. The spec (I1 §13) records this decision.
- **What a check passes.** For a session the fit never saw, each calibrated quantity it measures must fall inside the calibrated interval (`inside`). A path must also have the simulator's prediction inside it (`predicted_inside`), since a path is not fitted directly: its prediction is the sum of its links. The old criterion, the prediction inside the session's own interval, is still reported (`inside_session_ci`).
- **Paths through the broker.** The simulator now sends a message once per leg, child to broker and broker to parent (store and forward), and the link bandwidths are fitted that way. The old model took the narrowest segment of the whole path.
- **Valid range.** The path model holds up to `valid_up_to` = 10 000 bytes. A sim run that sends more over a calibrated link lists it in `run.json` (`outside_calibration`), and `onion_fl fidelity` refuses that run.

## Identity

| Identity | Value |
|---|---|
| `calibration_id` | `b55104b3f71d684dedc413a000eec3a67eb1caba134cc9ff6a95648fb35c8dfc` (`calibrations/<id>/`, 13 126 measurements from 4 sessions) |
| `infrastructure_id` measured | `da875674ad22d77f7aabcc6d9ac2fe642e9e8be204889668bb4adc5f238df43e` (`infrastructures/emulated_lab.yaml`) |
| `infrastructure_id` calibrated | `607f32090edd3603590b2d97e5648e95c0a199e33d3e97f368c268f1a7311e69` (`infrastructures/emulated_lab_calibrated.yaml`) |
| spec | `calibrations/emulated_lab.yaml` (pin above, `train.jobs` 14, `warmup` 2, `links.fit_up_to` 10 000) |
| code of every session | `d3d050e`, clean tree (each session's `code`) |
| image | `onionfl-a197b-emulated:latest`, `sha256:a826e66e…`: torch 2.14.1, numpy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, scipy 1.17.1 |
| workload `0ed95108ec44feea` | `modular_mlp` adapter 32, trunk [16], `standard`, batch 32, SWELL physiology (690 parameters) |
| workload `f81f0235a3520d08` | adapter 64, trunk [64, 32], same trainer (6 562 parameters) |

These are microbenchmarks, not FL runs: they have no topology, `config_id`, `run_id` or `run_hash`.

The sessions are listed below. Their roles were declared before measuring: s1, s2, s4 and s5 fit, s3 and s6 check. After the parent session restarted, s7 and s8 were added as checks, on another day.

| Session | Start (UTC) | Role | Small workload (samples/s) | Large | Aggregate (10⁸ B/s) | Machine busy % | Performance % | VM load |
|---|---|---|---|---|---|---|---|---|
| s1 | 2026-11-06 09:06 | fit | 12 864 | 14 716 | 1.31 | 60 | 147 | 3.5 |
| s2 | 09:20 | fit | 20 398 | 13 131 | 1.60 | 54 | 151 | 5.3 |
| s3 | 09:34 | check | 20 036 | 12 782 | 1.50 | 42 | 155 | 6.6 |
| s4 | 09:48 | fit | 29 536 | 20 833 | 1.74 | 32 | 161 | 4.6 |
| s5 | 10:02 | fit | 10 759 | 16 362 | 1.35 | 38 | 158 | 3.9 |
| s6 | 10:16 | check | 14 391 | 16 944 | 1.51 | 42 | 156 | 3.9 |
| s7 | 2026-11-09 15:56 | check | 34 134 | 27 136 | 2.33 | 20 | 138 | 0.5 |
| s8 | 16:10 | check | 31 709 | 30 917 | 2.63 | 19 | 116 | 1.1 |

The busy %, performance % and VM load are means of the values taken at each phase's edges. Each session took 14 minutes.

Machine: Intel Core i7-14650HX (24 logical CPUs), Windows 11, Docker Desktop 28.3.2 on WSL2 (kernel 6.6.87.2), with no other Onion-FL agent running.

- **2026-11-06:** Docker Desktop's Kubernetes cluster was running (Prometheus, Grafana, Loki, a MySQL and a backend, all idle), as were the user's desktop applications.
- **2026-11-09:** Docker Desktop had been restarted, and the Kubernetes workloads were scaled to 0, except a `promtail` pod. Three idle containers of other projects ran (`fg-obs-grafana-1`, `redsaras-api`, `postgres`).


## The calibration

| Value | Fitted | 95 % prediction interval | Fit sessions |
|---|---|---|---|
| `edge_device` train, small workload (samples/s) | 16 990 | [3 348, 86 260] | 12 864, 20 398, 29 536, 10 759 |
| `edge_device` train, large workload | 16 020 | [7 953, 32 270] | 14 716, 13 131, 20 833, 16 362 |
| `fog_small` aggregate (10⁸ B/s) | 1.49 | [0.92, 2.41] | 1.31, 1.60, 1.74, 1.35 |
| `cloud_vm` aggregate | 1.49 | [0.92, 2.41] | 1.31, 1.59, 1.74, 1.35 |
| `field`–`fog_a` latency (ms) | 10.17 | [9.37, 11.03] | 10.51, 9.99, 10.10, 10.08 |
| `fog_a`–`dc` latency | 40.63 | [40.45, 40.81] | |
| `fog_b`–`dc` latency | 60.68 | [60.50, 60.86] | |
| `fog_a`–`dc` bandwidth up / down (Mbit/s) | 9.48 / 9.57 | [8.70, 10.34] / [9.26, 9.90] | |
| `fog_b`–`dc` bandwidth up / down | 9.73 / 9.59 | [9.33, 10.16] / [9.30, 9.88] | |
| `field`–`fog_a` bandwidth | 47.96 (derived: it bottlenecks no leg) | | |
| loss, every link | 0 of 236 messages pooled | Wilson [0, 0.016] | |
| MQTT, `cloud-1` to itself | 0.54 ms | [0.40, 0.72] | reported, not simulated |

With store and forward through the broker, the fitted bandwidths are 95–97 % of netem's 10 Mbit/s, the goodput the I9 spike measured. The old fit gave 6.6–7.3 Mbit/s, a compromise between paths that cross a link once and paths that cross it twice. Each of the 10 measured directions and 4 sizes up to 10 kB also has its delay interval (`paths` in `calibration.json`).

## The check

There are 54 quantities per check session: 2 training rates, 2 aggregation rates, 10 link fields and 40 path delays (10 directions × 4 sizes up to 10 kB).

| Check | Training | Aggregation | Links | Paths (measured and predicted inside) | All |
|---|---|---|---|---|---|
| s3 (same day) | 2 / 2 | 2 / 2 | 10 / 10 | 40 / 40 | **passed** |
| s6 (same day) | 2 / 2 | 2 / 2 | 10 / 10 | 40 / 40 | **passed** |
| s7 (3 days later) | 2 / 2 | 2 / 2 | 10 / 10 | 38 / 40 | failed |
| s8 (3 days later) | 2 / 2 | **0 / 2** | 10 / 10 | 39 / 40 | failed |

What fails, exactly:

- **s8, aggregation.** `cloud_vm` and `fog_small` combined 2.61 and 2.63 × 10⁸ B/s, above the interval's 2.41: 8 % past it.
- **Paths, 3 of 80.** In each case the simulator's prediction is inside the interval, but the delay that session measured is just past it:

  | Session | Path and size | Measured | Interval | Prediction |
  |---|---|---|---|---|
  | s7 | `field-edge_device-1` → `fog-b`, 5 kB | 120.68 ms | [119.27, 120.47] ms | 119.87 ms |
  | s7 | `fog-a` → `field-edge_device-1`, 1 kB | 94.06 ms | [91.91, 94.05] ms | 93.10 ms |
  | s8 | `field-edge_device-1` → `fog-a`, 1 kB | 93.45 ms | [92.47, 93.16] ms | 93.10 ms |

  The excess is 0.01–0.3 ms.

The acceptance holds on the two check sessions of the fit's day: a calibrated profile reproduces its own microbenchmarks within its interval in another session. It does not fully hold three days later, on a machine in another state: 49 of 54 quantities pass, and the misses are the ones above.

Other figures of the checks:

- **The old criterion still fails, as expected.** The prediction falls inside the check session's own interval in 0–1 of 2 training rates and 9–17 of 40 paths.
- **Path errors in the fitted range.** The prediction is within −2.2 % to +0.5 % of every measured delay up to 10 kB, in every check. The old model missed by −4.3 % to +7.7 %.
- **Timing errors.** The training time the simulator predicts is −20 % to +18 % off on the fit's day, and +69 % to +101 % off three days later: the devices trained about twice as fast. The aggregation time is 0–2 % off on the fit's day and +56 % to +76 % off three days later. All but s8's aggregation are inside the intervals: the intervals are that wide.
- **Past 10 kB**, the prediction is 12–56 % short in every check (`fitted_range: false`, not part of the acceptance).

## Reading

Measured:

- **Pinning made training at least 7 times faster, and the sessions did not get closer.** The small workload trained at 723–1 442 samples/s unpinned and at 10 759–34 134 pinned. Between the four fit sessions, an hour apart with the same code, image, spec and CPUs, it still moved by a factor of 2.7. Each session's own interval is ±4–11 %. The small workload's prediction interval is therefore a factor of 26 wide, and the large one's a factor of 4. These intervals are honest about the machine, but they are too wide to predict an FL run's time. The aggregation rate moved by a factor of 1.3 and its interval spans a factor of 2.6.
- **Three days later, everything ran faster.** On 2026-11-09, with Docker's Kubernetes at 0 pods and a VM load of 0.5–1.1 (against 3.5–6.6 on 2026-11-06), training ran 1.7–2.0 times faster and aggregation 1.6–1.8 times. The prediction intervals of the fit's day held the training rates but not s8's aggregation. The machine's state, not the code, sets the compute rates. A calibration describes the machine as it was in its fit sessions.
- **Within a session, the large workload was not slower.** In three of the eight sessions it trained faster than the small one, which has 9.5 times fewer FLOPs. For models this small the time goes to per-batch overhead, not arithmetic. The FLOPs estimate misses by +507 % to +1 020 % in one direction and −84 % to −91 % in the other (`flops_estimates`), which is why the harness refuses estimated rates.
- **The network side is calibrated tightly and holds.** Link latencies match netem within 0.7 ms (the MQTT overhead goes into them), and their intervals are a few tenths of a millisecond wide. With the store-and-forward model, the paths up to 10 kB fall within 2.2 % of the prediction, and 157 of the 160 delays measured in the four checks fall inside intervals of about ±0.6 ms.
- **Slow start after an idle connection is what bends large paths.** With messages 0.5 s apart, a 30 kB message from `fog-a` to `cloud-1` took 136.5 ms, against a prediction of 65.9 ms. With messages 50 ms apart (the Nagle sessions), it took 70.1 ms, 6 % above the prediction. Linux restarts the congestion window after an idle period longer than the retransmission timeout, so past the first window (about 14.6 kB) each message pays extra round trips. This is why `valid_up_to` is 10 kB. Modelling it would need the simulator to know the connection's idle time.

### Nagle

Two sessions of the links alone, with messages 50 ms apart, differ only in `links.nodelay` (`calibrations/emulated_lab_nagle.yaml` and `calibrations/emulated_lab_nodelay.yaml`). They started at 10:31 and 10:34 UTC on 2026-11-06, at `d3d050e` with a clean tree. Paths: `fog-a`↔`cloud-1`, `field-edge_device-1`↔`fog-a` and `cloud-1`→`cloud-1`, 15 messages per size.

| | Result |
|---|---|
| Up to 10 kB, every path and size | The difference between Nagle on and `TCP_NODELAY` on is at most 0.46 ms, 1.1 % of the delay, with either sign |
| 30 kB | Differences within their own spread (intervals of ±13 to ±57 ms) |

Nagle does not delay these messages. The 2–3× slowdown of small messages that an uncommitted preliminary session of the first calibration attributed to it does not reproduce. Each message here waits for its acknowledgement before the next goes, as an FL round's request and reply do. A burst of updates without waiting was not measured. Neither was whether Nagle explains the unexplained link latency of `results/fidelity_emulated`: that remains for #198. The per-size differences, with bootstrap intervals, are in `nagle_vs_nodelay.json`.

## Consequences

- **What it is good for.** `infrastructures/emulated_lab_calibrated.yaml` cites this calibration. A sim run on it takes these rates and links, and is refused by `onion_fl fidelity` if a rate is estimated or a message on a calibrated link exceeds 10 kB.
- **The FL backend is not pinned yet.** The emulated backend's FL runs (`--backend emulated`) are not pinned the same way. Unpinned, torch runs 12 threads under the quota and trains at least 7 times slower. A validation against this calibration (#198) must run its emulated runs with the same `cpuset` and one torch thread per CPU. Otherwise it compares two different machines.
- **Compare like with like.** Compute times of a validation should be compared within the calibrated interval, and on a machine in the state of the fit sessions. A validation on another day should report the machine's load next to the calibration's. Better, it should refit on sessions of its own day: a new calibration, declared as such (`docs/RULES.md` §3).

## Files

- `../../calibrations/b55104b3…/`:
  - `calibration.json`: the fitted values with their sessions, intervals and path delays, each fit session's start, code, CPUs, load and digest, and `valid_up_to`;
  - `measurements.jsonl`: the four fit sessions' measurements, each row with its `session` (0–3 = s1, s2, s4, s5).
- `check_session_<n>/` (n = 3, 6, 7, 8): `session.json` and `measurements.jsonl` of each check session.
- `check_<n>.json`: the output of `onion_fl calibrate check` on each one, with its `acceptance`.
- `nagle_session/`, `nodelay_session/`, `nagle_vs_nodelay.json`: the Nagle comparison.

To reproduce, each session in its own emulation (`--image onionfl-a197b-emulated:latest --project onionfl-a197b --subnet 10.197.0.0/16` only name the image, the containers and their networks):

```bash
onion_fl calibrate measure calibrations/emulated_lab.yaml --out <session>     # x 6
onion_fl calibrate fit s1 s2 s4 s5 --infrastructure-out infrastructures/emulated_lab_calibrated.yaml
onion_fl calibrate check <check session> --infrastructure infrastructures/emulated_lab_calibrated.yaml --out check_<n>.json
```
