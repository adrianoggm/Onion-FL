# Calibration of the emulated lab, and its check (issue #197)

> **Superseded** by [`../calibration_emulated_pinned/`](../calibration_emulated_pinned/INDEX.md): calibration `b55104b3…`, fitted on four pinned sessions with intervals between sessions. This one stays as the record of why the first method failed.

`calibrations/emulated_lab.yaml` was measured three times on 2026-11-06, each time in a fresh emulation of `infrastructures/emulated_lab.yaml` (Docker Compose + netem, the `emulated` backend). One session was fitted; the other two only check it. No FL experiment was run for this calibration, and no measurement of a check session went into the fit (`onion_fl calibrate check` refuses the session a calibration was fitted on).

```bash
onion_fl calibrate measure calibrations/emulated_lab.yaml --out <fit session>
onion_fl calibrate fit <fit session> --infrastructure-out infrastructures/emulated_lab_calibrated.yaml
onion_fl calibrate measure calibrations/emulated_lab.yaml --out <check session>
onion_fl calibrate check <check session> --infrastructure infrastructures/emulated_lab_calibrated.yaml --out check_<n>.json
```

The sessions ran with `--image onionfl-a197-emulated:latest --project onionfl-a197 --subnet 10.197.0.0/16`, which only name the containers and their networks.

## Identity

These are microbenchmarks, not FL runs: there is no topology, `config_id`, `run_id` or `run_hash`. What identifies them:

| Identity | Value |
|---|---|
| `calibration_id` | `5d7d0e7011bf1fd693725bcb85ac61c5abae3d460867169f8909954c1b937cde` (`calibrations/<id>/`, its fit session's 3 716 measurements) |
| `infrastructure_id` measured | `da875674ad22d77f7aabcc6d9ac2fe642e9e8be204889668bb4adc5f238df43e` (`emulated_lab`, the one of `results/fidelity_emulated`) |
| `infrastructure_id` calibrated | `e0a82033b2e81c912ad0d0e5d7331830738a7c9ba1b077afe7952b2321c5de78` (`infrastructures/emulated_lab_calibrated.yaml`) |
| workload `0ed95108ec44feea` | `modular_mlp` adapter 32, trunk [16], `standard` trainer, batch 32, on SWELL physiology (3 features): 690 parameters |
| workload `f81f0235a3520d08` | `modular_mlp` adapter 64, trunk [64, 32], same trainer: 6 562 parameters |
| fit session | started 02:09 UTC, code `bad5145`, clean tree |
| check session 1 | started 02:24 UTC, code `bad5145`, clean tree; 3 543 measurements in `check_session_1/` |
| check session 2 | started 02:40 UTC, code `c21e9cf`; `dirty: true` only because `check_session_1/` was in this folder, not yet committed; 3 954 measurements in `check_session_2/` |

Environment (in `calibration.json` and each `session.json`): Docker Desktop 28.3.2 on WSL2 (kernel 6.6.87.2), 24 logical CPUs, Windows 11, Intel Core i7-14650HX. Torch in the containers saw 12 threads. The laptop was shared with other workloads, among them other agents' containers, that this calibration neither controls nor records.

## What was measured

- **Training.** Five edge devices (1 CPU, 2 GB each) trained at once, as the edges of the lab's experiments do. Each trained in turn the models of 4 real SWELL training subjects (the roles of `experiments/fidelity_emulated.yaml`), 13 trainings per device and workload, 5 local epochs each (about 640 samples). The first training of each device and those outside the window when all five were training are left out.
- **Aggregation.** `fedavg` over 10 and 20 models of the small workload, on both fogs and the cloud at once, for 5 s per size, timed in batches of 100 combinations.
- **Paths.** MQTT through the broker on `cloud-1`, QoS 1, the project's own `MqttTransport`, both directions of 5 host pairs plus `cloud-1` to itself. Messages of 100 B to 100 kB, 15 of each, one at a time and 0.5 s apart, with the other hosts idle. The one-way delay uses the clock the containers share. The first message of each direction is left out.

## The calibration (fit session)

| Value | Fitted | 95 % CI | Nominal (`emulated_lab`) |
|---|---|---|---|
| `edge_device` train rate, small workload (samples/s) | 934 | [838, 1 051] (41 trainings) | 2 000 (a guess) |
| `edge_device` train rate, large workload | 751 | [679, 838] (40) | — |
| `fog_small` aggregate rate (bytes/s) | 2.10 × 10⁸ | [2.07, 2.13] × 10⁸ | 1.0 × 10⁸ |
| `cloud_vm` aggregate rate | 2.13 × 10⁸ | [2.09, 2.18] × 10⁸ | 5.0 × 10⁸ |
| `field`–`fog_a` latency (ms) | 10.04 | [9.86, 10.16] | 10 |
| `fog_a`–`dc` latency | 40.59 | [40.52, 40.72] | 40 |
| `fog_b`–`dc` latency | 60.59 | [60.55, 60.64] | 60 |
| `fog_a`–`dc` bandwidth up / down (Mbit/s) | 6.59 / 6.72 | | 10 |
| `fog_b`–`dc` bandwidth up / down | 7.26 / 7.20 | | 10 |
| `field`–`fog_a` bandwidth | 38.2 (derived: it bottlenecks no measured path) | | 50 |
| loss, every link | 0 of 60 messages per path | Wilson [0, 0.061] | 0 |
| MQTT, `cloud-1` to itself | 0.49 ms | [0.45, 0.54] | not in the simulation |

`field`–`fog_b` is crossed by no measured path (the field reaches `dc` through `fog_a`), so it is not calibrated and keeps its nominal fields.

## The check

**Acceptance is not met on this machine.** A calibrated profile does not reproduce its own microbenchmarks within their 95 % interval in another session.

Compute: the calibrated rate against the rate measured in each check session (with that session's interval), and the error of the time the simulation predicts for that session's trainings or combinations, (predicted − measured) / measured.

| Quantity | Calibrated | Check 1 | Check 2 |
|---|---|---|---|
| train, small workload (samples/s) | 934 | 723 [591, 895], time −23 %: outside | 1 442 [1 143, 1 807], time +54 %: outside |
| train, large workload | 751 | 464 [377, 585], time −38 %: outside | 901 [737, 1 117], time +20 %: **inside** |
| aggregate, `fog_small` (10⁸ B/s) | 2.10 | 1.87 [1.84, 1.90], time −11 %: outside | 2.46 [2.44, 2.48], time +17 %: outside |
| aggregate, `cloud_vm` | 2.13 | 1.87 [1.83, 1.92], time −12 %: outside | 2.44 [2.41, 2.47], time +14 %: outside |

Paths: the delay the simulation predicts with the calibrated links (`paths.effective` and its `LinkProfile`, through the broker), against the measured mean of each direction and size. 10 directions per size; the ranges are those of both checks together (they differ by at most 0.6 point):

| Size | Relative error | Inside the interval |
|---|---|---|
| 100 B | −2.2 % to +0.3 % | 1–2 of 10 |
| 1 kB | −0.4 % to +0.8 % | 1–2 of 10 |
| 5 kB | −2.5 % to +4.0 % | 0–1 of 10 |
| 10 kB | −4.3 % to +7.7 % | 0 of 10 |
| 30 kB (not fitted) | −60 % to −39 % | 0 of 10 |
| 100 kB (not fitted) | −60 % to +21 % | 2 of 10 |

`cloud-1` to itself is predicted at 0 (the simulation's loopback) and measured at 0.5–0.7 ms.

## Reading

Measured:

- **The compute rates move between sessions far more than inside one.** The small workload trained at 723, 934 and 1 442 samples/s in three sessions an hour apart, with the same code, image and spec, and each session's interval is about ±12 %. The interval of a session therefore says little about the next one, and no single calibration falls inside both checks. The aggregation rates move the same way: the predicted times are 11–12 % short in check 1 and 14–17 % long in check 2.
- **Check 1 overlapped with work of this calibration on the same laptop.** During its first minutes, two one-round simulations of `fidelity_emulated` ran on the host to check that a calibrated infrastructure resolves its workload. That alone may explain part of its −23 %; check 2 was run with nothing else of this issue running and missed by +54 % the other way (the devices trained faster than in the fit). Both are committed.
- **The FLOPs estimate is wrong by a factor of 5 to 6 for these models.** The large model has 9.5 times the FLOPs of the small one but trained only 1.2–1.6 times slower. Estimating one from the other missed by +494 % to +511 % (small from large) and −83 % to −84 % (large from small). For models this small, the time goes to per-batch overhead, not arithmetic. This is why the harness refuses estimated rates.
- **The calibrated links reproduce messages up to 10 kB within 8 %**, and small messages within 2.5 %. The fitted latencies are within 0.6 ms of the netem latencies. The intervals of the means are a few hundredths of a millisecond wide, so most predictions still fall outside them.
- **The simulation's path model does not hold beyond 10 kB.** At 30 kB, the measured delay is 1.6–2.5 times the prediction, and at 100 kB the error depends on the path (−60 % to +21 %).
- **Through the broker, a path's goodput is that of its slowest leg divided by the number of legs at that rate.** Device to fog crosses `fog_a`–`dc` twice and gets about half the goodput of fog to cloud, which crosses it once. The fitted bandwidths (6.6–7.3 Mbit/s) are a compromise between the two.
- **The links absorb the transport.** MQTT adds 0.5 ms from a process to the broker on its own host and back out.

Hypothesised, not measured here:

- The spread between sessions comes from the shared laptop (other agents' containers and processes, turbo and the hybrid P/E cores, which the I9 spike already blamed for 6–12 %). No session recorded the host's load. A calibration that should hold across sessions needs several fit sessions with an interval between sessions, or a dedicated machine, or pinned cores (`cpuset`, not measured).
- Beyond the first TCP window (about 14.6 kB), slow start after 0.5 s of idle adds round trips that a constant bandwidth cannot model. The broker's store-and-forward makes a path's transfer time the sum of its legs, not that of its narrowest segment, as the simulation assumes. A simulator change (per-leg transmission through the broker, slow start) would be needed. The FL messages of `results/fidelity_emulated` are 12 kB on average, at the edge of the fitted range.
- A preliminary session of this work, not committed and with an older spec, sent messages 50 ms apart instead of 0.5 s. Small messages then took 2–3 times their propagation: paho does not set `TCP_NODELAY`, nor does mosquitto by default, and Nagle held each message until the previous one's ACK arrived. A burst of updates through the broker may queue the same way in an FL run. This is a candidate for the 0.75 s of unexplained link latency in `results/fidelity_emulated`, to be measured in #198.

## Consequences

- `infrastructures/emulated_lab_calibrated.yaml` cites the calibration, and a `sim` run on it takes these rates and links. A run whose model is not calibrated is marked `estimated`, and `onion_fl fidelity` refuses it.
- By the rule of `docs/RULES.md` §3, a validation (#198) must use another topology, model or dataset, and cite this `calibration_id`. Given the spread above, the compute errors of such a validation would be dominated by the machine, not by the simulator, until the calibration covers the variation between sessions.

## Files

- `../../calibrations/5d7d0e70…/`: `calibration.json` (fitted values, environment, code, spec) and the fit session's `measurements.jsonl`.
- `check_session_1/`, `check_session_2/`: `session.json` (spec, hosts, workloads, environment, code, tc commands) and `measurements.jsonl` of each check session.
- `check_1.json`, `check_2.json`: the output of `onion_fl calibrate check`, one entry per (profile, workload), aggregating profile, FLOPs estimate and (path, size).
