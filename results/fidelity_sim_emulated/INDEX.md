# Fidelity of the simulation against the emulation, validated (issue #198)

**Status: run on 2026-11-09.** Of the pre-registered hypotheses, H1a, H2a, H3a and H3c are met; H1b, H1c, H2b, H2c and H3b are not.

`experiments/fidelity_sim_emulated.yaml` validates the `sim` backend (B) against the `emulated` backend (A) at the three levels of I1 §12: functional, performance and decision. It uses the calibration of #197, closed and cited, on loads that calibration never saw. The hypotheses, the criteria and the analysis are fixed in [PREREGISTRATION.md](PREREGISTRATION.md), written before any run.

## The calibration cited

`b55104b3f71d684dedc413a000eec3a67eb1caba134cc9ff6a95648fb35c8dfc`: the refit of #197 on four sessions with pinned CPUs (`results/calibration_emulated/`), with image `onionfl-a197b-emulated:latest` (`sha256:a826e66e…`) and `valid_up_to` 10 000 bytes. It replaced the provisional single-session `5d7d0e70…` on 2026-11-09, before any run, following step 1 of the pre-registration (§2). The harness's `--calibration` refuses any run that cites another.

### The machine on the day of the runs, recorded before running

The pre-registration handles drift within a session (seed-by-seed order, pairing by seed). It does not address a change of the machine between the calibration's fit and the validation, so this is recorded here before any run, and the hypotheses are not changed:

- The four fit sessions ran on 2026-11-06, with Docker Desktop's Kubernetes active.
- The calibration's checks on 2026-11-09 ran on a quieter machine (Kubernetes scaled to 0). It trained 1.6–2 times faster than the fit, and one check session's aggregation came out 8 % above the interval.
- The validation runs on 2026-11-09 on that quieter machine. No other agent is running and Kubernetes is scaled to 0. Three idle containers of other projects remained (`fg-obs-grafana-1`, `redsaras-api`, `postgres`), plus Kubernetes' `promtail` log collector, which kept running after the scale-down; they were left alone.
- **So the emulated compute may be faster than the calibration predicts.** The simulation would then be the slower one. The compute errors (H2a) and the round-time errors (H2b), (B − A) / A, may then be positive, by about +0.6 to +1.0 if the whole 1.6–2× shows up, for a reason that is the machine's and not the simulator's.
  - H2a's band, from the refit's prediction interval, is wide: [−0.803, +4.076]. A miss of that size still falls inside it.
## Setup

- SWELL physiology, the calibrated workload (`modular_mlp` 32/[16], batch 32), `npz` codec, 10 rounds, seeds 0–4.
- Infrastructure `infrastructures/fidelity_validation_lab.yaml`: the calibrated lab with the 5 devices the calibration measured. Every container was pinned to its session's CPUs, and torch saw 1 thread and 1 CPU on the devices (checked in the first run's containers).
- Six configurations: every edge under the cloud, or under one fog on `fog-a` or on `fog-b`, times 1 or 5 local epochs.
- The pre-registration (§3) gives the full design.

```bash
python scripts/run_fidelity_validation.py --backend emulated --calibration b55104b3…   # 16:39:56–16:52:42 UTC
python scripts/run_fidelity_validation.py --backend sim --calibration b55104b3… --workers 3   # 16:53:06–16:54:30 UTC
onion_fl fidelity runs/fidelity_validation_emulated runs/fidelity_validation_sim \
  --calibration b55104b3… --metric time --metric bytes --metric macro_f1 --metric loss \
  --series global/swell --csv fidelity.csv --html fidelity.html > fidelity.json
```

All 60 runs finished at the first attempt. None is `estimated` or `outside_calibration`, every `run_hash` verified, and the harness raised no warning. The emulated runs took 20–30 s of wall clock each, with a container setup of about 3 s.

## Identity

| Identity | Value |
|---|---|
| `calibration_id` (cited by every sim run) | `b55104b3f71d684dedc413a000eec3a67eb1caba134cc9ff6a95648fb35c8dfc` |
| `infrastructure_id` | `ac43b5a3b0e4a1355e1b76032899cc673455795e46a1629a8386d7e38e59b940` |
| `data_id` | `7949a27b972ad85bf284ca213fcbfe77c610276f373755f866de29850c0bfc53` |
| `topology_id` cloud | `7033b5cb5738208aefa908a0b0290219175f9fad279a9c68b521d6c34754823e` |
| `topology_id` one fog (fog_a and fog_b) | `296d4fe28b43e41de00bcb98e5dfbe5e846c0f6258dfcb13c54fce61a8e6e8a4` |
| code | commit `1720dc8`, clean tree, for all 60 runs |
| image | `onionfl-a197b-emulated:latest` (`sha256:a826e66e…`) |

`config_id` per configuration, emulated / sim (they differ because `backend` is part of the config; `onion_fl fidelity` pairs on `design_id`):

| Configuration | emulated | sim |
|---|---|---|
| cloud-1ep | `073cc75f…` | `38c357a1…` |
| fog_a-1ep | `a638a5db…` | `1342b435…` |
| fog_b-1ep | `fa64d2ac…` | `923754b5…` |
| cloud-5ep | `abe817df…` | `4dc91667…` |
| fog_a-5ep | `ccefe624…` | `082b0768…` |
| fog_b-5ep | `bbc995bc…` | `e1317beb…` |

Each run's full ids, `run_id` and `run_hash` are in its `run.json`.

## Results against the pre-registration

### H1. Functional

| | Pre-registered | Result | |
|---|---|---|---|
| H1a | protocol events match in 30/30 pairs | 30/30 | **met** |
| H1b | bytes per link and round match exactly in 30/30 | 0/30; total bytes differ by 0–25 in 2.2–2.3 MB per run (≤ 1.1 × 10⁻⁵) | **not met** |
| H1c | final model within rtol 10⁻⁶, atol 10⁻⁷ in 30/30 | 14/30; largest difference 1.2 × 10⁻⁷ to 3.5 × 10⁻⁵ | **not met** |

- **H1b's premise was wrong.** The `npz` codec writes the arrays at a fixed size, but each message also carries a JSON header, and its floats (metrics, weights) are written with as many digits as they need. Torch on Windows and on Linux computes slightly different floats, so a few bytes differ on most links.
- **H1c.** The models were not identical bit for bit, as expected. In 16 pairs the difference grew past `test_runtime_equivalence`'s tolerance, up to 3.5 × 10⁻⁵ (seed 4, 5 epochs).
  - The difference depends only on the seed and the epochs: each backend gives the same model in its three topologies, so hierarchical FedAvg is exact.
  - The scores did not move: macro-F1 is identical in every pair, and the loss differs from the 7th decimal on.

### H2. Performance (relative error (B − A) / A, mean over 30 pairs, 95 % CI)

| Quantity | Emulated (A) | Sim (B) | Error | 95 % CI | Criterion | |
|---|---|---|---|---|---|---|
| Edge compute time per round (s) | 0.069 | 0.037 | −0.095 | [−0.334, +0.151] | point inside [−0.803, +4.076] | **met** (H2a) |
| Round time at the root (s) | 0.718 | 0.385 | −0.395 | [−0.421, −0.368] | \|e\| ≤ 0.25, CI within ±0.40 | **not met** (H2b) |
| Link latency (s) | 0.149 | 0.085 | −0.483 | [−0.494, −0.474] | \|e\| ≤ 0.25 | **not met** (H2c); the predicted direction holds |
| Run duration (s) | 13.2 | 4.1 | −0.694 | [−0.718, −0.671] | described only | — |

- **H2a is met only through a wide band.** The refit's prediction interval for the devices' training rate spans 3 348 to 86 256 samples/s, a factor of 26, so almost any compute error falls inside it.
  - The mean also hides two opposite errors. With 1 epoch the simulation is 75 % too fast (0.008 s against 0.065 s), with 5 epochs 56 % too slow (CI [+0.51, +0.61]).
  - The absolute relative error is 0.73 [0.70, 0.75].
  - A plausible reading, not measured here: an emulated training has a fixed cost of about 0.06 s per edge and round (the model's arrival, deserialising, the optimiser's setup) that a rate in samples per second does not model.
- **Round time and links: the simulation is 40–50 % too fast.** Each emulated link took 0.149 s from send to delivery against the simulation's 0.085 s. The extra 64 ms per message adds up to the round's −40 %.
  - In #195 (json codec, uncalibrated) the gap was 0.87 s against 0.12 s. Here, with calibrated links, 1 CPU per container and 5 kB messages, it is much smaller, but it is still there.
  - Its cause is not established. Candidates: Nagle on the MQTT connections (FL runs do not set `TCP_NODELAY`), processes handling one event at a time, and Python's per-message overhead in the containers.
  - By epochs: round time −0.454 [−0.479, −0.429] with 1 epoch and −0.336 [−0.351, −0.323] with 5; link latency −0.478 and −0.488.
- **Machine drift** (recorded before the runs, above). The compute error came out negative on average, not positive as a 1.6–2× faster machine would give. The drift is therefore not the main source of these errors.

### H3. Decision (K = 6, 5 seeds; ceiling from the seeds of A)

Means over seeds, emulated / sim:

| Configuration | time (s) | bytes | macro-F1 | loss |
|---|---|---|---|---|
| cloud-1ep | 10.06 / 2.10 | 2 241 206 / 2 241 205 | 0.4625 / 0.4625 | 0.5811 / 0.5811 |
| fog_a-1ep | 14.03 / 4.06 | 2 340 487 / 2 340 095 | 0.4625 / 0.4625 | 0.5811 / 0.5811 |
| fog_b-1ep | 15.80 / 4.93 | 2 341 272 / 2 340 095 | 0.4625 / 0.4625 | 0.5811 / 0.5811 |
| cloud-5ep | 10.55 / 2.91 | 2 241 424 / 2 241 429 | 0.4730 / 0.4730 | 0.6849 / 0.6849 |
| fog_a-5ep | 14.36 / 4.87 | 2 342 089 / 2 340 327 | 0.4730 / 0.4730 | 0.6849 / 0.6849 |
| fog_b-5ep | 14.51 / 5.74 | 2 340 323 / 2 340 327 | 0.4730 / 0.4730 | 0.6849 / 0.6849 |

| Metric | τ-b [95 % CI] | top-1 (P) | regret | Ceiling: top-1, regret | Criterion | |
|---|---|---|---|---|---|---|
| time | 0.867 [0.333, 0.867] | agrees, `cloud-1ep` (0.9995) | 0 s | 0.60, 0.121 s | τ-b ≥ 0.6, top-1, regret ≤ ceiling | **met** (H3a) |
| bytes | 0.645 [0.371, 0.964] | agrees (1.0) | 0 | 1.0, 0 | τ-b = 1, regret 0 | **not met** (H3b) |
| macro_f1 | 1.0 [1.0, 1.0]; rounded to 4 decimals 1.0 | agrees (1.0) | 0 | 0.60, 0.030 | rounded τ-b = 1 | **met** (H3c) |
| loss | 1.0 [1.0, 1.0]; rounded 1.0 | agrees (1.0) | 0 | 1.0, 0 | rounded τ-b = 1 | **met** (H3c) |

- **Time.** Both backends pick `cloud-1ep`, and within each number of epochs both order cloud < fog_a < fog_b (τ-b = 1 in each group; with 3 configurations its interval is [0.33, 1]).
  - The only inversion is across epochs: emulated, `fog_b-1ep` (15.8 s) is slower than `fog_b-5ep` (14.5 s), while the simulation has 1 epoch faster everywhere.
  - On this machine the emulated devices train so fast (0.07 s per edge and round) that the epochs barely change a run (10.06 against 10.55 s). The time is in the links, which the simulation underestimates by the same factor in every configuration. That is why its ranking survives errors of −40 % to −70 %.
- **Bytes.** The decision metric is the run's total from `summary.json`. Unlike H1b's per-link bytes in rounds, it also counts `hello` and acknowledgements, which a real child repeats until its parent answers. That adds 0 to 1.8 kB per emulated run on average and breaks the ties between the fog configurations, which send the same in the simulation, and τ-b ranks that noise. The decision itself is right: `cloud` sends 4 % fewer bytes, and both pick it, with zero regret.
- **Learning.** The three topologies tie exactly within each number of epochs in both backends. The epochs contrast is the same in both: 5 epochs give the higher macro-F1 and the higher loss. Only that contrast ranks anything (pre-registration H3c). macro-F1 is about 0.46–0.47 on average and varies by seed (0.58 for seed 0), so it is not a constant predictor, but the learning signal on SWELL physiology is weak.
- **Not available: the factorised predictor** (M5). This validation does not say whether the simulation decides better than rounds × an analytical time per round. With fixed rounds and no deadline, that predictor would also be decided by the link delays.

## Reading

- **Functionally**, the simulation reproduces the emulation's protocol in every pair.
  - Its models and bytes differ only by the two torch builds' floating point.
  - Two pre-registered criteria (exact bytes, the equivalence test's tolerance) were too strict for that; they are reported as not met, and not loosened after the fact.
- **In performance**, with the calibration closed, the simulation is still 40–50 % too fast in round time and link latency, and 69 % in a run's duration, which includes starting the processes.
  - The compute error is inside the calibration's band, but that band is wide, and the per-epoch errors (−75 %, +56 %) show a fixed per-training cost that the rate model misses.
  - So the calibrated simulation does not give representative times on this lab, with these small models.
- **In decisions**, it picks the same configuration as the emulation on every metric with zero regret, and it ranks the topologies within each number of epochs as the emulation does.
  - Its errors are mostly a common factor on the links.
  - With K = 6 and 5 seeds, this supports "the simulation ranks these designs as the emulation does". It does not support "its times are right".

## Deviations from the pre-registration

- §2 was edited before any run, as its steps 1–4 allow: the calibration id, the image and H2a's band.
- The INDEX records the machine's state before the runs (above). The hypotheses were not changed.
- The secondary per-epoch figures come from `onion_fl fidelity` run on the runs split by epochs (`--metric time`), not committed separately.
- No run was repeated, discarded or added.

## Files

- `runs_emulated/<run_id>/` and `runs_sim/<run_id>/`: `run.json`, `summary.json` and `model.npz` of each run.
  - `host` and `pid` are removed from `run.json`, and so is `config.paths`, which named this machine's folders. `verify_run` therefore cannot check these copies; it passed on the original folders when `onion_fl fidelity` read them.
  - `events.jsonl` (78 MB for the 60 runs) is left out.
- `fidelity.json`, `fidelity.csv` and `fidelity.html`: the output of the analysis command above.
- `PREREGISTRATION.md`.

## Follow-up, 2026-11-10: where the 64 ms went

Added after the fact; the results above are unchanged. [`followup_2026-11-10/`](followup_2026-11-10/INDEX.md) traced and sniffed each message's path and tested the causes one at a time in paired runs:

- **About two thirds: the receiving process's queue.** Four edges share one process, so a model waits there while its neighbours train. The emulation counts that wait as link latency; the simulation counts it as compute.
- **About a quarter: Nagle.** It delayed back-to-back updates and control messages by 11–50 ms. It is now off on the MQTT clients and the emulated broker (`5ac0133`).
- **The same-host links,** which the simulation makes free, alone account for −0.19 of H2c.

Without Nagle and without the queue, the simulated links are within −3.9 % [−8.4 %, +0.4 %] of the emulated ones. A reduced rerun (2 designs × 5 seeds, on a busy machine) moves H2c from −0.562 to −0.447 and the uplinks from −0.316 to −0.076. The pre-registered criteria are not re-judged.
