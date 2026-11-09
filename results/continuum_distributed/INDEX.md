# The distributed continuum on containers acting as machines (new framework, real data)

Issue #164 (continuum C9). Run on 2026-11-09 with SWELL-KW and WESAD, on **five Docker containers acting as machines** on one Docker Desktop VM (Windows 11 host). **No real hardware was used**: no Raspberry Pi, no physical server, no second computer. The acceptance on real machines stays pending (`docs/distributed-hardware.md`).

```bash
python scripts/continuum_distributed.py up --prefix onionfl-a164 --image onionfl-a164-distributed:latest --into DIR
python scripts/continuum_distributed.py run experiments/continuum_distributed.yaml \
    --inventory DIR/inventory.yaml --runs runs/<mode> --seed <0|1|2> --mode <sim|distributed|faults>
python scripts/continuum_distributed.py run experiments/continuum_distributed.yaml \
    --runs runs/sim-faults --seed <0|1|2> --mode sim-faults   # no machines needed
python scripts/continuum_distributed.py times runs/*/*/ > times.csv
python scripts/continuum_distributed.py resumed runs/*/*/ > resumed.csv
python scripts/continuum_benchmark.py metrics runs/*/*/ --subjects test --curves curves.csv > run_metrics.csv
python scripts/continuum_distributed.py down --prefix onionfl-a164
onion_fl fidelity runs/faults runs/sim-faults --metric bytes --csv fidelity_faults.csv > fidelity_faults.json
```

**In one paragraph.** A C8 scenario (warm start behind `champion_challenger`, a `schedule` trigger every 10 data minutes, speed 60) ran on the machines with every edge saving its part of the bundle on its own machine after each event. Without failures it gives the simulation's 18 rounds, and every edge trains exactly the rows it trains in the simulation. It sends 3% fewer bytes and its prequential score on SWELL is 0.02 lower, because in round 1 some edges miss the fogs' 6 s deadline (their first training in a fresh process is slow). With real failures (the process of `field-1`, 13 SWELL edges, killed with `kill -KILL` at 60 s and started again 15 s later, and the 5 edges of `field-3` joining at 90 s), every run finished its 18 rounds. All 39 restarted edges came back from their saved state, and every edge of every run trained exactly the labelled rows it trains in the simulation: **no row was lost or trained twice across a restart**. The failures cost rounds without those edges, fewer trainings and bytes, and prequential samples that were never scored. The final model is the initial one in every run (simulated, distributed or with failures), because the promotion rejects every challenger, as in C8. So the final and AUC scores agree by construction and say nothing about fidelity.

**The same failures in simulation** (added afterwards, commit `bbe0c9a`). The `sim` backend now takes the same `start_in`, `checkpoints`, `restarts` and `joins`, so the failure scenario also runs in simulation (`sim-faults`). Every edge trains exactly the labelled rows it trains on the machines, and the restarted edges come back with the same consumed rows in 87% of the cases. The simulation restarts a process at once, while a real one took 12–16 s to rebuild its scenario before its edges were back. So in simulation the restarted and late edges take part in rounds the real ones missed: 3 failed quorums against 6–13, 143–150 trainings against 127–138, and 181 MB against 143–172. The prequential SWELL score is 0.032 and 0.017 lower in simulation in seeds 0 and 1, and 0.008 higher in seed 2. In seed 2 the simulation promoted one challenger that the machines never promoted, so its final model differs.

## Setup

- **Scenario** (`experiments/continuum_distributed.yaml`). C8's `continuum_benchmark_promotion` warm baseline: SWELL + WESAD streams with a 20-minute bootstrap, a tenth of the rows labelled 10 minutes late, `modular_mlp` 64 / [64, 32], SGD lr 0.1 and 10 local epochs, `champion_challenger` (macro-F1, `min_samples: 20`), scored on the test subjects. It differs from C8 in three ways:
  - **Topology.** `four_fogs_swell_wesad_distributed`: root deadline 9 s, fog deadline 6 s (a round every 10 s of wall clock) and a fog `register_timeout` of 20 s, so a fog closes a round without its down edges and starts without the late ones;
  - **Infrastructure.** `continuum_containers`: `cloud-1` (root, test evaluators, validators, broker), `fog-1` (the four fogs and the validation evaluators), `field-1` (13 SWELL edges), `field-2` (10 WESAD edges) and `field-3` (the 5 SWELL edges `swell-20…24`). Its network (1 ms, 1 Gbps) and `train_rate` (2000, as C8's edge device) are not calibrated: they only shape the simulation's times.
  - **Because of these changes**, the simulated runs here are not C8's runs (other `config_id`), and C8's numbers are not compared with these.
- **Four modes per seed** (seeds 0, 1, 2):
  - `sim`: the simulation of the same config, the reference;
  - `distributed`: the backend of #199 over SSH, with `checkpoints: true` (§7d of `docs/architecture.md`) and `start_in: 60` (the run's clock starts 60 s after the launch, so every node is up before the streams start);
  - `faults`: the same with `restarts: [{host: field-1, at: 60, down: 15}]` and `joins: {field-3: 90}`;
  - `sim-faults`: the simulation of `faults`, with the same `start_in`, `checkpoints`, `restarts` and `joins` on the `sim` backend (§7d of `docs/architecture.md`). Run later, on commit `bbe0c9a` (clean tree).
- **Machines.** The `onionfl-a164-distributed` image (`src/onion_fl/infrastructure/distributed.Dockerfile` on the emulated host image, with the launcher's numeric versions), this checkout mounted read-only, reached only through SSH with a throwaway key. The broker runs in `cloud-1`'s network namespace. The preflight measured clock offsets under 0.016 s (`distributed.machines` in each `run.json`). The script's `max_offset` is 1 s because, right after Docker Desktop restarted, the VM was 0.73 s off; every container shares the VM's clock.

| Identity | Value |
|---|---|
| code | commit `ab5e66f`, clean tree (`code_version` in every `run.json`) |
| `topology_id` | `2a33649e5785c6bf467c30d51ca2ed6015706a3df6b17a6b1268584c7532352f` |
| `data_id` | `9fd573754157d68b111f037ecd891bdb62abcbff1d5e5e818da789dde470257c` (C8's) |
| `infrastructure_id` | `6a0ab6bd2dbca08f…` (in each `run.json`) |
| `config_id`, sim | `80246db9339d19adc53b704217002e821e4cf9676b9394c65a0aa5a023954e34` |
| `config_id`, distributed | `63502190c22465c5d4ea0f0f2d4c6ab82e2ed7006b0ff61af1ea71b60afabc41` |
| `config_id`, faults | `fb04a4d3a3a526efccdfeeb97aa76a37341e62f5f1143ea0fecacbc6d2250051` |
| `config_id`, sim-faults | `dd93eaca8e8c447c120477471404b0abada1e1b837708a42eac87f770be73b77` (code `bbe0c9a`) |

Every row of every CSV carries its `run_id`, and `times.csv` and `run_metrics.csv` carry `config_id`, `topology_id` and `run_hash`. The twelve run folders keep `run.json` and `summary.json`; the event files (4–8 MB each) and models are not committed. The `run.json` of each run records the launcher's scratch paths (`paths.runs`, the inventory file), which are signed and kept as they are.

## Results (3 seeds each)

Per seed, SWELL (WESAD's prequential is 0.056 in every mode and seed: its stream is stress-only after t₀, as in C8):

| Seed | Mode | Prequential macro-F1 | Rows scored | Trainings | Rejected | Messages | MB sent | Late updates | Wall s |
|---|---|---|---|---|---|---|---|---|---|
| 0 | sim | 0.409 | 1959 | 196 | 18 | 2362 | 214.8 | 0 | 32 |
| 0 | distributed | 0.387 | 1859 | 185 | 17 | 2426 | 208.5 | 16 | 238 |
| 0 | faults | 0.381 | 1507 | 138 | 12 | 2076 | 172.2 | 26 | 240 |
| 1 | sim | 0.439 | 1959 | 194 | 18 | 2366 | 215.1 | 0 | 38 |
| 1 | distributed | 0.422 | 1874 | 185 | 17 | 2486 | 208.4 | 15 | 246 |
| 1 | faults | 0.408 | 1518 | 127 | 5 | 1873 | 143.0 | 28 | 244 |
| 2 | sim | 0.268 | 1959 | 190 | 18 | 2360 | 213.6 | 0 | 30 |
| 2 | distributed | 0.247 | 1870 | 187 | 17 | 2456 | 208.5 | 14 | 244 |
| 2 | faults | 0.233 | 1522 | 138 | 12 | 2108 | 171.5 | 23 | 244 |
| 0 | sim-faults | 0.349 | 1506 | 146 | 15 | 2207 | 180.8 | 0 | 32 |
| 1 | sim-faults | 0.391 | 1506 | 143 | 15 | 2211 | 181.0 | 0 | 29 |
| 2 | sim-faults | 0.241 | 1506 | 150 | 14 (1 promoted) | 2203 | 180.8 | 0 | 31 |

(`run_metrics.csv`, `times.csv`.) Every run has 18 rounds and 2890 sample-epochs. Every run but one has no promotion, and its final model scores 0.344 on SWELL's test subjects and 0.521 on WESAD's (the initial model: no challenger was promoted). The exception is `sim-faults` seed 2, which promoted one challenger: its final model scores 0.351 on SWELL and 0.532 on WESAD, against 0.264 and 0.392 for the initial model of that seed.

**Against the simulation, without failures.**

- **Rounds and data.** The same 18 rounds. Every edge trains the same labelled rows as in the simulation: 28 edges × 3 seeds, compared one by one in `resumed.csv` (`examples`).
- **Round 1 is where they differ.** In each distributed run, three fogs and the root fail round 1's quorum: 14–16 edges answer after the 6 s deadline, so their first training (v0 on the history, in a fresh process) is dropped. Those edges train the same rows in round 2. Because of this, the distributed runs make 6–11 fewer trainings, send 3% fewer bytes, and lose one candidate version (17 rejections, not 18).
- **Rows scored.** 85–100 fewer SWELL arrivals are scored prequentially, and the score is 0.017–0.022 lower in each seed. The rows of round 1's late edges are predicted by the model they then served, and the last seconds of each run end at a slightly different time on the wall clock (`t_end` 164–172 s against 170 s).
- **Times.** The simulation's root rounds last 0.07 s (its network and compute are a guess). On the containers they last 2.1 s on average and up to 5.6 s. A run takes 240 s of wall clock (60 s of start, 170 s of streams) against 30–38 s for the simulation.

**With failures.**

- **Every run finished** with its 18 rounds and exit code 0 on every host.
- **Restarts.** `field-1` was killed at 60.0 s and started again at 75.0 s. Its 13 edges came back from their saved state at 86.9–90.5 s, after rebuilding the scenario. All 39 `node.resumed` events say `restored`, and each edge had already consumed 2–7 rows and predicted 32–48 when it came back.
- **No row lost or trained twice.** Every edge, restarted or late, trained exactly the labelled rows it trains in the simulation. An edge restarted without its state would train its rows again: `tests/test_continuum_distributed.py` checks both, and that an edge restarted from its saved part ends bit for bit as one that never stopped.
- **Late edges.** The five edges of `field-3` started at 90 s and first trained at 114–117 s, on their backlog: 1–6 trainings, against 5–11 in the simulation. Their rows from before they joined were never predicted, since they had no model then. So they score 7–42 rows prequentially, against 91–114.
- **Cost of the failures.** 6–13 failed quorums (the rounds while `field-1` was down, closed by the 6 s deadline without its edges) and 127–138 trainings against 190–196. 1873–2108 messages and 143–172 MB, a fifth to a third less. 5–12 candidate versions, every one rejected. Prequential SWELL is 0.006–0.014 below the run without failures.

**Against the simulation of the same failures** (`sim-faults` against `faults`, the same 3 seeds; `fidelity_faults.json`, `resumed.csv`).

- **Rows.** Every edge (28 × 3 seeds) trains exactly the labelled rows it trains on the machines, and in the failure-free simulation. When they come back, the 39 restarted edges have consumed the same rows as on the machines in 34 cases out of 39. They had predicted 21–27 rows, against 32–48 on the machines, because their saved state is from their last handled event and the real processes' timers fall at other seconds.
- **Start-up is not modelled.** In the simulation a process is up at once: `field-1`'s edges come back at 75.0 s, and `field-3`'s say hello at 90.0 s and train in that round. On the machines they came back at 86.9–90.5 s and said hello at 103.0–104.6 s: 12–16 s to rebuild the scenario. The late edges first trained at 90.0 s in simulation against 114–117 s, and scored 30–48 rows against 7–42.
- **Rounds and traffic.** So in simulation the down edges miss fewer rounds: 3 failed quorums against 6–13, 143–150 trainings against 127–138, 2203–2211 messages against 1873–2108, and 180.8–181.0 MB against 143.0–172.2. No update is late in simulation; 23–28 were late on the machines (round 1's cold start, as without failures).
- **Scores.** Prequential SWELL macro-F1 is 0.349, 0.391 and 0.241, against 0.381, 0.408 and 0.233 on the machines (mean 0.327 against 0.341; 0.372 for the failure-free simulation). Both score about the same number of rows (1506 against 1507–1522).
- **`onion_fl fidelity`** (A = `faults`, B = `sim-faults`, `--metric bytes`). Final model: equal bit for bit in seeds 0 and 1 (the initial one), different in seed 2 (maximum difference 0.021), where the simulation promoted a challenger. Protocol events differ at the root and the four fogs in every seed (the failed quorums). Bytes differ in every pair (+5% to +27% in simulation). Root rounds last 0.07 s in simulation against 1.96 s (relative error −0.96), edge compute −0.85, link latency +75 (the network and `train_rate` are not calibrated, as said above); the run's duration −0.5%. The decision on bytes is trivial: one configuration, 3 seeds (fewer than the 5 asked for decision fidelity).
- **What it means.** The simulation reproduces what the failures do to the data (no row lost or repeated) but not their cost in rounds: a restart in simulation is optimistic by the time a process takes to start. Modelling that time would need a separate calibration (docs/RULES.md §3), not one fitted on these runs.

## What this does and doesn't show

- **Shows.** On machines reached only through SSH, a continuum with promotion runs to its end, keeps each edge's state on its machine, and resumes it after a real `kill -KILL`, with no row lost or repeated. Edges join mid-continuum.
- **Doesn't show.**
  - Anything about real hardware, a real network or real clocks: the five "machines" share one VM, its clock and its CPU, which another experiment was using at the same time.
  - The model's fidelity. The champion never changes, so the final models are equal by construction. The edges' own models train, but are not compared.
  - Restarting a fog or the root: only hosts of edges may restart.
- **The simulation models the failures, not the start-up time.** The fault runs are compared with the failure-free simulation, to measure what the failures cost, and with the simulation of the same failures (`sim-faults`), which is optimistic by the 12–16 s a real process takes to come back.
- **Round 1's late updates** come from the 6 s deadline against a cold first training. A longer round-1 deadline, or a warm-up, would remove them; it was not tuned.
