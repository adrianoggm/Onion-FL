# The distributed continuum on containers acting as machines (new framework, real data)

Issue #164 (continuum C9). Run on 2026-11-09 with SWELL-KW and WESAD, on **five Docker containers acting as machines** on one Docker Desktop VM (Windows 11 host). **No real hardware was used**: no Raspberry Pi, no physical server, no second computer. The acceptance on real machines stays pending (`docs/distributed-hardware.md`).

```bash
python scripts/continuum_distributed.py up --prefix onionfl-a164 --image onionfl-a164-distributed:latest --into DIR
python scripts/continuum_distributed.py run experiments/continuum_distributed.yaml \
    --inventory DIR/inventory.yaml --runs runs/<mode> --seed <0|1|2> --mode <sim|distributed|faults>
python scripts/continuum_distributed.py times runs/*/*/ > times.csv
python scripts/continuum_distributed.py resumed runs/*/*/ > resumed.csv
python scripts/continuum_benchmark.py metrics runs/*/*/ --subjects test --curves curves.csv > run_metrics.csv
python scripts/continuum_distributed.py down --prefix onionfl-a164
```

**In one paragraph.** A C8 scenario (warm start behind `champion_challenger`, a `schedule` trigger every 10 data minutes, speed 60) ran on the machines with every edge saving its part of the bundle on its own machine after each event. Without failures it gives the simulation's 18 rounds, and every edge trains exactly the rows it trains in the simulation. It sends 3% fewer bytes and its prequential score on SWELL is 0.02 lower, because in round 1 some edges miss the fogs' 6 s deadline (their first training in a fresh process is slow). With real failures (the process of `field-1`, 13 SWELL edges, killed with `kill -KILL` at 60 s and started again 15 s later, and the 5 edges of `field-3` joining at 90 s), every run finished its 18 rounds. All 39 restarted edges came back from their saved state, and every edge of every run trained exactly the labelled rows it trains in the simulation: **no row was lost or trained twice across a restart**. The failures cost rounds without those edges, fewer trainings and bytes, and prequential samples that were never scored. The final model is the initial one in every run (simulated, distributed or with failures), because the promotion rejects every challenger, as in C8. So the final and AUC scores agree by construction and say nothing about fidelity.

## Setup

- **Scenario** (`experiments/continuum_distributed.yaml`). C8's `continuum_benchmark_promotion` warm baseline: SWELL + WESAD streams with a 20-minute bootstrap, a tenth of the rows labelled 10 minutes late, `modular_mlp` 64 / [64, 32], SGD lr 0.1 and 10 local epochs, `champion_challenger` (macro-F1, `min_samples: 20`), scored on the test subjects. It differs from C8 in three ways:
  - **Topology.** `four_fogs_swell_wesad_distributed`: root deadline 9 s, fog deadline 6 s (a round every 10 s of wall clock) and a fog `register_timeout` of 20 s, so a fog closes a round without its down edges and starts without the late ones;
  - **Infrastructure.** `continuum_containers`: `cloud-1` (root, test evaluators, validators, broker), `fog-1` (the four fogs and the validation evaluators), `field-1` (13 SWELL edges), `field-2` (10 WESAD edges) and `field-3` (the 5 SWELL edges `swell-20…24`). Its network (1 ms, 1 Gbps) and `train_rate` (2000, as C8's edge device) are not calibrated: they only shape the simulation's times.
  - **Because of these changes**, the simulated runs here are not C8's runs (other `config_id`), and C8's numbers are not compared with these.
- **Three modes per seed** (seeds 0, 1, 2):
  - `sim`: the simulation of the same config, the reference;
  - `distributed`: the backend of #199 over SSH, with `checkpoints: true` (§7d of `docs/architecture.md`) and `start_in: 60` (the run's clock starts 60 s after the launch, so every node is up before the streams start);
  - `faults`: the same with `restarts: [{host: field-1, at: 60, down: 15}]` and `joins: {field-3: 90}`.
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

Every row of every CSV carries its `run_id`, and `times.csv` and `run_metrics.csv` carry `config_id`, `topology_id` and `run_hash`. The nine run folders keep `run.json` and `summary.json`; the event files (4–8 MB each) and models are not committed. The `run.json` of each run records the launcher's scratch paths (`paths.runs`, the inventory file), which are signed and kept as they are.

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

(`run_metrics.csv`, `times.csv`.) Every run has 18 rounds, 2890 sample-epochs and no promotion, and its final model scores 0.344 on SWELL's test subjects and 0.521 on WESAD's (the initial model: no challenger was promoted).

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

## What this does and doesn't show

- **Shows.** On machines reached only through SSH, a continuum with promotion runs to its end, keeps each edge's state on its machine, and resumes it after a real `kill -KILL`, with no row lost or repeated. Edges join mid-continuum.
- **Doesn't show.**
  - Anything about real hardware, a real network or real clocks: the five "machines" share one VM, its clock and its CPU, which another experiment was using at the same time.
  - The model's fidelity. The champion never changes, so the final models are equal by construction. The edges' own models train, but are not compared.
  - Restarting a fog or the root: only hosts of edges may restart.
- **The simulation does not model the failures.** The fault runs are compared with the failure-free simulation, to measure what the failures cost.
- **Round 1's late updates** come from the 6 s deadline against a cold first training. A longer round-1 deadline, or a warm-up, would remove them; it was not tuned.
