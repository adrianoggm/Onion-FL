# Degradation × topology: pre-registration

Issue #290 (E3). This file is committed **before the main run**. The INDEX of `results/degradation_topology/` will cite its commit. Only the number of seeds (§6) is filled in after the pilot, in a separate commit, before any main run. Any other change after a main-run result is seen needs a new pre-registration, dated and committed, and the INDEX names both.

## 1. Question

Does the choice of topology (flat, two levels, three levels) change, in a statistically significant way, how federated learning and the system degrade under link loss, latency, edge dropout, a fog crash, a WAN partition and the quorum?

Today the four architectures of `results/infrastructure_architectures/` tie on learning under nominal conditions (5 seeds). The interaction with degradation was never measured.

## 2. What is fixed

Everything runs in `sim`. The experiments are `experiments/degradation_topology_wesad.yaml` and `experiments/degradation_topology_cifar10.yaml`. The infrastructures are `infrastructures/degradation_lab*.yaml`. All are committed with this file.

- **Physical continuum** (`degradation_lab`, the same machines in every arm). Two zones. Each zone is a field of Raspberry Pi devices (900 samples/s) behind an access link to its zone's servers (`gw-x`, `fog-x1`, `fog-x2`). Each zone reaches the cloud (`cloud-1`) over its own WAN link (70 ms, 10 Mbps). The nominal access link has the numbers of the runtime's `lan` preset (1 ms ± 0.2 ms, 1 Gbps). Bandwidth is shared. The numbers are illustrative, not calibrated (I12, #197).
- **Zones.** Each edge sits on its own device in its zone's field, in every topology, by an explicit assignment. The membership is fixed by subject, so it is the same in every arm and every seed.
  - WESAD: the roles of `infrastructure_architectures` (`roles.seed` 0). Test subjects S6, S7, S10. Training subjects: zone A S2, S3, S5, S8, S9; zone B S13–S17. The two val subjects are zone evaluators: S4 in zone A, S11 in zone B.
  - CIFAR-10: `cifar10_cnn`'s 10 synthetic clients (Dir(0.5), `partition_seed` 0; RULES §1). Zone A clients 0–4, zone B clients 5–9. The global test is the official test batch.
- **Topology factor** (3 levels). The leaf level is always `fog`, so one config path reaches the leaves of every topology.
  - *flat*: one aggregator, `hub`, on `cloud-1`. Every edge talks to the cloud across its zone's WAN link. The coordinator takes evaluators only, so the star is the hub, as in `pilot_deadline_coupling`. Hub and coordinator share a host: their link is free.
  - *two* (the **reference**): one fog per zone, `fog_a` on `fog-a1` and `fog_b` on `fog-b1`. This is the canonical edge–fog–cloud tree (architecture B of `infrastructure_architectures`).
  - *three*: a region per zone (`region_a` on `gw-a`, `region_b` on `gw-b`), each with two leaf fogs (`fog-x1`, `fog-x2`). WESAD: fog_a1 S2, S3, S5; fog_a2 S8, S9; fog_b1 S13–S15; fog_b2 S16, S17. CIFAR-10: fog_x1 the first three clients of the zone, fog_x2 the last two.
- **Protocol constants**, the same in every arm. FedAvg at every level, server optimizer `replace`. Quorum 0.5 at every aggregator, so a round goes on without a cut zone. Deadlines: 5 s at the leaves, 8 s at the regions and 12 s at the root on WESAD. The slowest nominal round, 4g in flat, took 1.1 s in the design check (§9), so the leaf deadline is about 4× it. On CIFAR-10 every time is ×100: the ratio of the nominal round times in two levels, 73 s against 0.74 s.
- **Learning**, not tuned for this study.
  - WESAD: `infrastructure_architectures`' modular MLP and SGD trainer. Its lr 0.1 was chosen there on validation subjects only. 10 local epochs, batch 32, 20 rounds.
  - CIFAR-10: `cifar10_cnn`'s NIID-Bench CNN and trainer (SGD, lr 0.01, momentum 0.9, 10 local epochs, batch 64), with 20 rounds instead of 50 (that run peaked by round 14).
- **Codec** `npz` on every link.

### Degradation factor (9 levels)

| Level | What changes | Mechanism |
|---|---|---|
| `nominal` | nothing | `degradation_lab` |
| `loss5` | 5 % loss on every physical link (access and WAN), both directions | `degradation_lab_loss5` |
| `loss20` | 20 % loss, the same way | `degradation_lab_loss20` |
| `4g` | the access links take the `4g` preset's latency, jitter and bandwidths (50 ms, lognormal 15 ms, 5/20 Mbps), without its 1 % loss | `degradation_lab_4g` |
| `dropout` | every edge is offline in a round with probability 0.2 | `edge.availability: bernoulli(0.2)` |
| `dropout_k1` | `dropout` with leaf quorum K = 1.0 instead of 0.5 | `fog.defaults.quorum: 1.0` |
| `crash` | host `fog-a1` crashes at 10 s (CIFAR-10: 1000 s) and never comes back; edges move to the nearest live leaf after 5 s (500 s) | `degradation_lab_crash[_cifar10]`, `failover: nearest_alive` |
| `partition` | zone A's WAN link is down from 10 s to 60 s (CIFAR-10: 1000–6000 s) | `degradation_lab_partition[_cifar10]` |
| `partition_autonomy` | `partition`, with `autonomy: {after: 15s}` (1500 s) on the leaf fogs (#289) | `fog.defaults.autonomy` |

Choices and what they cost:

- **Latency is one level, not two.** `lan` is the nominal access link, so the issue's `lan`/`4g` pair is `nominal` against `4g`.
- **The `4g` level has no loss**, so that latency and loss are separate factors.
- **Loss hits every physical link.** An update that crosses two links (an edge's in the flat topology) is lost with 1 − (1 − p)². That is the physical consequence of the topology and part of what is measured.
- **The quorum is a protocol knob, not a degradation.** It is crossed with dropout only (`dropout_k1`), where it decides whether a round survives a missing edge. Crossing it with every level would double the design.
- **Fog crash in the flat topology.** `fog-a1` carries nothing in flat, so `crash` equals `nominal` there by construction (the design check gave the same model bit for bit). It stays in the design: "flat has no fog to lose" is part of the answer, and the interaction contrasts need the cell.
- **Fog crash in three levels hits a leaf fog**, not a region: failover of non-leaf aggregators does not exist (architecture §7a, "Not yet"). `crash_at` as a node's own availability does not trigger a failover, which reacts to hosts. So the crash is the host's.
- **Autonomy sits on the leaf fogs** in every topology. In two levels the leaf is the cut fog itself. In three levels the cut region has no autonomy, and its two leaf fogs go on separately. In flat the hub is never cut, so autonomy cannot act: there `partition_autonomy` equals `partition`. Putting autonomy on the regions too would need a path that only three levels has.
- **Times are wall-clock times of the simulation, not rounds.** The crash and the partition fall at different rounds in different topologies, as a real failure would.

So there are **27 cells** (3 × 9) per dataset. `onion_fl plan` validates both files: 27 distinct `config_id`s each.

## 3. Hypotheses

All tests are two-sided. The reference topology is *two*. The contrasts are *flat − two* and *three − two*.

- **H1, the topology effect at each degradation level.** For each dataset, degradation level and metric: is *flat − two* (and *three − two*) different from 0? Is it equivalent to 0 within the margin?
- **H2, the interaction.** For each dataset and metric: does the degradation slope depend on the topology? Per seed, the difference-in-differences is

  `DiD(t, d) = [m(t, d) − m(t, nominal)] − [m(two, d) − m(two, nominal)]`, for t ∈ {flat, three} and the 8 degraded levels d.

- **H0 of equivalence at `nominal`.** FedAvg weighted by samples at every level gives the flat average (a protocol test checks it with the stub trainer). So the learning metrics should be equivalent across topologies at `nominal`. The design check gave the same model bit for bit in the three topologies.
- **Expected directions.** These are stated for reading, not used by the tests. They are partly informed by the design check of seed 0 (§9), which was seen.
  - Under loss, flat loses more updates per round, because its updates cross two links.
  - Under `dropout_k1`, flat fails most rounds: all 10 edges are present with probability 0.8¹⁰ ≈ 0.11. Three levels fails least, with 2–3 edges per leaf.
  - Under `crash`, two levels pays a failover across the WAN, three levels a local one, and flat nothing.
  - Autonomy can only help the hierarchical topologies.

## 4. Metrics

All per run.

**Primary**

1. **Final global macro-F1** at round 20, from the coordinator's test evaluators. WESAD: the three test subjects (172 windows). CIFAR-10: the official test batch.
2. **Time to target.** The first `t_virtual` at which the global macro-F1 reaches the target, fixed now: **0.70 on WESAD** and **0.45 on CIFAR-10**.
   - WESAD's nominal runs plateau at 0.75 by round 3.
   - `cifar10_cnn` reached 0.498 at round 50 and peaked at 0.51.
   - A run that never reaches the target is censored at its `run.finished` instant and counts there, so the mean is a lower bound.
   - It is analysed as `ln(time / reference time)` per seed.

**Secondary**

3. Final global accuracy.
4. **Lost rounds.** The root's rounds that ended in `round.quorum_failed`. The count of `round.quorum_failed` over every aggregator is reported too.
5. **WAN bytes.** The bytes of every `message.sent` between a node on `cloud-1` and a node on another site: the traffic over the zone–cloud links. The hub sits on `cloud-1`, so in flat every edge message counts. Evaluators sit on their parent's host.
6. **Per-zone fairness**, WESAD only (CIFAR-10's clients have no `local_val` rows). At round 20, the mean macro-F1 of the received model on each edge's `local_val`, per zone. Reported: the gap |A − B| and the worst zone. Zone A is the one the crash and the partition hit.
7. **Run duration.** `t_virtual` of `run.finished`. It is not the last event: a scheduled `link.up` can come after the end.

Equivalence margins (the largest irrelevant difference), fixed now:

| Metric | Margin | Why |
|---|---|---|
| macro-F1, accuracy, fairness gap | ± 0.02 absolute | On WESAD, 0.02 is about 3 of the 172 test windows. A smaller change is not interpretable with three test subjects. The 95 % CI half-width over 5 seeds in `infrastructure_architectures` was 0.012–0.013 on WESAD. On CIFAR-10 two points sit inside the spread NIID-Bench reports for one algorithm across runs. |
| time to target, run duration, WAN bytes | ratio within [1/1.1, 1.1], i.e. ± ln 1.1 on the log scale | Below 10 % a time or traffic difference changes no deployment decision in an illustrative, uncalibrated infrastructure. |
| lost rounds | ± 1 round (of 20) | One lost round is the smallest unit. |

## 5. Analysis

Each dataset is analysed separately. α = 0.05. The pairing unit is the seed: all 27 cells share each seed, hence the same test subjects, partition, initial model and edge availability draws (`bernoulli` draws per edge id, seed and round).

- **H1.** For each degradation level `d` and metric, the per-seed differences against *two* at the same `d`. Reported:
  - the mean Δ with a seeded 95 % paired-bootstrap CI;
  - `sd_diff` and Cohen's `d_z`;
  - `p_perm`, the sign-flip permutation p (exact up to 13 pairs);
  - `p_adj`, Holm over the **2 contrasts** of that (dataset, `d`, metric);
  - `p_tost` with the margin of §4;
  - the verdict: `equivalent` if TOST < α, else `different` if `p_adj` < α, else `inconclusive`.

  On the learning metrics this is `Runs.compare(level="global", metric=..., by=("topology_id",), paired=True, reference=<two's topology_id>, margin=...)` on the runs of one degradation level, at round 20. That is also `onion_fl report --reference --margin`. The run-level metrics (time to target, lost rounds, WAN bytes, fairness, duration) use the same functions of `onion_fl.observability.stats` (`paired_test`, `holm`, `verdict`) on one value per run, in the study's summary script. `Runs.compare` refuses `paired` with `target` and works per round.
- **H2.** The per-seed `DiD(t, d)` of §3, tested the same way (bootstrap CI, `p_perm`, TOST with the same margin). Holm runs over the **16 contrasts** (2 topologies × 8 levels) of each (dataset, metric).
- **Families.** There is no correction across datasets or metrics. Each of the two primary metrics answers its own question. The secondary metrics use the same machinery and are labelled secondary: none of them can turn a primary answer around.
- **What "no difference" means.** Only the `equivalent` verdict: TOST below α with the margin of §4.
  - An `inconclusive` contrast is reported as "cannot tell", never as "no difference".
  - A contrast that is significant and also inside the margin is `equivalent` (the repository's `verdict` gives TOST precedence). The INDEX says when `p_adj` < α too: detectable but irrelevant.
- **Censoring.** A censored time to target enters at its run's duration. The INDEX states how many runs per cell are censored. If more than half of a cell's runs are censored, its time-to-target contrasts are reported but flagged as lower bounds.

## 6. Seeds

- **Pilot:** seeds 0, 1 and 2 on WESAD, all 27 cells. On CIFAR-10, the cells and seeds of §8.
- **Rule**, decided now:
  - From the pilot's H1 contrasts (*flat − two* and *three − two*, at every degradation level) on the two primary metrics, take each `sd_diff` (final macro-F1; `ln` time to target).
  - For each, compute `seeds_needed(delta=margin, sd=sd_diff)` (paired t-test, 80 % power, α = 0.05): Δ = 0.02 for macro-F1, ln 1.1 for time. A contrast with `sd_diff` = 0 needs 2.
  - n = the largest of them, clamped to [10, 30]. 30 is this machine's budget.
  - If the largest exceeds 30, the cells whose power at n = 30 (`paired_power`) is under 0.8 are listed in the INDEX as underpowered. Their `inconclusive` verdicts are no evidence of anything.
  - H2 is not powered by this rule. Its power at n is reported.
- **Main-run seeds:** 100 … 100 + n − 1. They are disjoint from the pilot and the design check, whose runs never enter the confirmatory analysis.
- **n (WESAD) = 30**, seeds 100–129. The pilot (`results/degradation_topology/pilot/wesad/`, 81 runs at `ee75386`, clean tree) gave a largest `seeds_needed` of 1205 (time to target, `loss5`, *flat − two*), so n is clamped to 30. The contrasts underpowered at n = 30 (power under 0.8 from the pilot's `sd_diff`), whose `inconclusive` verdicts are no evidence of anything:
  - macro-F1: `dropout_k1` *flat − two* (0.08), `partition` *flat − two* (0.63);
  - time to target: `loss5` *flat − two* (0.07) and *three − two* (0.17), `loss20` *three − two* (0.27), `dropout` *flat − two* (0.24) and *three − two* (0.22), `dropout_k1` *three − two* (0.51).
- **n (CIFAR-10) = 0: no main run** (§8).

## 7. Exclusions

- A run that is not `finished`, or whose `run_hash` does not verify, is rerun once with the same seed and code.
  - If it fails again, that seed is dropped from **every** cell of its dataset, because the pairs need all cells.
  - The INDEX lists every rerun and every dropped seed.
- No run is excluded for its result.
- All runs of a dataset come from one commit with a clean tree. If the code has to change mid-run, everything after the change is rerun.

## 8. CIFAR-10 scope

A CIFAR-10 run takes about 14 min of wall time on this machine (§9), against about 15 s for WESAD. The full 27 cells × n seeds is not affordable here, so CIFAR-10 runs a reduced design.

- **Proposed:** the 3 topologies × 5 levels `nominal`, `loss20`, `dropout_k1`, `crash` and `partition_autonomy`, i.e. one level per mechanism at its strongest, with autonomy.
- **Cost of the cut:**
  - the loss and latency slopes on CIFAR-10 have one point (no `loss5`, no `4g`);
  - `dropout` without the strict quorum is missing;
  - the autonomy effect cannot be separated from the partition's.
- WESAD keeps all 27 cells and answers those parts alone.
- The user confirms the CIFAR-10 cells and n before the main run, and §6 records them.
- **Decision (2026-11-16, before the main run):** CIFAR-10 has no main run. The pilot's CIFAR-10 runs took about 23 min each with six in parallel. The seed rule gives n = 30, so the reduced design would be 450 runs, about 29 h on this machine. The study's confirmatory answer is WESAD only. The 6 CIFAR-10 pilot runs (`loss20`, *flat* and *two*, seeds 0–2) are committed in `results/degradation_topology/pilot/cifar10/` and are exploratory: the INDEX may describe them, never as a test.

## 9. Before this file: the design check and the pilot

- **Design check.** One run per cell, seed 0, WESAD, to check that every mechanism fires: losses drop messages, `bernoulli` takes edges offline, the crash re-parents edges, the partition cuts zone A, and autonomy runs zone rounds.
  - It also set the deadlines. It ran on 2026-11-13 and was looked at. It is not committed.
  - It found a bug in the sweep: a path swept inside another swept value leaked across scenarios. It was fixed in `1d5cca9`, and the check was rerun.
- **Timing.**
  - A WESAD run takes 13–30 s of wall time with six in parallel on this PC (24 logical cores).
  - A 2-round CIFAR-10 run in two levels took 79 s of wall time alone (about 37 s per round), with nominal virtual rounds of 73 s. So a 20-round run takes about 13–14 min alone.
- **Pilot results:** filled in §6 from `results/degradation_topology/pilot/`.
