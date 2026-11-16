# Degradation × topology (#290)

**Question.** Does the choice of topology (flat, two levels, three levels) change, in a statistically significant way, how federated learning and the system degrade?

The study is pre-registered in [`docs/preregistration/degradation_topology.md`](../../docs/preregistration/degradation_topology.md):

- the design was registered at commit `8ee889e`;
- the seeds were fixed at `b277ac7`, from the pilot and before the main run;
- the confirmatory answer is **WESAD only**, because CIFAR-10 was dropped before the main run (§8).

## Answer

On WESAD, with 30 paired seeds per cell, **the topology does not change how learning degrades under most degradations: the final macro-F1 is equivalent within ± 0.02**. Where it does change it, the cause is the quorum, the traffic or the composition of the training subjects, not resilience as such.

1. **Equivalent final macro-F1 (TOST < 0.05, margin ± 0.02), for both *flat − two* and *three − two*:**
   - `nominal`, 5 % loss, `4g` latency, 20 % edge dropout with quorum 0.5, and the crash of a fog host;
   - the WAN partition of zone A with autonomy.

   Without autonomy, *three − two* is equivalent too.
2. **The strict quorum (`dropout_k1`) is where the topology matters most.**
   - Flat loses 0.163 macro-F1 against two levels (95 % CI [−0.242, −0.089], p_adj = 0.002).
   - Its hub needs all 10 edges in a round, so it loses 17.8 of 20 root rounds, against 9.3 for two levels and 0.7 for three levels.
   - It reaches the target in only 13 of 30 runs, and 3 runs never close a single round.
   - Three levels loses 0.013 against two (CI [−0.025, −0.001], p_adj = 0.04). That is significant, but it is not equivalent within the margin.
3. **Flat "wins" on macro-F1 under 20 % loss (+0.024) and under the partition (+0.037). This is not robustness.**
   - The model scores higher when zone A's updates are missing. Under the partition, two and three levels also rise from 0.769 to 0.793, and with autonomy to 0.82–0.83.
   - The test set is three subjects (S6, S7, S10). Leaving out zone A's subjects happens to suit them better.
   - Flat loses more of zone A under loss, because each update crosses two links. It loses all of zone A under the partition, because it has no fog to keep it.
   - These are effects of which subjects get averaged in, and the study cannot separate them from the topology.
4. **Time to target: flat is about 15 % slower** (ln ratio 0.137, so ×1.15, p_adj < 0.001) whenever the early rounds are clean. Each of its rounds pays the WAN latency to every edge.
   - Under the strict quorum, flat is ×2.9 slower than two levels and three levels is ×0.58. Both are significant.
   - Under loss, every time contrast is inconclusive and underpowered, as the pilot predicted.
5. **The system cost is where the topologies really differ.**
   - **WAN traffic.** Flat moves about 5× the WAN bytes of the hierarchies at every level (ln ratio 1.54–1.91).
   - **Fog crash.** Two levels pays a cross-zone failover: its WAN traffic grows from 3 to 7 MB (2.4× three levels'), and its run lasts 94 s of virtual time against 67 s for three levels and 17 s for flat. Three levels re-parents its edges locally.
   - **Lost rounds under 20 % loss:** flat 12.1, two 5.9, three 3.0 (both contrasts p_adj < 0.001).

**Where there is no difference, say so:** the final macro-F1 under nominal conditions, 5 % loss, 4g, dropout with quorum 0.5 and a fog crash. **Where we cannot tell:** time to target under loss, and *three − two* macro-F1 under 20 % loss (both underpowered, below).

## Primary metrics, H1 (topology effect at each degradation level)

Each row gives the mean paired difference against two levels, a 95 % paired-bootstrap CI, the Holm-adjusted sign-flip p over the two contrasts, the TOST p and the verdict. *equivalent* means TOST < 0.05; *different* means p_adj < 0.05 and not equivalent.

### Final global macro-F1 (margin ± 0.02)

| Level | flat − two | verdict | three − two | verdict |
|---|---|---|---|---|
| nominal | 0.000 [0.000, 0.000] | equivalent | 0.000 [0.000, 0.000] | equivalent |
| loss5 | +0.000 [−0.006, 0.006] | equivalent | +0.002 [−0.004, 0.009] | equivalent |
| loss20 | +0.024 [0.005, 0.043], p_adj 0.037 | **different** | −0.007 [−0.027, 0.014] | inconclusive ⚠ |
| 4g | 0.000 [0.000, 0.000] | equivalent | 0.000 [0.000, 0.000] | equivalent |
| dropout | +0.000 [−0.003, 0.004] | equivalent | −0.004 [−0.008, 0.001] | equivalent |
| dropout_k1 | **−0.163** [−0.242, −0.089], p_adj 0.002 | **different** | −0.013 [−0.025, −0.001], p_adj 0.040 | **different** |
| crash | −0.002 [−0.005, 0.000] | equivalent | −0.001 [−0.003, 0.001] | equivalent |
| partition | +0.037 [0.030, 0.044], p_adj < 0.001 | **different** | +0.000 [0.000, 0.001] | equivalent |
| partition_autonomy | +0.008 [−0.001, 0.017] | equivalent | +0.009 [0.004, 0.014], p_adj 0.004 | equivalent (detectable, irrelevant) |

### Time to target 0.70 (ln ratio; margin ± ln 1.1 ≈ ± 0.095)

| Level | flat − two | verdict | three − two | verdict |
|---|---|---|---|---|
| nominal | +0.137 | **different** | +0.003 | equivalent |
| loss5 | +0.174 [−0.062, 0.426] | inconclusive ⚠ | −0.188 [−0.576, 0.240] | inconclusive ⚠ |
| loss20 | +0.075 [−0.115, 0.254] | inconclusive ⚠ | −0.115 [−0.256, 0.027] | inconclusive ⚠ |
| 4g | −0.035 | equivalent | +0.004 | equivalent |
| dropout | −0.035 | equivalent | +0.011 | equivalent |
| dropout_k1 | **+1.07** [0.90, 1.25] | **different** | **−0.54** [−0.68, −0.40] | **different** |
| crash, partition, partition_autonomy | +0.137 | **different** | +0.003 | equivalent |

All p_adj for *different* are below 0.001. ⚠ marks a contrast underpowered at n = 30 (below): its *inconclusive* is no evidence of anything.

- **Censoring.** Every run reaches the target except flat under `dropout_k1`, where 17 of 30 runs are censored at the end of the run. That is more than half, so its time contrasts are **lower bounds**: flat is at least ×2.9 slower there.
- **Time to target cannot see the crash or the partition.** Both fall at 10 s of virtual time, and every arm reaches the target by about 2.6 s at those levels. Their rows only repeat the nominal latency effect. This is a limit of the design. It was not foreseen in the pre-registration, and it is reported here instead of being changed.

## H2: the interaction (difference-in-differences against nominal)

Holm runs over the 16 contrasts of each metric. The significant ones on the primary metrics:

- **macro-F1:**
  - flat under `dropout_k1`: −0.163, p_adj 0.018;
  - flat under the partition: +0.037, p_adj 0.003;
  - three levels under the partition with autonomy: +0.009, p_adj 0.031, but equivalent within the margin.
- **time to target:**
  - flat under `dropout_k1`: +0.93, and three levels under it: −0.54;
  - flat under `4g` and `dropout`: −0.17, because flat's nominal latency penalty shrinks when the access links get slower.

  All have p_adj 0.003.

Every other H2 contrast on macro-F1 is equivalent, or inconclusive under 20 % loss. In short, the degradation slopes differ by topology only under the strict quorum and the partition, and in the partition the cause is the zone's absence, not its resilience (point 3).

## Secondary metrics

Cell means, 30 seeds each. WAN bytes are in MB.

| Level | Root lost rounds (flat / two / three) | WAN MB (flat / two / three) |
|---|---|---|
| nominal | 0.0 / 0.0 / 0.0 | 14 / 3 / 3 |
| loss5 | 0.0 / 0.2 / 0.2 | 14 / 3 / 3 |
| loss20 | 12.1 / 5.9 / 3.0 | 12 / 2 / 3 |
| 4g | 0.0 / 0.0 / 0.0 | 14 / 3 / 3 |
| dropout | 0.0 / 0.0 / 0.0 | 13 / 3 / 3 |
| dropout_k1 | 17.8 / 9.3 / 0.7 | 13 / 2 / 3 |
| crash | 0.0 / 0.0 / 0.0 | 14 / 7 / 3 |
| partition | 0.0 / 0.0 / 0.0 | 13 / 3 / 3 |
| partition_autonomy | 0.0 / 0.0 / 0.0 | 13 / 3 / 3 |

- **Per-zone fairness** (WESAD local_val, received model):
  - Under `dropout_k1`, flat's gap is smaller than two levels' (−0.147, p_adj < 0.001), but only because both of its zones are poor: 0.52 and 0.53.
  - Under the crash, flat's gap is 0.015 larger (p_adj 0.002).
  - Under loss, the gap is defined on only 12–22 pairs (edges whose last evaluation was lost) and is inconclusive.
- **Autonomy, descriptive only** (not a pre-registered contrast):
  - Under the partition, the cut zone runs 10 autonomous rounds in two levels and 20 in three (two leaf fogs).
  - The global macro-F1 rises from 0.793 to 0.822 in two levels and to 0.831 in three.
  - All of `contrasts.json` is in this folder.

## Power at n = 30

The seed rule (pre-registration §6) recomputed on the main run gives a largest `seeds_needed` of 1146, so these contrasts stay underpowered:
- macro-F1: `loss20` *flat − two* (power 0.52) and *three − two* (0.45), `dropout_k1` *flat − two* (0.07, yet significant: the effect is large);
- time to target: `loss5` (0.11 and 0.07), `loss20` (0.16 and 0.24), `dropout_k1` (0.17 and 0.26, yet significant).

H2 is not powered by the rule.

## Deviations from the pre-registration

1. **Three runs never closed a root round.** They are flat under `dropout_k1`, seeds 115, 117 and 128, which is about 10 % of 30, as 0.89²⁰ predicts.
   - No global evaluation was emitted, so their final macro-F1 is that of the coordinator's model, which is still the initial one: 0.514, 0.262 and 0.676.
   - It is scored by `scripts/degradation_topology_summary.py` (`rescore`) on the same test subjects, per subject and weighted by rows, as `global/*` is.
   - On 6 random runs that do have `global/*`, the same function gives identical values (|Δ| < 1e-9).
   - The `rescored` column marks the three. No run was excluded (§7).
2. **Three runs were rerun** (§7): two levels `crash` seed 105, and three levels `partition_autonomy` seeds 122 and 123.
   - Their `code_version` read `unknown`: the `git` call has a 10 s limit and timed out with the CPU at 100 %.
   - They were rerun at `b277ac7` with a clean tree, and the final models are identical bit for bit.
   - The replaced runs are not committed.

## Identity

| | |
|---|---|
| experiment | `experiments/degradation_topology_wesad.yaml`, 27 `config_id`s (one per cell), seeds 100–129 |
| code | commit `b277ac7`, clean tree, for all 810 runs (`run_metrics.csv`: `commit`, `dirty`) |
| analysis | `scripts/degradation_topology_summary.py` at this folder's commit; `run_metrics.csv` (one row per run with `run_id`, `run_hash`, `topology_id`, `infrastructure_id`, `config_id`, `data_id`) and `contrasts.json` (H1, H2, seed rule) |
| runs | `wesad/runs/<run_id>/`: signed `run.json` and `summary.json` of every run, and `model.npz` of the three rescored runs. The other models (40 kB each) and the event files are not committed |
| pilot | `pilot/`: 81 WESAD runs (seeds 0–2) and 6 exploratory CIFAR-10 runs at `ee75386`, used only for the seed rule |

Reproduce: `onion_fl run experiments/degradation_topology_wesad.yaml --workers 6` at `b277ac7`, then the two commands in the summary script's docstring with `--target 0.7`.
