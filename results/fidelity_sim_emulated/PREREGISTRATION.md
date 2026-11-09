# Pre-registration: fidelity of the simulation against the emulation (issue #198)

Written on 2026-11-06, before any run of `experiments/fidelity_sim_emulated.yaml` on either backend. Its git commit is its date: nothing in this folder may predate it except this file and the `INDEX.md` that points to it. A change to this file after the first run of the experiment is a deviation and goes in the INDEX under "Deviations".

The only run of this experiment before this file was a smoke test on `sim`: one configuration (`fog_b`, 5 epochs), seed 0, 2 rounds instead of 10, into a scratch folder. It changes the `design_id`, so it can never pair with the real runs, and it is not kept. It checked three things only: that the runs cite the calibration, that none is `estimated`, and that the largest message is 5.06 kB, within the sizes the paths were fitted on. No score or time from it is used below.

## 1. Question

With the profiles and links that the calibration of #197 fitted from microbenchmarks, how faithful is the `sim` backend (B, under test) to the `emulated` backend (A, reference) on federated runs that the calibration never saw?

The three levels are those of I1 §12. The rule is `docs/RULES.md` §3: the validation cites a closed calibration and uses other loads. The spec is `docs/superpowers/specs/2026-11-03-onion-fl-infrastructure-design.md`, §12 and §13.

## 2. The calibration cited

| | |
|---|---|
| `calibration_id` | `b55104b3f71d684dedc413a000eec3a67eb1caba134cc9ff6a95648fb35c8dfc` (swapped in on 2026-11-09 by steps 1–4 below; it was **PROVISIONAL**: `5d7d0e7011bf1fd693725bcb85ac61c5abae3d460867169f8909954c1b937cde`) |
| spec | `calibrations/emulated_lab.yaml` |
| image of its sessions | `onionfl-a197b-emulated:latest`, `sha256:a826e66e…` (`calibration.json`, `environment.image`; it was `onionfl-a197-emulated:latest`) |

The id above is the one-session calibration of #197. Its own check failed: the compute rates moved by −23 % to +54 % between sessions (`results/calibration_emulated/INDEX.md`). It is being refitted on several sessions, with pinned CPUs and a prediction interval between sessions. **No run of this validation starts before that refit is closed.** Then, in one commit, before any run:

1. Replace the id in `infrastructures/fidelity_validation_lab.yaml`, every occurrence. The refit's `--infrastructure-out` rewrites `infrastructures/emulated_lab_calibrated.yaml`, and `tests/test_fidelity_validation.py` fails until both files agree.
2. Set `backend.image` in the experiment to the image of the refit's sessions.
3. If the refit's spec pins other CPUs, set `backend.pin` to match. The same test checks it against `calibrations/emulated_lab.yaml`.
4. Replace the id in this section and in §5. That is the only edit of this file allowed after it is committed, and it is not a deviation.
5. If the refit's `valid_up_to` is below 5.1 kB, or its spec no longer measures workload `0ed95108ec44feea`, stop. The design no longer fits the calibration, and this file must be rewritten before any run.

From then on the calibration is closed. Changing a profile after seeing a result of this validation needs a new calibration, declared in the INDEX (I1 §13).

## 3. Design

**Fixed in every run:**
- Data: SWELL-KW physiology (`datasets/swell_physiology.yaml`), roles `test: 0.2`, `scaler: global`, role seed 0. That gives 20 training edges and 5 test subjects at the root.
- Learning: `modular_mlp` (adapter 32, trunk [16]), the `standard` trainer (lr 0.01, batch 32) and FedAvg, for 10 rounds. Full quorum and no deadline at every level.
- Infrastructure: `infrastructures/fidelity_validation_lab.yaml`, which is the calibrated lab with the 5 devices the calibration measured. The edges go 4 per device (`colocated`), the root runs on `cloud-1`, and so does the MQTT broker.
- Codec: `npz`. Its messages are 0.5–5.1 kB, and the paths were fitted up to 10 kB.
- Seeds: 0–4.

**The six configurations:**

| Label | Topology | Fog on | `local_epochs` |
|---|---|---|---|
| `cloud-1ep` | `swell_cloud_npz` (2 levels) | — | 1 |
| `fog_a-1ep` | `swell_one_fog_npz` | `fog-a` (40 ms to the broker) | 1 |
| `fog_b-1ep` | `swell_one_fog_npz` | `fog-b` (60 ms to the broker) | 1 |
| `cloud-5ep` | `swell_cloud_npz` | — | 5 |
| `fog_a-5ep` | `swell_one_fog_npz` | `fog-a` | 5 |
| `fog_b-5ep` | `swell_one_fog_npz` | `fog-b` | 5 |

**How it differs from what was calibrated.** The calibration ran microbenchmarks only: lone trainings, `fedavg` combinations and MQTT messages one at a time. No FL run went into it. This experiment is also not `experiments/fidelity_emulated.yaml` (#195), which used two fogs, the `json` codec, 5 epochs and 3 seeds: the topologies, the codec, the number of hosts and the epochs all differ here. The workload is the calibrated one, because any other would be estimated by FLOPs and refused. The calibration found that estimate off by a factor of 5 to 6 for these models.

**Conditions.**
- Both backends run on the same machine, Docker Desktop on WSL2, with the image of the calibration's sessions. Each container is pinned to the CPUs it had in those sessions (`backend.pin`), with no other heavy work running.
- Before the emulated session, `docker ps` must show no other project's containers, and no other agent's training may run. The INDEX records the start and end of the session and anything else that ran during it.
- Emulated runs go one at a time, seed by seed: the six configurations of seed 0, then those of seed 1, and so on. A drift of the machine then falls on the seeds, which the analysis pairs. `sim` runs may run in parallel, since their clock is virtual.
- A run that fails or does not finish is run again once with the same seed; the script resumes. If it fails again, it is reported and left unpaired. No finished run is ever discarded or repeated because of its result.

## 4. Analysis

One command, with the bootstrap's defaults (2 000 resamples, seed 0):

```bash
onion_fl fidelity runs/fidelity_validation_emulated runs/fidelity_validation_sim \
  --calibration <calibration_id> --metric time --metric bytes \
  --metric macro_f1 --metric loss --series global/swell \
  --csv fidelity.csv --html fidelity.html > fidelity.json
```

The harness refuses runs that cite another calibration, that have `estimated` rates or that send messages larger than the paths hold. The intervals are 95 % bootstrap intervals over the pairs (performance), or a bootstrap paired by seed over the configurations (decision). The noise ceiling is that of the seeds of A, single runs against single runs. That is more conservative than the means over 5 seeds that the decision compares.

### H1. Functional fidelity (all 30 pairs)

- **H1a.** The protocol events of every node match in order (`functional.protocol`).
- **H1b.** The bytes per logical link and round match exactly (`functional.bytes`). The `npz` arrays have fixed sizes, so this should hold even though the floats differ.
- **H1c.** The final models match within the tolerance of `test_runtime_equivalence` (rtol 10⁻⁶, atol 10⁻⁷).
  - They are not expected to match bit for bit: torch on Windows and on Linux use other kernels, as in #195. That is reported, not judged.

H1 holds if all three hold in 30 of 30 pairs.

### H2. Performance fidelity (relative error (B − A) / A, mean over pairs, with its 95 % CI)

Let r̂ be the calibrated training rate of `edge_device` for workload `0ed95108ec44feea`, and [r_lo, r_hi] its 95 % interval in the calibration. For the refit, that is the prediction interval of one more session. Simulated compute time goes as 1/r̂ and emulated as 1/r, so an emulated rate r inside that interval gives an error inside the **calibration band** [r_lo / r̂ − 1, r_hi / r̂ − 1]. With the provisional calibration the band was [−0.103, +0.126]; with the refit (r̂ = 16 993 samples/s, prediction interval [3 348, 86 256]) it is [−0.803, +4.076] (step 4).

- **H2a, compute time of an edge per round.** The point estimate of the mean relative error lies inside the calibration band.
- **H2b, round duration at the root.** |mean relative error| ≤ 0.25, and its 95 % CI lies within [−0.40, +0.40].
- **H2c, link latency.** |mean relative error| ≤ 0.25.
  - The prediction is that this fails, with the simulation too fast: the CI lies entirely below 0. #195 measured 0.87 s emulated against 0.12 s simulated, and the cause was not established.
  - The latency is measured from send to delivery, and a delivery to a device busy training waits until the training ends. Its magnitude is reported either way.
- **Run duration** is described, not judged: it includes starting Python and torch in the containers, which the simulation does not model.

The same quantities are reported separately for 1 and 5 epochs, from the pairs of each. That split is secondary.

### H3. Decision fidelity (K = 6, 5 seeds)

For each metric, the harness reports Kendall's τ-b between the rankings of the means over seeds, top-1 agreement, the selection regret (in the metric's units and relative), their 95 % intervals and the bootstrap probability of top-1 agreement, each next to the noise ceiling.

- **H3a, `time`** (the run's duration on each backend's clock; lower is better). τ-b ≥ 0.6, the top-1 agrees, and the regret is no larger than the noise ceiling's.
  - Prediction: both backends pick `cloud-1ep`.
  - The epochs dominate (5 against 1). The finer test is the order within each number of epochs. The simulation should put `cloud` before `fog_a` before `fog_b`, because a fog adds a fog–cloud leg through the broker, longer on `fog-b`.
  - τ-b within each epoch group, from the means in `fidelity.json`, is reported as secondary. With 3 configurations its interval is too wide to judge.
- **H3b, `bytes`.** τ-b = 1 and the regret is 0: `cloud` sends fewer bytes than the fog configurations, and the epochs change nothing. This follows from H1b. If H1b holds and H3b does not, it is a bug, not a fidelity result.
- **H3c, `macro_f1` and `loss` on `global/swell`.**
  - With full quorum, FedAvg through one fog gives the flat FedAvg up to floating point, so the three topologies should tie within each number of epochs.
  - Only the 1 vs 5 epochs contrast can rank anything. The top-1 among tied configurations goes to the first one, so it is not interpreted.
  - The harness ranks exact floats, so inside a tie group it ranks floating-point noise (about 10⁻⁷ between the two torch builds), which deflates its τ-b.
  - H3c is therefore judged on the τ-b of the means rounded to 4 decimals, computed from `decision.<metric>.scores` in `fidelity.json`, and the harness's raw τ-b is reported beside it. H3c holds if that rounded τ-b is 1.
  - If every configuration also ties across epochs (a constant predictor: #195's macro-F1 was 0.47–0.48 for every seed), the harness says so, and the decision on learning is reported as uninformative. It is not counted for or against the simulation.

**Reading H3.** The simulation is decision-faithful on a metric when τ-b, top-1 and regret all meet the criterion above. A disagreement counts only when it exceeds the noise ceiling (top-1 below the ceiling's, regret above it).

**Not available: the factorised predictor.** Rounds to target × analytical time per round (I1 §12, capabilities N7/M5) is not implemented yet, and the harness leaves its slot empty. So this validation cannot say whether the simulation decides better than that analytical model. Its INDEX must not claim so.

## 5. What is reported

`INDEX.md` gives the following:
- the identities: `calibration_id` (final), `infrastructure_id`, the `topology_id`s, `data_id`, the `config_id` of each configuration and backend, and the code commit with whether the tree was clean;
- the session's times and load;
- H1–H3 with every number from `fidelity.json`, each marked as met or not met against this file;
- the deviations.

The run folders are copied as in `results/fidelity_emulated/`: `run.json`, `summary.json` and `model.npz`, with the scratch paths removed. `events.jsonl` is kept only if the folder stays under the size of the committed results of #195; otherwise the INDEX says where `verify_run` passed.
