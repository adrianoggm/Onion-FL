# Deadline-coupling pilot: pre-registration

Issue #213 (A9). It implements §5 of `docs/superpowers/specs/2026-11-04-onion-fl-capabilities.md`. This file was committed **before any run of the pilot**. The INDEX cites its commit. Nothing here changes after a result is seen. A change would need a new pre-registration, dated and committed, and the INDEX would name both.

## Question

When a deadline couples which clients take part with how much their data weigh, does the factorised predictor (N7: rounds to target from a synchronous control × analytic time per round) choose the wrong configuration of a hierarchical federation? And does the ex ante diagnostic (N8) tell beforehand when it does?

Possible verdicts: **the gap exists**, **the gap falls**, or **the question is not identifiable with these data**.

## Design

Everything runs in `sim`. The experiment is `experiments/pilot_deadline_coupling.yaml`, the infrastructure is `infrastructures/pilot_deadline_coupling.yaml`, and `scripts/pilot_deadline_coupling.py` runs the cells and analyses them. All three are committed before this file.

- **Data.** FEMNIST, the descriptor's 200 fixed writers, one edge per writer. The roles are those of `femnist_reference` (role seed 0): 40 writers reserved for the test, 160 that train. The test writers are the same in every run, and the scaler is fitted on the training writers. The pilot does not use SWELL or WESAD: §5 allows them only if step 0 shows that the noise permits it, and with 15–25 clients the ceiling would hide the configurations.
- **Model and training.** These are fixed before any run and never tuned: `femnist_reference`'s model and trainer (`modular_mlp`, adapter 128, trunk [128], dropout 0.2, Adam lr 0.001, batch 32, one local epoch) and 20 rounds. FedAdam runs at the library's default server rate (0.01). No search is run, and no hyperparameter is chosen on test or validation scores.
- **Configurations (K = 6).** A federation × an algorithm.
  - Federations: flat, 2 fogs or 4 fogs, with κ₁ = 1 (every fog round goes up).
  - Algorithms: FedAvg (`replace`) and FedAdam, from two different families.
  - Flat is a star: one aggregator, `hub`, sits on the cloud's own host, because the root takes evaluators only. Its link to the coordinator costs nothing.
- **Regimes.**
  - *Control:* synchronous, with no deadline and no coupling. Only the predictor uses it.
  - *Deadline:* a 1 s deadline with `drop`, quorum 1, on the node the edges answer to.
  - *FedBuff-style:* the round closes once a quarter of the participants have answered (`close_at_quorum`). Late updates join a later round, weighted by `(1 + staleness)^-0.5`. The 1 s deadline stays as a cap.
  - With fogs, the cloud is synchronous in every regime.
- **Infrastructure.**
  - 160 field devices in three classes, with train rates of 1000, 300 and 100 samples/s. Each rate is lognormal with σ = 0.4 (the new `train_sigma` of a profile), so meeting the deadline is a probability.
  - Fogs combine 1e8 B/s and the cloud 2e8 B/s.
  - Bandwidth is dedicated: no contention, which the predictor cannot see.
  - Before any run, the ex ante diagnostic gave a mean participation of 0.57–0.64 under the deadline, in the cells checked (seed 0, without training).
- **Coupling.** The assignment is `coupled` on a 0.34/0.33/0.33 mix, with five cells:
  - `rho = 0` on the samples: the **naive simulation**;
  - `rho = ±0.6` on the samples;
  - `rho = ±0.6` on the label divergence.
  - The Spearman achieved is about ±0.48: three speed classes tie. `plan` reports it. The assignment's own seed is 0 in every run, so a cell keeps its devices across training seeds.
- **Target.** Test accuracy at the root of **0.35**, fixed now. `femnist_reference` (seed 0) was at 0.24 at round 5 and 0.38 at round 10, so the synchronous FedAvg should reach it near round 9 of 20. A run that never reaches it is censored at its duration, and counts at that duration (a lower bound). Lower time to target is better.
- **Seeds.** Training seeds 0–9 in the naive cells (step 0), and 0–4 everywhere else. Each truth run is paired with the control of the same configuration and seed, by `control_id`.

| Step | Cells | Runs |
|---|---|---|
| 0 | control (6 configurations × seeds 0–4); naive cell of each regime (2 × 6 × seeds 0–9) | 150 |
| 1 | the 4 coupled cells of each regime that passed step 0 (2 × 4 × 6 × seeds 0–4) | ≤ 240 |

## Analysis and criteria

`scripts/pilot_deadline_coupling.py analyse` computes everything below. Its pure functions are tested in `tests/test_pilot_deadline_coupling.py`.

**Step 0: noise ceiling (the gate).**

- In the naive cell of each regime, the 10 seeds are split into two disjoint halves of 5, every way (126 splits). Each half is a replica of the study, and it picks the configuration with the lowest mean time to target.
- The **split-half top-1** is the share of splits whose two halves agree. The **ceiling regret** is the 95th percentile of the relative regret of one half's winner in the other half (both directions).
- For reference, the script also reports the top-1 between single seeds (`ceiling` over the seeds) and the controls' top-1.
- **If a regime's split-half top-1 is < 0.6, the regime is not identifiable.** Its coupled cells are not run, and this is documented. If both regimes fail, the pilot stops: *the question is not identifiable with these data*.

**Steps 1–2: regret against the truth.**

- The **truth** of a cell is the mean over seeds 0–4 of each configuration's time to target in that cell.
- Each contender picks a winner: the factorised predictor and its FLOSS-style variant (the mean of their predictions over seeds 0–4), and the naive simulation (the winner of the same regime's `rho = 0` cell, seeds 0–4).
- Its **relative regret** is the truth's time at that winner minus the best truth time, divided by the best.
- A contender is **above the ceiling** in a cell when its relative regret exceeds that regime's ceiling regret.
- **Primary contender:** the FLOSS-style predictor. It already accounts for the deadline, so a failure cannot be blamed on ignoring it. The plain predictor and the naive simulation are secondary.
- Over the coupled cells (`|rho| = 0.6`) of the regimes that passed step 0 (8 cells if both did), count the cells above the ceiling:
  - **The gap exists** if that count would be this unlikely as noise: binomial tail `P(X ≥ count | n, 0.05) ≤ 0.01`, which means 3 or more of 8, or 3 or more of 4.
  - **The gap falls** if the count is 0 or 1: the regret stays within the ceiling even with a deadline and `|rho| = 0.6`.
  - **Inconclusive** otherwise.
- The `rho = 0` cells are reported apart. A predictor above the ceiling there fails because of the deadline alone, not the coupling.

**Step 3: the diagnostic.**

- The units are the truth cells of the regimes that passed step 0, the `rho = 0` cell included (10 if both did). The outcome is whether the FLOSS predictor is above the ceiling in that cell.
- Scores, computed before training and saved in each `run.json`:
  - the diagnostic: the largest `|D|` over the cell's configurations, with the label divergence as statistic (the primary one);
  - baseline: 1 − the mean participation rate;
  - baseline: the largest FLOSS factor on the rounds to target.
  - The gradient-norm `D` is reported as secondary.
- For each score, the Mann–Whitney AUC with a 95 % percentile bootstrap over the units (2000 resamples, seed 0).
- **The diagnostic's contribution falls** if its AUC interval overlaps the interval of the participation rate alone (§5). It is **not identifiable** if every unit has the same outcome. With about 10 units the intervals are wide: an overlap is expected unless the separation is nearly perfect. The INDEX says so.

## Deviations from §5, decided now

- **No HPO replicas.** §5 asks for "5 seeds × 2 HPO replicas". The only score a search could pick trials on here is the test score, which would leak. So the hyperparameters are fixed a priori, and the replicas are seeds: two disjoint 5-seed replicas, over every split of 10 seeds.
  - Effect: the ceiling leaves out the noise of choosing hyperparameters. It **underestimates** the noise, which makes "the gap exists" easier to reach.
  - Consequence: a positive result has to be read as an upper bound, and it calls for a second pilot with a validation-based search.
- **Where the ceiling comes from.** Each regime's ceiling is measured in its `rho = 0` cell and applied to its coupled cells, which have 5 seeds only.
- **Assignment seed.** The coupled assignment keeps seed 0. The coupling is one realisation of devices per cell, not a distribution of them.
- **FedBuff with a deadline.** The diagnostic and the FLOSS variant see deadlines, not quorums. The FedBuff-style regime keeps the 1 s deadline as a cap, and their `p_i` is the deadline's.
- **The rho actually achieved.** It is about ±0.48 instead of ±0.6, because of ties between devices of the same class.

## What is committed

- The `run.json` (without `host` and `pid`) and `summary.json` of each cited run.
- `step0.json` and `analysis.json`: per run, its identity (`run_id`, `run_hash`, `config_id`, `topology_id`, commit), its time to target and the diagnostic; per cell, the tables and the verdicts.
- No model is committed: the pilot makes no claim about a final model. The events (needed to recompute the time to target) stay out because of their size.
