# Deadline-coupling pilot (sim, real data)

Issue #213 (line A, capabilities spec §5). The design, the target and the criteria were pre-registered in [`PREREGISTRATION.md`](PREREGISTRATION.md), committed in `b7f0f2f` before any run. Nothing in it was changed after a result was seen. The deviations taken during the runs are listed at the end.

```bash
python scripts/pilot_deadline_coupling.py run --step 0 --workers 6   # 150 runs (resumed twice, see below)
python scripts/pilot_deadline_coupling.py analyse --step 0           # step0.json: the gate
python scripts/pilot_deadline_coupling.py run --step 1 --workers 8   # 240 runs
python scripts/pilot_deadline_coupling.py analyse                    # analysis.json
python scripts/pilot_deadline_coupling.py export                     # runs/<run_id>/
```

It ran from 2026-11-06 to 2026-11-09: 390 runs, all `finished`, all in `sim`. FEMNIST is the descriptor's 200 fixed writers (40 test and 160 training writers, role seed 0), as in `femnist_reference`.

## Identity

| Identity | Value |
|---|---|
| pre-registration | commit `b7f0f2f` |
| `data_id` | `7d8238e25baf310766da9c287196bd4e42a45a3a709873d21edc89830d10ad20` (the same as `femnist_reference`) |
| `infrastructure_id` | `10a39546e2e3476b646a1559ebf347a995314f30e42c8f4bc07ef0064c28f039` |
| `topology_id` flat | `4aeae8368838f01c971588156d47235c0ed25e42ec2321df3a3c8a4abe3c6e78` |
| `topology_id` 2 fogs | `b4142e77a9082bd89b5640ea9a6cdf776726235cc917dbf8d6340a06674df2f9` |
| `topology_id` 4 fogs | `80d45cbee3ed481093da8f2e29f7fde652ac1049eec791d7406cd7e3e1ac2067` (the same as `femnist_reference`: the id ignores round settings) |
| code, step 0 (150 runs) | commit `b7f0f2f`, clean tree |
| code, step 1 (240 runs) | commit `a0af5e3`, which only adds `step0.json` and `step0_analysis.json` to `b7f0f2f`; 238 clean, 2 rerun (see deviations) |
| analysis and export | commit `428f42f` (export change only) |

There are 66 `config_id`s: 3 regimes × 5 couplings (only `rho = 0` for the control) × 6 configurations. Each run's `run_id`, `run_hash`, `config_id`, `topology_id` and commit are in `analysis.json` (`runs`) and in `runs/<run_id>/run.json`.

## Step 0: the noise ceiling (the gate)

In the naive cell (`rho = 0`) of each regime, over 10 seeds (`step0.json`):

| Regime | Split-half top-1 (126 splits) | Ceiling regret (p95) | Single-run top-1 | Gate (≥ 0.6) |
|---|---|---|---|---|
| Deadline | 0.921 | 9.1e-5 | 0.467 | passed |
| FedBuff-style | 1.000 | 0.000 | 0.644 | passed |

- **Both regimes passed**, so step 1 ran all 8 coupled cells (240 runs).
- **The ceiling regret is practically zero in both regimes.** Two halves of 5 seeds almost always pick the same winner, so a contender is "above the ceiling" as soon as it picks any configuration other than the truth's winner. This is what the pre-registered rule gives with these data, and it is applied as it is.
- **The controls (synchronous, seeds 0–4)** reach the target in 56–59 virtual seconds. Their single-run top-1 is 0.6.

Mean time to the target (0.35 accuracy), in virtual seconds, in the naive cells (seeds 0–9):

| Configuration | Deadline | FedBuff-style |
|---|---|---|
| flat, FedAvg | 10.64 | 6.21 (censored, 10/10) |
| flat, FedAdam | 10.04 | 6.21 (censored, 10/10) |
| 2 fogs, FedAvg | 11.26 | 2.73 |
| 2 fogs, FedAdam | 10.62 | 9.07 (censored, 10/10) |
| 4 fogs, FedAvg | 10.37 | **2.59** |
| 4 fogs, FedAdam | **9.78** | 8.93 (censored, 10/10) |

- **In the FedBuff-style regime, FedAdam never reaches the target, and flat FedAvg doesn't either.** Over the 180 FedBuff-style runs of the pilot, these four configurations end at round 20 with a test accuracy of 0.08–0.13. The censored runs count at their duration, as pre-registered, which still makes them slower than the hierarchical FedAvg runs.
- With the deadline, no run is censored. Final accuracy is 0.47 (FedAvg) and 0.50 (FedAdam), against 0.49 and 0.51 for the controls.

## Steps 1–2: regret against the truth

The truth of each cell is the mean over seeds 0–4. Relative regret of each contender's pick (`analysis.json`, `step2`):

| Regime | Coupling | Truth winner | FLOSS (primary) | Factorised | Naive sim |
|---|---|---|---|---|---|
| Deadline | `rho = 0` | 4 fogs, FedAdam | 0 | 0 | 0 |
| Deadline | samples −0.6 | 4 fogs, FedAdam | 0 | 0 | 0 |
| Deadline | samples +0.6 | 4 fogs, FedAdam | 0 | 0 | 0 |
| Deadline | label div. −0.6 | 4 fogs, FedAdam | 0 | 0 | 0 |
| Deadline | label div. +0.6 | 4 fogs, FedAdam | 0 | 0 | 0 |
| FedBuff | `rho = 0` | 4 fogs, FedAvg | **2.46** | **2.46** | 0 |
| FedBuff | samples −0.6 | 4 fogs, FedAvg | **2.85** | **2.85** | 0 |
| FedBuff | samples +0.6 | 2 fogs, FedAvg | **2.08** | **2.08** | **0.95** |
| FedBuff | label div. −0.6 | 4 fogs, FedAvg | **3.06** | **3.06** | 0 |
| FedBuff | label div. +0.6 | 4 fogs, FedAvg | **2.32** | **2.32** | 0 |

Bold means above the ceiling. Both predictors pick 4 fogs with FedAdam in every cell.

Pre-registered verdict over the 8 coupled cells:

| Contender | Cells above the ceiling | Verdict |
|---|---|---|
| FLOSS-style predictor (primary) | 4 of 8 | **the gap exists** (`P(X ≥ 4 | 8, 0.05)` = 3.7e-4 ≤ 0.01) |
| Factorised predictor | 4 of 8 | the gap exists |
| Naive simulation | 1 of 8 | the gap falls |

**How to read it.** The pre-registered rule says "the gap exists", but the data do not attribute it to the coupling:

- **All 4 failures are in the FedBuff-style regime, and the predictors fail just as badly in its `rho = 0` cell** (regret 2.46). The pre-registration says such a failure comes from the regime alone. Here the cause is concrete: the predictors see the deadline but not the quorum or the staleness weighting, and under buffering FedAdam (at the fixed server rate 0.01) collapses. They choose a configuration that never reaches the target.
- **With the deadline alone, the gap falls:** 0 of 4 coupled cells, and the `rho = 0` cell is also right. Every contender picks the truth's winner (4 fogs, FedAdam) in all five cells. With `|rho| = 0.6` (an achieved Spearman of about ±0.48), the coupling shifts the times (from 9.0 s to 12.8 s), not the ranking.
- **The naive simulation's only failure** (FedBuff, samples +0.6) comes from one censored seed: 4 fogs with FedAvg, seed 3, reaches 0.338 at round 20 and counts at 16.6 s. Without it, 4 fogs and 2 fogs are tied at about 2.9 s. It is fragile.
- **The ceiling is practically zero,** so "above the ceiling" means "any other winner". The pre-registration already says the ceiling underestimates the noise, because it leaves out hyperparameter search. The regrets that count here are large (2–3 times the best time), so this does not change the reading.

In short, the answer to the pilot's question is: **with a deadline, the factorised predictor (and its FLOSS variant) does not choose wrongly, with or without coupling. It does choose wrongly in the FedBuff-style regime, but it does so without coupling too, so the coupling is not what explains it.** Telling the coupling's effect apart would need a regime where the predictor is right at `rho = 0` and wrong at `|rho| > 0`, and no cell in this pilot is like that.

## Step 3: the diagnostic

There are 10 units (the 5 cells of each regime), and 5 positives (the FLOSS predictor is above the ceiling). Mann–Whitney AUC, with a 95 % bootstrap interval (2000 resamples, seed 0):

| Score | AUC | 95 % interval |
|---|---|---|
| Diagnostic: max `|D|`, label divergence (primary) | 0.50 | [0.13, 0.90] |
| Diagnostic: max `|D|`, gradient norm (secondary) | 0.50 | [0.13, 0.90] |
| 1 − mean participation | 0.50 | [0.13, 0.88] |
| Max FLOSS factor on the rounds | 0.50 | [0.12, 0.88] |

- **Pre-registered verdict: the diagnostic's contribution falls.** Its interval overlaps that of participation alone.
- **The reason is structural, not a matter of noise.** The positives are exactly the FedBuff-style cells. Every score is computed before training from the deadline and the coupling, which are the same in both regimes, so each score is identical between a cell and its FedBuff twin. No score that does not see the quorum can separate them, so the AUC is 0.5 by construction.
- This pilot does not test whether `D` predicts a coupling failure, because no failure due to the coupling happened.

## Deviations during the runs

- **Two interrupted batches.** The parent session restarted twice during step 0 (at 24/150 and at 96/125), killing the batch.
  - Each time, the runs without a `finished` status (6 each time) were deleted, and the script resumed only the missing scenarios. The finished runs are unchanged.
  - The second resume used 8 workers instead of 6. Workers do not change a `sim` result.
- **Two runs rerun.** Two step-1 runs (FedBuff, samples +0.6, 2 fogs, FedAvg, seeds 2 and 3) had `code_version.commit = unknown`. The git call probably failed while a commit was being made at the same moment.
  - They were moved out of `runs/` and rerun at the same commit.
  - The new `model.npz` files are bit-identical to the old ones (SHA-256), and so are their summaries.
  - The new runs are marked `dirty: true`, only because untracked analysis output was present.
- **The exported `run.json` files leave out `diagnostic.edges`.** These are the per-edge inputs of the diagnostic, about 70 % of each file: with them, the 390 runs would take about 14 MB.
  - Every value the analysis uses (`D`, its statistics, participation, FLOSS) is kept.
  - As with `host` and `pid`, `verify_run` cannot check these copies.
  - This was decided after the runs. It changes no number.

## Files

- `PREREGISTRATION.md`: the design and the criteria, committed before any run.
- `step0.json`: the gate. `step0_analysis.json`: the same analysis, with the 150 step-0 runs.
- `analysis.json`: the gate, the step 2 cells and verdicts, the step 3 AUCs and every run's identity, time to target, rounds, censoring, final accuracy and diagnostic.
- `runs/<run_id>/`: `run.json` (without `host`, `pid` and `diagnostic.edges`) and `summary.json` for each of the 390 runs.

No `model.npz` and no `events.jsonl` are committed: the pilot makes no claim about a final model, and the events (11–27 MB per run) are left out because of their size.
