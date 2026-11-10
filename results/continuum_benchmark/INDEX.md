# The continual benchmark on the SWELL and WESAD streams (new framework, real data)

The experiments below were run on 2026-11-06 with SWELL-KW (computer interaction, per minute) and WESAD (wrist, 60 s windows). Issue #163 (continuum C8).

```bash
onion_fl run experiments/continuum_benchmark_select.yaml --workers 4             # step 1 of the selection, on validation
onion_fl run experiments/continuum_benchmark_select_replay.yaml --workers 4      # step 2
onion_fl run experiments/continuum_benchmark_select_unlabelled.yaml --workers 4  # step 3
onion_fl run experiments/continuum_benchmark.yaml --workers 4                    # twice: the second is the determinism check
onion_fl run experiments/continuum_benchmark_sweeps.yaml --workers 4
onion_fl run experiments/continuum_benchmark_promotion.yaml --workers 4
python scripts/continuum_benchmark.py metrics runs/<run_id> ... --subjects test --curves curves.csv > run_metrics.csv
python scripts/continuum_benchmark.py table run_metrics.csv --by baseline dataset
```

**In one paragraph.** On these streams, no baseline learns a model that tells the classes apart on held-out subjects. Every model that keeps updating ends as a constant predictor on SWELL's test subjects (macro-F1 ≈ 0.40), and on WESAD's (0.26). What the baselines differ in is how fast they follow the class that is arriving, which is what the prequential score rewards. Warm start follows it best (SWELL prequential 0.795). Replay and the semi-supervised term slow it down slightly (0.72), and retraining from scratch more (0.55). Frozen does not follow it at all (0.32). Frozen has the best WESAD test score (0.46 against 0.26), but not because it separates the classes: its v0 mostly predicts no stress, which is the majority of WESAD's test rows, while the updating models end at stress. So what this measures is adaptation to a prior that switches in blocks, not learning of the stress signal. Memory and the unlabelled rows add nothing measurable here. A champion/challenger promotion with these validation streams almost never promotes, so it serves the initial weights.

## Setup

- **Streams** (as in C5, [continuum_unlabelled/](../continuum_unlabelled/INDEX.md)):
  - SWELL + WESAD in time order after a 20-minute bootstrap, `aligned`, `speed: 60`;
  - a tenth of the rows labelled, 10 minutes late (`labels: {fraction: 0.1, delay: 10m}`). The rest waits unlabelled;
  - a `schedule` trigger every 10 minutes of data, with a status every 10 minutes (C6), until the last label: 17 or 18 rounds;
  - FedAvg with SGD on `four_fogs_swell_wesad_lossless` (segregated placement, quorum 1.0), `modular_mlp` 64 / [64, 32], dropout 0.2.
- **Five baselines.** They differ in `continual.update`, a setting added by this issue:
  - **frozen** (`update: frozen`): each edge trains v0 on its history, with the first model it gets, and then answers idle. An edge with no labelled row in its history does not take part in v0;
  - **scratch** (`update: scratch`): each training starts from the initial weights and uses every trainable row the edge has, old and new, all of them weighted. It trains in the same rounds as warm, the rounds in which new rows arrived. The model served between rounds is the one received;
  - **warm** (`update: warm`, the default): from the model received, on the new rows (C3);
  - **warm + replay**: warm with a reservoir memory of 512 rows and `replay_ratio: 0.25` (C4);
  - **warm + replay + semi-supervised**: the same with the `consistency` objective (noise 0.1, weight 0.5) on the unlabelled pool (C5).
- **Selection on validation, before any test run.** Each baseline's settings were chosen on the validation subjects (`data.roles.val: 0.1`), never on test:
  - **criterion:** the root scores each round's model on what the validation streams brought since the round before. These scores are pooled over the run per dataset (`heldout_pooled_macro_f1`), then averaged over SWELL and WESAD and over 3 seeds. The final model's validation macro-F1 is reported too, but it ties at the constant-predictor value for almost every configuration;
  - **step 1** (`continuum_benchmark_select`): lr ∈ {0.1, 0.03, 0.01, 0.003} × local epochs ∈ {1, 10}, for warm, frozen and scratch separately;
  - **step 2** (`…_select_replay`): memory ∈ {fifo, reservoir, class_balanced} at 512 × `replay_ratio` ∈ {0.25, 0.5} × lr ∈ {0.1, 0.03}, with step 1's warm trainer;
  - **step 3** (`…_select_unlabelled`): six `pseudo_label` settings (threshold 0.8 / 0.9 / 0.95 × weight 0.5 / 1.0) and `consistency` at noise 0.05 and 0.1, with step 2's choice.
- **Test.** The five chosen baselines on seeds 0–4 (`continuum_benchmark`), with the root scoring every round's global model on the test subjects.
- **Sweeps** (`continuum_benchmark_sweeps`, seeds 0–2, on test). One axis at a time around warm + replay, to describe it; nothing is chosen from them.
- **Process** (`continuum_benchmark_promotion`, seeds 0–2, on test). The four baselines that update, behind `champion_challenger` (C7) with its defaults and `min_samples: 20`. That value was set to fit the size of the validation streams, not tuned.

| Identity | Value |
|---|---|
| `topology_id` | `ff55fe4d2ce1a807f764bcc5205a8d2d4d43ab3d533163a1d13e173a9a6725fa` |
| `data_id` | `9fd573754157d68b111f037ecd891bdb62abcbff1d5e5e818da789dde470257c` (the same as C4's and C5's) |
| code, test, sweeps and promotion | commit `3bb359b`, clean tree |
| code, selection | step 1: `a910ce1` (warm, scratch) and `f16479a` (frozen, rerun after the fix below). Step 2: `5fd7dca`. Step 3: `e34e383`. All clean. The commits differ only in experiment files and in the frozen path, which none of the other runs takes |
| `config_id`, frozen | `71a64d2a1131c9dafac73211d8ee8a05704f867d895fd8bf703ec6ad7e75b4c5` |
| `config_id`, scratch | `531d1fb7bf6dbc13f77b837a3744d7a3a98f3789784022e42e93d1766caf72cc` |
| `config_id`, warm | `6eaf304368fd34406444222742b6fca18ff2a1d1eab68f7976e78259948b7ab6` |
| `config_id`, warm + replay | `34e3aa6c5884cb64511cdd50f808843c26ca83390ca29e192ceeace9eceedc57` |
| `config_id`, warm + replay + semi | `969087dbfd533a21263f258786d3f5332d3b7c85722925ad67be11461c951512` |
| `config_id`s of the selection, sweeps and promotion | one per configuration, in `selection.csv`, `sweeps.csv` and `promotion.csv` |

Every row of every CSV carries its run's `run_id`, `config_id`, `topology_id` and `run_hash`.

**The frozen baseline was run twice.** In the first version of `update: frozen`, an edge without labels in its history trained once its first labels arrived, in round 2 or 3. That is not v0. The code was fixed (`f16479a`), frozen's selection was rerun, and the test runs were made after that. Only the fixed runs are kept. With the fix, frozen's choice moved from lr 0.003 to lr 0.01 (both 1 epoch). The two are within 0.001 of each other on the criterion.

## Metrics

All of them come from `scripts/continuum_benchmark.py`, from each run's `events.jsonl`, `summary.json` and `model.npz`:

- **Prequential macro-F1.** Every arrival is scored with the model its edge was serving when it came, from the stored predictions, pooled over the run. `prequential_labelled` keeps only the rows whose label arrived.
- **Area under the curve.** The root's per-round macro-F1 on the test subjects, weighted by how long each round's model was active (in data time) and divided by the time spanned.
- **Final macro-F1.** The final `model.npz`, scored offline on every row of the test subjects.
- **Forgetting.** The best per-round test macro-F1 of an earlier round minus the last round's.
- **Recovery after a drift.** The first round whose arrivals change majority class (the hidden labels, so simulation only) is the drift. The level is the pooled prequential accuracy of the rounds before it. Recovery is the number of rounds, and data minutes, until a round's prequential accuracy is back within 0.1 of that level.
  - On SWELL the drift is each subject's switch from no stress to stress, in round 6.
  - On WESAD the stream changes class at t₀ itself (history: baseline; every later window: stress), so there is no earlier level and recovery is not defined.
- **Costs.** Rounds, rounds with any training at the root, bytes sent, trainings that worked, and *sample-epochs*: the rows each training went through (new, replayed or retrained) times its local epochs. The unlabelled pool's passes are not counted.
- **Process.** Candidates, promotions, rejections with their reason, and rollbacks.

## Selection (validation; mean ± 95% CI over 3 seeds; `selection.csv`)

Step 1, the top of each baseline and its runner-up (all 24 are in `selection.csv`):

| Baseline | lr, epochs | Criterion | SWELL | WESAD | Final model, mean of datasets |
|---|---|---|---|---|---|
| warm | **0.1, 10** | 0.567 ± 0.053 | 0.998 ± 0.008 | 0.136 ± 0.098 | 0.329 ± 0.000 |
| warm | 0.03, 10 | 0.492 ± 0.012 | 0.984 ± 0.023 | 0.000 ± 0.000 | 0.329 ± 0.000 |
| scratch | **0.1, 10** | 0.365 ± 0.148 | 0.729 ± 0.296 | 0.000 ± 0.000 | 0.329 ± 0.000 |
| scratch | 0.03, 10 | 0.348 ± 0.135 | 0.695 ± 0.271 | 0.000 ± 0.000 | 0.324 ± 0.021 |
| frozen | **0.01, 1** | 0.186 ± 0.080 | 0.363 ± 0.153 | 0.008 ± 0.035 | 0.389 ± 0.073 |
| frozen | 0.003, 1 | 0.185 ± 0.074 | 0.362 ± 0.145 | 0.008 ± 0.035 | 0.387 ± 0.068 |

Step 2: `replay_ratio` 0.25 at lr 0.1 scores 0.538 ± 0.108, against 0.479 (lr 0.03) and 0.461 (ratio 0.5). **The three memories tie exactly in every cell.** At a tenth of the labels a SWELL edge has about 13 labelled rows, so a capacity of 512 never binds, and the three keep the same rows. The tie was broken for `reservoir`, C4's memory.

Step 3: `consistency` at noise 0.1 scores 0.517 ± 0.146 and at 0.05 scores 0.503. The best `pseudo_label` (threshold 0.95, weight 0.5) scores 0.470. All of them are below warm + replay without an objective (0.538, step 2).

**What the selection can and can't tell.**
- Most gaps are within the intervals. The choice follows the point estimate, as planned.
- The criterion is favourable to quick switching. Each round's model scores validation rows that arrived while the train edges' labels from the same block (10 minutes late) were training it. That is why warm scores 0.998 on SWELL.
- WESAD contributes almost nothing. Its validation stream is stress-only after t₀, and the models rarely predict stress there.

## Results (test; mean ± 95% CI over 5 seeds; `run_metrics.csv`)

**Performance.**

| Baseline | Prequential SWELL | Prequential WESAD | AUC SWELL | AUC WESAD | Final SWELL | Final WESAD | Forgetting SWELL | Forgetting WESAD | Recovery, SWELL rounds (minutes) |
|---|---|---|---|---|---|---|---|---|---|
| frozen | 0.319 ± 0.115 | 0.033 ± 0.058 | 0.298 ± 0.087 | 0.446 ± 0.116 | 0.302 ± 0.092 | **0.463 ± 0.147** | 0.000 ± 0.000 | 0.000 ± 0.000 | never, in 5 of 5 seeds |
| scratch | 0.546 ± 0.096 | 0.000 ± 0.000 | 0.325 ± 0.025 | 0.365 ± 0.029 | 0.405 ± 0.003 | 0.262 ± 0.000 | 0.108 ± 0.061 | 0.318 ± 0.148 | 7.4 ± 2.3 (74 ± 23) |
| warm | **0.795 ± 0.018** | 0.000 ± 0.000 | 0.353 ± 0.012 | 0.324 ± 0.020 | 0.404 ± 0.000 | 0.262 ± 0.000 | 0.053 ± 0.069 | 0.405 ± 0.243 | **3.2 ± 0.6** (32 ± 6) |
| warm + replay | 0.724 ± 0.024 | 0.000 ± 0.000 | 0.356 ± 0.012 | 0.334 ± 0.020 | 0.411 ± 0.020 | 0.262 ± 0.000 | 0.066 ± 0.026 | 0.458 ± 0.124 | 4.2 ± 0.6 (42 ± 6) |
| warm + replay + semi | 0.728 ± 0.024 | 0.000 ± 0.000 | 0.353 ± 0.009 | 0.339 ± 0.013 | 0.372 ± 0.088 | 0.312 ± 0.140 | 0.111 ± 0.094 | 0.436 ± 0.187 | 4.2 ± 0.6 (42 ± 6) |

**What the final models predict** (`final_confusion`):
- Scratch, warm and warm + replay predict stress for every SWELL test row in 4 or 5 of the 5 seeds.
- Warm + replay + semi-supervised does too in 4 seeds, and predicts no stress for every row in seed 1.
- On WESAD, every updating baseline ends at stress for every row, except warm + replay + semi-supervised in seed 1, which mostly predicts no stress.
- The frozen v0 mostly predicts no stress, on both datasets. On WESAD it predicts no stress for every row in 3 of the 5 seeds, and both classes in the other 2. No stress is the minority of SWELL's test rows (203 of 628) and the majority of WESAD's (111 of 172), which is why frozen scores worst on SWELL and best on WESAD.

**Costs** (per run; WESAD's are the same runs):

| Baseline | Rounds | Rounds with training | Trainings | Bytes | Sample-epochs |
|---|---|---|---|---|---|
| frozen | 17.6 ± 0.7 | 1 | 19.6 ± 2.6 | 142 MB | 35 ± 4 |
| scratch | 17.6 ± 0.7 | 17.6 ± 0.7 | 191.4 ± 4.2 | 182 MB | 12,600 ± 410 |
| warm | 17.6 ± 0.7 | 17.6 ± 0.7 | 191.8 ± 4.3 | 183 MB | 2,890 ± 0 |
| warm + replay | 17.6 ± 0.7 | 17.6 ± 0.7 | 191.6 ± 4.4 | 182 MB | 3,470 ± 54 |
| warm + replay + semi | 17.6 ± 0.7 | 17.6 ± 0.7 | 190.8 ± 3.0 | 180 MB | 3,470 ± 23 |

Frozen still sends most of the bytes: the rounds stay open, and the model goes down to every edge in each of them. Only the updates going up are saved.

**Per round, SWELL** (`curves.csv`; mean over seeds of the pooled prequential accuracy of each round's arrivals, rounds 2–12):

| Baseline | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen | 0.89 | 0.89 | 0.90 | 0.90 | 0.01 | 0.09 | 0.16 | 0.14 | 0.12 | 0.13 | 0.11 |
| scratch | 0.89 | 0.99 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.09 | 0.34 | 0.37 |
| warm | 0.89 | 0.99 | 1.00 | 1.00 | 0.00 | 0.00 | 0.15 | 0.95 | 1.00 | 1.00 | 1.00 |
| warm + replay | 0.89 | 0.99 | 1.00 | 1.00 | 0.00 | 0.00 | 0.01 | 0.27 | 0.94 | 1.00 | 1.00 |
| warm + replay + semi | 0.89 | 0.99 | 1.00 | 1.00 | 0.00 | 0.00 | 0.01 | 0.32 | 0.96 | 1.00 | 1.00 |

On WESAD, the per-round test macro-F1 of warm and its variants peaks once, in round 6, at 0.64–0.75, and falls to 0.26 (stress for every row) by round 8. Scratch peaks later and lower: 0.47 in round 9, down to 0.26 by round 14. That fall is what the WESAD forgetting measures.

**Checks:**
- **Determinism** (`determinism.csv`): each of the 25 test runs, run twice, gives the same final model and the same events. The sweeps and promotion runs were not rerun.
- `verify_run` passed on the 211 original run folders: 132 for the selection, 25 for the test runs (plus their 25 reruns), 42 for the sweeps and 12 for promotion.
- **Selection and test do not share subjects.** The test runs score only the test subjects. The selection runs score only the validation streams and never evaluate a test subject.

## Sweeps (test; around warm + replay; mean ± 95% CI over 3 seeds; `sweeps.csv`)

**Labels.** The **label efficiency** is read off this table: how the scores move with the labelled fraction.

| Labels (fraction / delay) | Prequential SWELL | Final SWELL | Final WESAD | Forgetting WESAD | Sample-epochs |
|---|---|---|---|---|---|
| 0.1 / 10 min | 0.723 ± 0.069 | 0.415 ± 0.051 | 0.262 ± 0.000 | 0.422 ± 0.159 | 3,480 |
| 0.25 / 10 min | 0.708 ± 0.057 | 0.441 ± 0.161 | 0.359 ± 0.420 | 0.362 ± 0.332 | 9,160 |
| 0.5 / 10 min | 0.699 ± 0.050 | 0.404 ± 0.000 | 0.262 ± 0.000 | 0.454 ± 0.207 | 18,400 |
| 1.0 / 10 min | 0.629 ± 0.347 | 0.351 ± 0.229 | 0.305 ± 0.187 | 0.406 ± 0.147 | 33,300 |
| 0.1 / 0 | 0.827 ± 0.050 | 0.413 ± 0.039 | 0.262 ± 0.000 | 0.444 ± 0.205 | 3,400 |
| 0.1 / 30 min | 0.525 ± 0.063 | 0.419 ± 0.067 | 0.262 ± 0.000 | 0.409 ± 0.149 | 3,490 |

- **More labels do not buy a better model here.** From a tenth to all of them, the compute grows almost tenfold and neither score rises. The prequential score even falls a little, with a wide interval at 1.0.
- **The delay is what matters.** Without delay, the SWELL prequential score rises from 0.723 to 0.827. With 30 minutes of delay, it falls to 0.525: the model follows the switch only once the labels show it.

**Memory** (at a tenth of the labels, 10 minutes late): fifo, reservoir and class_balanced at 4 and at 16 rows score the same prequential macro-F1 as the 512-row reservoir to the third decimal (0.723), with final SWELL between 0.404 and 0.415.
- Capacity 4 binds, but each training replays r·n / (1 − r) rows. With n at one or two new labelled rows and r = 0.25, that is at most one row, so the memory barely acts.
- In this setup the memory axis cannot be told apart. This is a property of the label budget, not of the memories.

**Trigger:**

| Trigger | Rounds | Bytes | Prequential SWELL | Final SWELL | Forgetting WESAD | Recovery SWELL (rounds) |
|---|---|---|---|---|---|---|
| schedule, 10 min | 18 | 186 MB | 0.723 ± 0.069 | 0.415 ± 0.051 | 0.422 ± 0.159 | 4.3 ± 1.4 |
| volume, 30 rows | 8 | 92 MB | 0.471 ± 0.158 | 0.404 ± 0.000 | 0.381 ± 0.131 | 3.0 ± 0.0 |
| drift (prior) | 6.3 ± 1.4 | 71 MB | 0.693 ± 0.139 | 0.404 ± 0.000 | 0.130 ± 0.000 | 2.0 ± 0.0 |

- **The prior-drift trigger is the cheapest.** It keeps most of the schedule's prequential score at 38% of its bytes, and forgets less of WESAD.
- **Its recovery is counted in its own rounds,** so its 2 rounds and the schedule's 4.3 are not the same time.
- The volume and the detector were not tuned. They are C6's, with the volume scaled to the label fraction.

## Process: champion and challenger (test; 3 seeds; `promotion.csv`, `promotion_decisions.csv`)

| Baseline | Candidates | Promoted | Rejected: min_samples / min_delta / max_forgetting | Champion at the end | Prequential SWELL | Final SWELL | Final WESAD |
|---|---|---|---|---|---|---|---|
| scratch | 52 | 2 | 42 / 8 / 0 | v8 in 2 seeds, v0 in 1 | 0.309 ± 0.226 | 0.275 ± 0.130 | 0.427 ± 0.148 |
| warm | 54 | 0 | 23 / 12 / 19 | v0 | 0.372 ± 0.226 | 0.344 ± 0.211 | 0.521 ± 0.357 |
| warm + replay | 53 | 0 | 23 / 12 / 18 | v0 | 0.372 ± 0.226 | 0.344 ± 0.211 | 0.521 ± 0.357 |
| warm + replay + semi | 47 | 0 | 23 / 9 / 15 | v0 | 0.372 ± 0.226 | 0.344 ± 0.211 | 0.521 ± 0.357 |

- **The policy almost never promotes**, and version 0 is the *initial* weights. So, behind this promotion, warm and its variants serve and save the untrained model for the whole run. That is why the three have the same scores: they serve the same initial model.
- **Round 1's challenger** (the v0 that frozen keeps) is always rejected for `min_samples`, since no validation label has arrived by then.
- **Later challengers** fail `min_delta` or `max_forgetting` on the switch.
- **This is a finding about the setup, not a tuned result.** With validation streams this small, a tenth labelled, and no promotion of a first trained model, the champion/challenger defaults do not fit. A policy that promotes the first trained model, or bigger validation streams, would be needed.

## Reading

- **Adaptation.** Every updating baseline beats frozen on SWELL's prequential score: 0.55–0.80 against 0.32. The ranking follows the speed of the switch after round 6: warm recovers in about 3 rounds, replay in 4, and scratch in 7.
  - Retraining from scratch is no ceiling here. Each training runs 10 epochs from the initial weights on every labelled row so far, at most about 13 per SWELL edge and from both blocks. It is the slowest to switch, and it costs 4.4× warm's compute.
  - The prequential score rewards following the block. It does not measure telling the classes apart.
- **Retention.** No baseline retains what it learnt.
  - On WESAD, the updating baselines forget the no-stress class within a few rounds of their peak (forgetting 0.32–0.46), since every WESAD label after t₀ is stress. Frozen never trains on that stream, so it has nothing to forget. Its better WESAD test score (0.463) comes from a v0 that leans to no stress, WESAD's test majority, not from telling the classes apart.
  - On SWELL, every final model is constant, so its forgetting (0.05–0.11) is the drop from a transient round to the constant.
- **Memory and unlabelled rows.**
  - At a tenth of the labels, replay has almost nothing to replay, so it cannot help against forgetting. It slows the switch a little instead (SWELL prequential 0.795 → 0.724).
  - The consistency term changes nothing measurable (0.728).
  - Neither result says how these mechanisms behave with label budgets where replay is large. C4's runs, at full labels, saw replay diverge at lr 0.1.
- **What would change the picture:**
  - streams with mixed classes per round, so that the learner sees both classes, for example `staggered` starts or shuffled blocks;
  - a replay budget that does not depend on the new labelled rows;
  - a promotion policy that can promote the first trained model;
  - more seeds and validation subjects. With 2 SWELL and 1–2 WESAD validation subjects, the selection intervals are wide.

## What was cut or not run

- **Seeds.** The test runs use 5 seeds; the selection, sweeps and promotion use 3.
- **Sweeps.** They are one axis at a time around warm + replay, not a full grid, and they were not rerun for determinism.
- **Selection.** Only the settings listed above were searched: no batch size, no model width, no other optimizer. The memory capacity in the selection was fixed at 512.
- **Drift types.** Concept drift and recovery on WESAD are not measured: WESAD's stream changes at t₀ (see Metrics).
- **Runtime.** Everything runs in simulation (`SimRuntime`); nothing ran on real nodes (C9).

## Files

- `<run_id>/`: `run.json` and `summary.json` of the 211 cited runs (132 selection, 25 test, 42 sweep and 12 promotion runs), and `model.npz` of the 25 test runs only. The other runs' final models are not committed, to keep the repository small: their numbers are in the CSVs, and each run can be repeated from its `config_id` at its commit (the 25 test runs reran identically). The 25 determinism reruns stay in `runs/`.
- `selection.csv`: per selection run (`step` 1–3) and dataset, every metric on the validation subjects.
- `run_metrics.csv`: per test run and dataset, every metric above, the final model's confusion matrix and the run's identity.
- `curves.csv`: per test run, dataset and round, the data time, the arrivals, the pooled prequential accuracy and the root's test macro-F1.
- `sweeps.csv` and `promotion.csv`: as `run_metrics.csv`, for the sweeps and the promotion runs.
- `promotion_decisions.csv`: per promotion run, its candidates, promotions, rejections by reason, rollbacks and final champion.
- `determinism.csv`: each test run against its rerun.

`events.jsonl` and `bundle/` are left out, and `host` and `pid` are removed from `run.json`. As a result, `verify_run` cannot check these copies. It passed on the original run folders.
