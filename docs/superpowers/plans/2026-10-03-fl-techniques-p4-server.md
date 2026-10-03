# FL techniques P4: server optimizers and asynchrony

**Goal:** add `fedyogi`, `fedadagrad` and `fedasync` (issue #150), the `staleness` round statistic they need, FedBuff as a documented combination, and the real-data comparison `experiments/techniques_server.yaml`.

**Architecture:**
- **Optimizers.** The adaptive server optimizers share one base with FedAdam and differ only in their second moment. `fedasync` mixes the aggregate into the global model with a staleness-discounted weight.
- **Staleness.** Each collector computes the mean staleness of what it combines and sends it up, so the root's optimizer sees the staleness of the updates inside the aggregate it applies.
- **FedBuff.** It needs an aggregator that closes a round as soon as K updates arrived (a new round setting, `close_at_quorum`); late updates then join the next round through `staleness: next_round`.

**Spec:** `docs/superpowers/specs/2026-10-03-fl-techniques-design.md` (§3.2 round statistics, §4 P4).

## Global constraints

- **Plugin texts.** Titles, descriptions and explain texts in Spanish; code, comments and docs in English.
- **Experiments.**
  - No synthetic data for ML; protocol tests use the `stub` trainer.
  - Every hyperparameter is chosen on the validation subjects, never on test scores.
  - Comparisons run over lossless links, so that lost rounds do not mix with the technique.
- **Results.** Results cite committed CSVs from runs at a clean commit.
- **Commits.** `type(scope): Imperative summary`, with no `Co-Authored-By`.

## Tasks

### Task 1: the `staleness` round statistic
- A collector's `staleness` is the mean, over the contributions it combines, of each contribution's age at this node (0 if fresh, `round − its round` if buffered) plus the staleness its sender reported.
- It travels up in the update's metrics as `staleness`, and the coordinator passes it to the server optimizer in `stats`.
- **Tests** (`fog_by_hand`):
  - a round with one fresh and one buffered update reports 0.5;
  - a child's reported staleness adds to its age;
  - the coordinator's stats carry it.

### Task 2: close a round at quorum (`close_at_quorum`)
- With `close_at_quorum: true`, a collector closes as soon as `quorum_needed(quorum, participants)` non-empty updates have arrived, instead of waiting for all of them or the deadline.
- Updates that arrive later are late: dropped, or buffered by `staleness: next_round`.
- **Tests:**
  - with quorum 1 of 2, the round closes on the first update;
  - the second update is buffered into the next round.

### Task 3: `fedyogi` and `fedadagrad`
- Reddi et al. (2021), Algorithm 2, on the pseudo-gradient Δ = x̄ − x:
  - m ← β1·m + (1 − β1)·Δ, and x ← x + η·m / (√v + τ);
  - FedAdagrad: v ← v + Δ²;
  - FedYogi: v ← v − (1 − β2)·Δ²·sign(v − Δ²);
  - v starts at τ², as in the paper.
- Auxiliary arrays are replaced, never stepped.
- **Tests:**
  - one and two hand-computed steps of each optimizer;
  - Yogi's v grows more slowly than Adam's when Δ² jumps;
  - auxiliary arrays are replaced;
  - dtype is kept;
  - the registry lists both.

### Task 4: `fedasync`
- x ← (1 − α_s)·x + α_s·x̄, with α_s = α·(1 + s)^(−a), where s is `stats["staleness"]` (0 when absent). This is Xie et al. (2019) applied once per round to the buffered aggregate.
- **Tests:**
  - α_s with s = 0 and with s = 3;
  - auxiliary arrays are replaced.

### Task 5: experiments and results
- **`experiments/fedbuff.yaml`.** The documented combination: fogs with `quorum: K` (an integer), `close_at_quorum: true` and `staleness: {name: next_round, weighting: {name: polynomial, a: 0.5}}`.
- **`experiments/techniques_server_select.yaml`.** Server learning rates and FedAsync's α, chosen on validation (zone evaluators, round 20, seed 0).
- **`experiments/techniques_server.yaml`.**
  - Scenarios: FedAvg (`replace`), FedAvgM, FedAdam, FedYogi, FedAdagrad, FedAsync, and FedBuff (a topology whose fogs buffer).
  - Setup: SWELL + WESAD, segregated placement, lossless links, SGD at lr 0.1 (FedAvg's rate on validation in the drift comparison), 3 seeds.
- **Results.** `results/techniques_server/`: the INDEX with identity, run metrics, simulated time to finish, and the report. The README section, and the architecture and spec rows.

### Task 6: final review and PR
- A fresh review of the whole branch.
- One fix pass, test-first.
- The full suite.
- A PR into `develop`, left for the user's review.
