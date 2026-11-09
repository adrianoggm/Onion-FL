# Fidelity of the simulation against the emulation, validated (issue #198)

**Status: pre-registered; calibration closed and cited; running.**

`experiments/fidelity_sim_emulated.yaml` validates the `sim` backend (B) against the `emulated` backend (A) at the three levels of I1 §12: functional, performance and decision. It uses the calibration of #197, closed and cited, on loads that calibration never saw. The hypotheses, the criteria and the analysis are fixed in [PREREGISTRATION.md](PREREGISTRATION.md), written before any run.

## The calibration cited

`b55104b3f71d684dedc413a000eec3a67eb1caba134cc9ff6a95648fb35c8dfc`: the refit of #197 on four sessions with pinned CPUs (`results/calibration_emulated/`), with image `onionfl-a197b-emulated:latest` (`sha256:a826e66e…`) and `valid_up_to` 10 000 bytes. It replaced the provisional single-session `5d7d0e70…` on 2026-11-09, before any run, following step 1 of the pre-registration (§2). The harness's `--calibration` refuses any run that cites another.

### The machine on the day of the runs, recorded before running

The pre-registration handles drift within a session (seed-by-seed order, pairing by seed). It does not address a change of the machine between the calibration's fit and the validation, so this is recorded here before any run, and the hypotheses are not changed:

- The four fit sessions ran on 2026-11-06, with Docker Desktop's Kubernetes active.
- The calibration's checks on 2026-11-09 ran on a quieter machine (Kubernetes scaled to 0). It trained 1.6–2 times faster than the fit, and one check session's aggregation came out 8 % above the interval.
- The validation runs on 2026-11-09 on that quieter machine. No other agent is running and Kubernetes is scaled to 0. Three idle containers of other projects remain (`fg-obs-grafana-1`, `redsaras-api`, `postgres`) and are left alone.
- **So the emulated compute may be faster than the calibration predicts.** The simulation would then be the slower one. The compute errors (H2a) and the round-time errors (H2b), (B − A) / A, may then be positive, by about +0.6 to +1.0 if the whole 1.6–2× shows up, for a reason that is the machine's and not the simulator's.
  - H2a's band, from the refit's prediction interval, is wide: [−0.803, +4.076]. A miss of that size still falls inside it.
## Setup

- SWELL physiology, the calibrated workload (`modular_mlp` 32/[16], batch 32), `npz` codec, 10 rounds, 5 seeds.
- Infrastructure `infrastructures/fidelity_validation_lab.yaml`: the calibrated lab with the 5 devices the calibration measured, each container pinned to its session's CPUs.
- Six configurations:
  - every edge under the cloud, or under one fog on `fog-a` or on `fog-b`;
  - times 1 or 5 local epochs.

The pre-registration (§3) gives the full design and how it differs from the calibration and from #195.

## How to run it (once the calibration is closed)

```bash
# 0. Cite the closed calibration (pre-registration §2) and check it.
python -m pytest tests/test_fidelity_validation.py -q
onion_fl plan experiments/fidelity_sim_emulated.yaml > /dev/null
docker ps                       # no other project's containers

# 1. Emulated: 30 runs, one at a time, seed by seed; it resumes if interrupted.
python scripts/run_fidelity_validation.py --backend emulated --calibration <id>
# 2. Simulation: the same 30 configurations and seeds.
python scripts/run_fidelity_validation.py --backend sim --calibration <id> --workers 3
# 3. Analysis (pre-registration §4).
onion_fl fidelity runs/fidelity_validation_emulated runs/fidelity_validation_sim \
  --calibration <id> --metric time --metric bytes --metric macro_f1 --metric loss \
  --series global/swell --csv fidelity.csv --html fidelity.html > fidelity.json
```

## Files (once run)

- `runs_emulated/<run_id>/` and `runs_sim/<run_id>/`: `run.json`, `summary.json` and `model.npz` of each run.
- `fidelity.json`, `fidelity.csv` and `fidelity.html`.
- `PREREGISTRATION.md`.

## Deviations from the pre-registration

None yet.
