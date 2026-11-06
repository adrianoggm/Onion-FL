# Fidelity of the simulation against the emulation, validated (issue #198)

**Status: pre-registered, not run.** No result is in this folder yet.

`experiments/fidelity_sim_emulated.yaml` validates the `sim` backend (B) against the `emulated` backend (A) at the three levels of I1 §12: functional, performance and decision. It uses the calibration of #197, closed and cited, on loads that calibration never saw. The hypotheses, the criteria and the analysis are fixed in [PREREGISTRATION.md](PREREGISTRATION.md), written before any run.

## The calibration cited

**Provisional:** `5d7d0e7011bf1fd693725bcb85ac61c5abae3d460867169f8909954c1b937cde`, the single-session calibration of #197 (`results/calibration_emulated/`). It will be replaced by the multi-session refit before any run, following step 1 of the pre-registration (§2). The final id goes here, and the harness's `--calibration` refuses any run that cites another.

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
