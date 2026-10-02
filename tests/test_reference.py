"""The SWELL reference experiment reproduces the pre-redesign setup (issue #101).

The layout is checked without data against the old reference config
(``configs/swell_federated_10runs.yaml``, removed in F9.1); running it needs
``data/SWELL`` and is skipped without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from onion_fl.data.ingest import check_spec, load_spec
from onion_fl.experiment.config import load_experiment

# From configs/swell_federated_10runs.yaml (manual_assignments minus test_assignments).
REFERENCE_FOGS = {
    "fog_0": [1, 2, 3, 4, 5, 6],
    "fog_1": [7, 9, 10, 12, 13, 14, 22],
    "fog_2": [15, 16, 17, 18, 21],
}
REFERENCE_TEST = [19, 20, 23, 24, 25]


def config():
    return load_experiment("experiments/swell_reference.yaml")


def test_the_reference_keeps_the_same_subjects_per_fog_and_in_test() -> None:
    reference = config()
    overrides = reference.data.roles.overrides["swell"]
    assignment = reference.data.placement["assignment"]

    assert [int(s) for s in overrides.test] == REFERENCE_TEST
    assert {
        fog: [int(c.split("-")[1]) for c in ids] for fog, ids in assignment.items()
    } == REFERENCE_FOGS
    used = {s for fog in REFERENCE_FOGS.values() for s in fog} | set(REFERENCE_TEST)
    assert sorted(used | {int(s) for s in overrides.exclude}) == list(range(1, 26))


def test_the_reference_trains_like_the_old_runs() -> None:
    reference = config()

    assert reference.rounds == 12 and len(reference.seeds) == 10
    assert reference.learning.trainer == {
        "name": "standard",
        "local_epochs": 14,
        "lr": 0.001,
        "batch_size": 32,
    }
    assert reference.learning.model[
        "adapter_width"
    ] == 128 and reference.learning.model["trunk_hidden"] == [64]
    assert reference.data.roles.scaler == "global"


def test_the_physiology_descriptor_is_valid() -> None:
    spec = load_spec("datasets/swell_physiology.yaml")

    assert spec.name == "swell"
    check_spec(spec)


@pytest.mark.skipif(not Path("data/SWELL").exists(), reason="data/SWELL not available")
def test_the_reference_plan_on_real_data() -> None:
    from onion_fl.experiment.runner import plan

    preview = plan(config())[0]

    assert preview["roles"]["swell"]["test"] == [str(s) for s in REFERENCE_TEST]
    assert {leaf: c["subjects"] for leaf, c in preview["composition"].items()} == {
        fog: len(subjects) for fog, subjects in REFERENCE_FOGS.items()
    }
