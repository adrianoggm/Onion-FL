"""The shipped dataset descriptors (issue #87).

The descriptors are checked without data. Parity with the previous loaders
needs the real datasets under ``data/`` and is skipped when they are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from onion_fl.data.ingest import check_spec, ingest, load_spec

DESCRIPTORS = Path("datasets")


def descriptor(name: str):
    return load_spec(DESCRIPTORS / f"{name}.yaml")


@pytest.mark.parametrize("name", ["swell", "sweet", "wesad"])
def test_descriptors_are_valid(name: str) -> None:
    spec = descriptor(name)

    assert spec.name == name
    check_spec(spec)


@pytest.mark.parametrize(
    "name, options",
    [
        ("swell", {"facial": True, "posture": True, "physiology": True}),
        ("swell", {"label": "binary_no_r"}),
        ("sweet", {"label": "ordinal", "selection": "selection2/users"}),
        ("sweet", {"label": "three_class"}),
        ("wesad", {"location": "chest", "signals": "ECG,EDA", "label": "three_class"}),
    ],
)
def test_descriptor_options_are_valid(name: str, options: dict) -> None:
    check_spec(descriptor(name), options)


def _same_where_present(new, old_X: np.ndarray, old_names: list[str]) -> None:
    for j, name in enumerate(new.feature_names):
        if name not in old_names:
            continue
        values, legacy = new.X[:, j], old_X[:, old_names.index(name)]
        present = ~np.isnan(values)  # the old loaders imputed over the whole dataset
        np.testing.assert_allclose(
            values[present], legacy[present], rtol=1e-5, err_msg=name
        )


@pytest.mark.skipif(not Path("data/SWELL").exists(), reason="data/SWELL not available")
def test_swell_reads_like_the_previous_loader() -> None:
    from onion_fl.datasets.swell import load_swell_all_samples

    X, y, subject_ids, info = load_swell_all_samples(modalities=["computer"])
    new = {s.subject: s for s in ingest(descriptor("swell"))}

    assert set(new) == {str(s) for s in np.unique(subject_ids)}
    for subject, data in new.items():
        mask = subject_ids.astype(str) == subject
        np.testing.assert_array_equal(data.y, y[mask])
        _same_where_present(data, X[mask], list(info["feature_names"]))


@pytest.mark.skipif(
    not Path("data/SWEET/sample_subjects").exists(), reason="SWEET sample not available"
)
def test_sweet_reads_like_the_previous_loader() -> None:
    from onion_fl.datasets.sweet_samples import load_sweet_sample_full

    X, y, groups, names = load_sweet_sample_full()
    new = {s.subject: s for s in ingest(descriptor("sweet"))}

    assert set(new) == set(np.unique(groups))
    for subject, data in new.items():
        mask = groups == subject
        assert data.n_samples == int(mask.sum()), subject
        np.testing.assert_array_equal(np.sort(data.y), np.sort(y[mask]))


@pytest.mark.skipif(not Path("data/WESAD").exists(), reason="data/WESAD not available")
def test_wesad_chest_reads_like_the_previous_loader() -> None:
    from onion_fl.datasets.wesad import _load_subject_data

    X, y, names = _load_subject_data(
        Path("data/WESAD"),
        "S2",
        ["ECG", "EDA", "RESP"],
        "chest",
        ["baseline", "stress"],
        60,
        0.5,
    )
    subjects = ingest(
        descriptor("wesad"), {"location": "chest", "signals": "ECG,EDA,RESP"}
    )
    new = {s.subject: s for s in subjects}["S2"]

    assert new.feature_names == names
    np.testing.assert_array_equal(new.y, y)
    np.testing.assert_allclose(new.X, X, rtol=1e-5)
