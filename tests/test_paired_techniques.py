"""Tests for the paired technique contrasts (issue #293).

Hand-made configs and score tables only: nothing is trained, nothing reads data.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paired_techniques", ROOT / "scripts" / "paired_techniques.py"
)
paired = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paired)


def test_arms_are_named_from_the_config() -> None:
    lossless = "four_fogs_swell_wesad_lossless"
    assert (
        paired.server(
            {"topology": lossless, "learning": {"server_optimizer": "replace"}}
        )
        == "replace"
    )
    buffered = {
        "topology": f"{lossless}_fedbuff",
        "learning": {"server_optimizer": {"name": "fedasync_mix"}},
    }
    assert paired.server(buffered) == "fedasync_mix+fedbuff"
    ditto = {"learning": {"sharing": "fedavg", "trainer": {"name": "ditto"}}}
    assert paired.personalisation(ditto) == "ditto"
    fedper = {
        "learning": {"sharing": {"name": "fedper"}, "trainer": {"name": "standard"}}
    }
    assert paired.personalisation(fedper) == "fedper"
    assert paired.placement({"data": {"placement": {"alpha": 0}}}) == "alpha=0"


def row(arm: str, seed: int, series: str, score: float) -> dict:
    return {
        "experiment": "techniques_personalisation",
        "arm": arm,
        "seed": seed,
        "series": series,
        "macro_f1": score,
    }


def test_arms_are_paired_by_seed_and_personal_models_meet_fedavgs_trained_one() -> None:
    rows = []
    for seed in range(10):
        rows += [
            row("fedavg", seed, "global/swell", 0.5 + seed / 100),
            row("ditto", seed, "global/swell", 0.6 + seed / 100),  # +0.1 every seed
            row("apfl", seed, "global/swell", 0.5 + seed / 100),  # identical
            row("fedavg", seed, "local/*", 0.4),
            row("apfl", seed, "personal/*", 0.4 + (0.01 if seed % 2 else -0.01)),
        ]
    out = {(r["series"], r["arm"]): r for r in paired.contrasts(rows)}

    ditto = out[("global/swell", "ditto")]
    assert ditto["n_pairs"] == 10 and abs(ditto["delta"] - 0.1) < 1e-12
    assert ditto["verdict"] == "different" and ditto["p_adj"] < 0.05
    assert out[("global/swell", "apfl")]["verdict"] == "equivalent"
    assert ("global/swell", "fedavg") not in out  # the reference is not an arm
    personal = out[("personal/*", "apfl")]
    assert personal["reference_mean"] == 0.4 and abs(personal["delta"]) < 1e-12
