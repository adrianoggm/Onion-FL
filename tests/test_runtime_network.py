"""Tests for link profiles and the simulated link channel (issue #81)."""

from __future__ import annotations

import numpy as np
import pytest

from onion_fl.core.registry import PluginError
from onion_fl.runtime.network import (
    LinkChannel,
    LinkProfile,
    link_profiles,
    resolve_profile,
)


def _channel(profile: LinkProfile, direction: str = "up", seed: int = 0) -> LinkChannel:
    return LinkChannel(profile, direction, np.random.default_rng(seed))


def test_presets_are_registered() -> None:
    assert link_profiles.names() == ["4g", "lan", "lora", "wifi"]


def test_profile_from_a_preset_name() -> None:
    profile = resolve_profile("4g")

    assert profile.bandwidth_up_bps < profile.bandwidth_down_bps
    assert profile.distribution == "lognormal"


def test_preset_with_overrides() -> None:
    base = resolve_profile("4g")
    profile = resolve_profile({"preset": "4g", "loss": 0.2})

    assert profile.loss == 0.2
    assert profile.latency_s == base.latency_s


def test_profile_from_a_full_mapping() -> None:
    profile = resolve_profile({"latency_s": 0.01, "bandwidth_up_bps": 1e6})

    assert profile.latency_s == 0.01
    assert profile.bandwidth_down_bps is None


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(PluginError, match="lan"):
        resolve_profile("5g")


@pytest.mark.parametrize(
    "fields",
    [
        {"loss": 1.5},
        {"latency_s": -1},
        {"bandwidth_up_bps": 0},
        {"distribution": "pareto"},
    ],
)
def test_invalid_profile_values_are_rejected(fields: dict) -> None:
    with pytest.raises(ValueError):
        LinkProfile(**fields)


def test_delivery_is_transmission_plus_latency_in_each_direction() -> None:
    profile = LinkProfile(
        latency_s=0.1, bandwidth_up_bps=8_000, bandwidth_down_bps=80_000
    )

    assert _channel(profile, "up").schedule(0.0, 500) == pytest.approx(0.6)
    assert _channel(profile, "down").schedule(0.0, 500) == pytest.approx(0.15)


def test_unlimited_bandwidth_adds_no_transmission_time() -> None:
    channel = _channel(LinkProfile(latency_s=0.25))

    assert channel.schedule(3.0, 10**9) == pytest.approx(3.25)


def test_back_to_back_messages_queue_on_the_link() -> None:
    channel = _channel(LinkProfile(latency_s=0.1, bandwidth_up_bps=8_000))

    first = channel.schedule(0.0, 1_000)
    second = channel.schedule(0.0, 1_000)

    assert first == pytest.approx(1.1)
    assert second == pytest.approx(2.1)


def test_arrivals_keep_fifo_order_despite_jitter() -> None:
    channel = _channel(LinkProfile(latency_s=0.05, jitter_s=0.5, distribution="normal"))

    arrivals = [channel.schedule(t * 0.01, 10) for t in range(200)]

    assert arrivals == sorted(arrivals)


@pytest.mark.parametrize("distribution", ["normal", "lognormal"])
def test_jitter_never_gives_a_negative_latency(distribution: str) -> None:
    channel = _channel(
        LinkProfile(latency_s=0.001, jitter_s=1.0, distribution=distribution)
    )

    for t in range(300):
        assert channel.schedule(float(t), 0) >= t


def test_fixed_distribution_ignores_jitter() -> None:
    channel = _channel(LinkProfile(latency_s=0.2, jitter_s=5.0, distribution="fixed"))

    assert channel.schedule(0.0, 0) == pytest.approx(0.2)


def test_loss_of_one_drops_everything_and_zero_drops_nothing() -> None:
    lossy = _channel(LinkProfile(loss=1.0))
    clean = _channel(LinkProfile(loss=0.0))

    assert all(lossy.schedule(float(t), 10) is None for t in range(50))
    assert all(clean.schedule(float(t), 10) is not None for t in range(50))


def _drop_pattern(seed: int) -> list[bool]:
    channel = _channel(
        LinkProfile(loss=0.3, latency_s=0.01, jitter_s=0.01, distribution="normal"),
        seed=seed,
    )
    return [channel.schedule(float(t), 10) is None for t in range(200)]


def test_losses_are_reproducible_with_the_same_seed() -> None:
    assert _drop_pattern(3) == _drop_pattern(3)
    assert _drop_pattern(3) != _drop_pattern(4)
    assert 20 < sum(_drop_pattern(3)) < 100
