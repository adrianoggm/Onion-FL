"""Tests for per-node randomness and the Node base class (issue #79)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from onion_fl.core.context import child_rng, node_rng
from onion_fl.core.message import Message
from onion_fl.core.node import Node

SRC = Path(__file__).resolve().parents[1] / "src"


def _draws(seed: int, node_id: str) -> list[int]:
    return node_rng(seed, node_id).integers(0, 2**31, size=5).tolist()


def test_same_seed_and_node_give_the_same_stream() -> None:
    assert _draws(7, "fog_0") == _draws(7, "fog_0")


def test_each_node_gets_its_own_stream() -> None:
    assert _draws(7, "fog_0") != _draws(7, "fog_1")


def test_the_experiment_seed_changes_every_stream() -> None:
    assert _draws(7, "fog_0") != _draws(8, "fog_0")


def test_streams_do_not_depend_on_python_hash_randomisation() -> None:
    code = "from onion_fl.core.context import node_rng; print(node_rng(7, 'edge_3').integers(0, 2**31, size=5).tolist())"
    outputs = set()
    for hash_seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(SRC)}
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outputs.add(result.stdout.strip())

    assert outputs == {str(_draws(7, "edge_3"))}


def test_node_rng_returns_a_numpy_generator() -> None:
    assert isinstance(node_rng(0, "cloud"), np.random.Generator)


def test_node_handlers_default_to_doing_nothing() -> None:
    node = Node("edge_1")
    msg = Message(kind="hello", src="fog_0", dst="edge_1")

    assert node.id == "edge_1"
    assert node.on_start(ctx=None) is None  # type: ignore[arg-type]
    assert node.on_message(msg, ctx=None) is None  # type: ignore[arg-type]
    assert node.on_timer("deadline", ctx=None) is None  # type: ignore[arg-type]


def test_a_child_stream_leaves_its_parent_untouched() -> None:
    parent, twin = node_rng(3, "e1"), node_rng(3, "e1")

    child = child_rng(parent)
    child.integers(0, 2**31, size=10)

    assert (
        parent.integers(0, 2**31, size=5).tolist()
        == twin.integers(0, 2**31, size=5).tolist()
    )


def test_successive_child_streams_differ_and_repeat_per_seed() -> None:
    a, b = node_rng(3, "e1"), node_rng(3, "e1")

    first, second = (
        child_rng(a).integers(0, 2**31, 3),
        child_rng(a).integers(0, 2**31, 3),
    )

    assert first.tolist() != second.tolist()
    assert child_rng(b).integers(0, 2**31, 3).tolist() == first.tolist()
