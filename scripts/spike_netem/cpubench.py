"""I9 spike: training throughput of a small MLP under a CPU limit.

Runs inside a spike container (``python3 - < cpubench.py [workers] [seconds]``)
and prints one JSON line: samples per second summed over ``workers`` processes,
each training a 64-128-2 MLP with plain SGD on one BLAS thread. The data is a
fixed random tensor: this measures CPU time, not learning.
"""

from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

BATCH = 64


def train(seconds: float) -> int:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((BATCH, 64))
    y = np.eye(2)[rng.integers(0, 2, BATCH)]
    w1, w2 = rng.standard_normal((64, 128)) * 0.1, rng.standard_normal((128, 2)) * 0.1
    done, end = 0, time.perf_counter() + seconds
    while time.perf_counter() < end:
        h = np.maximum(x @ w1, 0)
        z = h @ w2
        p = np.exp(z - z.max(1, keepdims=True))
        g = (p / p.sum(1, keepdims=True) - y) / BATCH
        gh = (g @ w2.T) * (h > 0)
        w2 -= 0.01 * h.T @ g
        w1 -= 0.01 * x.T @ gh
        done += BATCH
    return done


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    with Pool(workers) as pool:
        total = sum(pool.map(train, [seconds] * workers))
    print(json.dumps({"workers": workers, "samples_per_s": total / seconds}))
