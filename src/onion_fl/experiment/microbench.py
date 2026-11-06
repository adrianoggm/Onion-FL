from __future__ import annotations

"""The microbenchmarks of a calibration, one process per host (I1 §13, I12).

``python -m onion_fl.experiment.microbench <job.json>`` runs one job and
appends its measurements, one JSON object per line, to the job's ``out``
file. ``onion_fl calibrate measure`` writes the jobs and starts them inside
the containers of the emulated backend, as that backend starts its nodes.

- **train**: the trainer of a workload trains, in turn, the model of each of
  the host's clients on its real training rows, as an edge does in a round;
  one line per training, with the samples it charged (epochs × rows) and its
  wall time.
- **aggregate**: ``fedavg`` combines ``updates`` models of the workload, as
  an aggregator does, for ``seconds`` per number of updates; one line per
  ``batch`` of combinations, with the bytes combined.
- **send** and **echo**: one-way delays over the transport. ``echo`` waits on
  its inbox and acknowledges each message; ``send`` sends each size
  ``repeats`` times, one at a time, and waits for the acknowledgement. Both
  ends stamp ``time.time()``: the containers share one kernel clock.

The hosts of a train or aggregate phase start together at the job's
``start_at``, and every measurement carries ``time.time()`` stamps, so the
launcher can keep those taken while every host of the phase was busy.
"""

import json
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

STAMP = 16  # bytes at the head of a link message: sequence and send time


def _write(out: Any, **row: Any) -> None:
    out.write(json.dumps(row, sort_keys=True) + "\n")
    out.flush()


def _threads() -> int:
    import torch

    return torch.get_num_threads()


def _start(job: dict[str, Any]) -> bool:
    """Wait for the phase's common start (``start_at``, epoch seconds), so that
    every host of a phase works at once; False if it was already past."""
    late = time.time() - job.get("start_at", 0.0)
    if late < 0:
        time.sleep(-late)
    return late < 0


def train(job: dict[str, Any], out: Any) -> None:
    from onion_fl.learning.model import models
    from onion_fl.learning.trainers import trainers
    from onion_fl.roles.policies import create

    # The launcher wrote this file in its own session folder for this job.
    with open(job["data"], "rb") as f:
        clients = pickle.load(f)[job["host"]]
    family = create(models, job["model"])
    trainer = create(trainers, job["trainer"])
    built = {cid: family.build([data.shape], seed=job["seed"]) for cid, data in clients}
    threads = _threads()
    on_time = _start(job)
    for i in range(job["jobs"]):
        cid, data = clients[i % len(clients)]
        t0, start = time.time(), time.perf_counter()
        result = trainer.train(built[cid], data)
        wall = time.perf_counter() - start
        _write(
            out,
            kind="train",
            workload=job["workload"],
            host=job["host"],
            profile=job["profile"],
            client=cid,
            job=i,
            samples=result.samples,
            wall_s=wall,
            t0=t0,
            t1=time.time(),
            threads=threads,
            on_time=on_time,
        )


def aggregate(job: dict[str, Any], out: Any) -> None:
    from onion_fl.learning.aggregators import Contribution, aggregators
    from onion_fl.learning.model import models, state_arrays
    from onion_fl.roles.policies import create

    with open(job["data"], "rb") as f:  # written by the launcher, as above
        shape = pickle.load(f)["shape"]
    family = create(models, job["model"])
    aggregator = create(aggregators, "fedavg")
    threads = _threads()
    sets = {}
    for updates in job["updates"]:
        given = []
        for k in range(updates):
            state = state_arrays(family.build([shape], seed=job["seed"] + k))
            given.append(Contribution(f"u{k:03d}", state, dict.fromkeys(state, 100.0)))
        sets[updates] = given
    on_time = _start(job)
    for updates, given in sets.items():
        size = float(sum(np.asarray(v).nbytes for c in given for v in c.state.values()))
        until, r = time.perf_counter() + job["seconds"], 0
        while time.perf_counter() < until:  # each row times a batch of combinations
            t0, start = time.time(), time.perf_counter()
            for _ in range(job["batch"]):
                aggregator.aggregate(given, source=job["host"])
            wall = time.perf_counter() - start
            _write(
                out,
                kind="aggregate",
                workload=job["workload"],
                host=job["host"],
                profile=job["profile"],
                updates=updates,
                repeat=r,
                combinations=job["batch"],
                bytes=size * job["batch"],
                wall_s=wall,
                t0=t0,
                t1=time.time(),
                threads=threads,
                on_time=on_time,
            )
            r += 1


def _transport(job: dict[str, Any]) -> Any:
    from onion_fl.transports.mqtt import MqttTransport

    return MqttTransport(broker=job["broker"], qos=job["qos"])


def echo(job: dict[str, Any], out: Any) -> None:
    """Receive until ``send`` says it is done, acknowledging every message."""
    transport = _transport(job)
    done = threading.Event()

    def received(data: bytes) -> None:
        t_recv = time.time()
        if data == b"done":
            done.set()
            return
        seq = int.from_bytes(data[:8], "big")
        t_send = np.frombuffer(data[8:STAMP], dtype=">f8")[0]
        _write(
            out,
            kind="link",
            src=job["peer"],
            dst=job["host"],
            seq=seq,
            size=len(data),
            t_send=float(t_send),
            t_recv=t_recv,
        )
        transport.send(job["ack"], data[:8])

    transport.on_receive(job["inbox"], received)
    transport.start()
    Path(job["ready"]).write_text("ready", encoding="utf-8")
    done.wait(job["timeout"])
    transport.stop()


def send(job: dict[str, Any], out: Any) -> None:
    """Each size ``repeats`` times, the next once the last is acknowledged."""
    transport = _transport(job)
    acked: dict[int, float] = {}
    arrived = threading.Condition()

    def ack(data: bytes) -> None:
        with arrived:
            acked[int.from_bytes(data[:8], "big")] = time.time()
            arrived.notify_all()

    transport.on_receive(job["ack"], ack)
    transport.start()
    seq = 0
    for size in job["sizes"]:
        for r in range(job["repeats"]):
            seq += 1
            head = seq.to_bytes(8, "big")
            stamp = np.array([time.time()], dtype=">f8").tobytes()
            transport.send(job["inbox"], head + stamp + bytes(max(size - STAMP, 0)))
            with arrived:
                arrived.wait_for(lambda s=seq: s in acked, job["wait"])
            _write(
                out,
                kind="sent",
                src=job["host"],
                dst=job["peer"],
                seq=seq,
                size=max(size, STAMP),
                repeat=r,
                acked=seq in acked,
            )
            time.sleep(job["gap"])
    transport.send(job["inbox"], b"done")
    time.sleep(0.5)
    transport.stop()


KINDS = {"train": train, "aggregate": aggregate, "echo": echo, "send": send}


def main(argv: list[str] | None = None) -> int:
    path = Path((argv if argv is not None else sys.argv[1:])[0])
    job = json.loads(path.read_text(encoding="utf-8"))
    with open(job["out"], "a", encoding="utf-8", newline="\n") as out:
        KINDS[job["kind"]](job, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
