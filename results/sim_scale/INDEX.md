# How the simulator scales with the edges (issue #206)

`scripts/sim_scale.py` at commit `a94baba`, run on 2026-11-05 on a Windows 11 workstation (24 logical CPUs, Python 3.11.15, CPU torch). Every size runs in its own process.

```bash
python scripts/sim_scale.py --edges 100 200 500 1000 2000 4000 \
    --diagnostics dataset_conflict drift participation fairness > npz_no_divergence.jsonl
python scripts/sim_scale.py --edges 100 200 500 1000 > npz_all.jsonl
python scripts/sim_scale.py --edges 100 200 --codec json > json_all.jsonl
```

## Setup

- One cloud, 10 fogs, the edges spread evenly over them, 3 rounds, lossless `lan` links.
- Each edge holds its own `modular_mlp` with FEMNIST's shape: 784 inputs, adapter 128, trunk [64], 62 classes, about 113k parameters.
- The edges run the stub trainer: no data and no learning. What is measured is the protocol's cost: one model per node, the messages with their encoding, aggregation and the diagnostics. Training comes on top, in proportion to the samples, however they are split into edges.
- Peak memory is the process's peak working set, and it includes the ~200 MiB that importing torch takes.

## Results

| Edges | Codec | Diagnostics | s per round | Peak memory (MiB) |
|---|---|---|---|---|
| 100 | npz | all but `divergence` | 1.1 | 348 |
| 200 | npz | all but `divergence` | 2.3 | 475 |
| 500 | npz | all but `divergence` | 4.8 | 832 |
| 1000 | npz | all but `divergence` | 9.6 | 1442 |
| 2000 | npz | all but `divergence` | 21.9 | 2668 |
| 4000 | npz | all but `divergence` | 43.7 | 5116 |
| 100 | npz | all | 1.2 | 348 |
| 200 | npz | all | 3.8 | 471 |
| 500 | npz | all | 11.2 | 840 |
| 1000 | npz | all | 39.4 | 1444 |
| 100 | json | all | 20.0 | 507 |
| 200 | json | all | 38.6 | 787 |

## Findings

- **Without `divergence`, time and memory grow linearly with the edges.** That is about 11 ms per edge and round, and about 1.2 MiB per edge (its model, the copies in flight and its node). The full FEMNIST, with ~3,600 writers, would fit: about 40 s per protocol round and 5 GB.
- **The `divergence` diagnostic is quadratic in the children of an aggregator.** It averages the cosine over every pair of child updates. At 100 children per fog (1,000 edges) it takes three quarters of the round. With hundreds of children per aggregator, leave it out of the topology's `diagnostics`, or use more fogs.
- **The JSON codec is the other cost:** it makes each round 10 to 16 times slower than `npz`, because every message is encoded and decoded even in simulation. `topologies/femnist_four_fogs.yaml` uses `npz`.
- **The limit we checked** is 4,000 edges on one machine, with `npz` and without `divergence`. Past that, memory is what runs out first, at ~1.2 MiB per edge for this model.

The files are the JSON lines that the script printed: `npz_no_divergence.jsonl`, `npz_all.jsonl` and `json_all.jsonl`.
