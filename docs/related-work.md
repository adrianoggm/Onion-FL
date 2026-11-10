# Related work: Onion-FL and the existing frameworks

- **Issue:** #202 (I17). **Spec:** `docs/superpowers/specs/2026-11-03-onion-fl-infrastructure-design.md` §1.
- **Checked on:** 2026-11-06. Every claim about another tool cites its own documentation, paper or repository ([Sources](#sources)), with the version or date that was read.
  A cell that says *not found* means the sources read do not mention it. It does not mean the tool cannot do it.
- **Onion-FL claims** cite a file in this repository, or a local branch for work that is not merged into `develop` yet.

## Scope

The issue asks for three families of tools:

| Family | Tools | What they are for |
|---|---|---|
| Federated learning frameworks | Flower, FedML, FedScale | Writing and running FL algorithms, in simulation and on devices |
| Continuum emulators | Fogify, EmuFog | Running real containers over an emulated fog network |
| Continuum simulators | iFogSim, EdgeSimPy | Simulating resource management on IoT, edge and fog infrastructures |

Onion-FL sits between them: an FL engine (the first family) whose experiments run on a declared Edge–Fog–Cloud infrastructure, simulated or emulated (the other two), with the fidelity between the two measured.

The platforms that the capabilities spec (`2026-11-04-onion-fl-capabilities.md` §4) names for the engineering paper (Flight, FedDES, FederNet, FLEET and Valli et al.) are not compared here: that comparison has to be made from their full texts, and it is left to the paper.

## The tools

### Flower

Read: the Flower framework docs, v1.43.x ([F1], [F2]); latest release 1.43.0, 2026-11-04 ([F3]).

- **Topology.** Hub and spoke: one long-running SuperLink, to which SuperNodes connect, with the server logic in a ServerApp and the client logic in ClientApps ([F2]). Several runs can share one SuperLink and its SuperNodes ([F2]).
- **Hierarchy.** Intermediate aggregators are not found in the architecture docs ([F2]).
- **Simulation.** The Simulation Runtime runs ClientApps on a backend, by default Ray, queuing them while resources are busy ([F1]). Every ClientApp is assumed to use the same `num_cpus` and `num_gpus` ([F1]). A virtual clock and network modelling (latency, bandwidth) are not found ([F1]).
- **Simulation and deployment.** The ClientApps of the Simulation Runtime are identical to those of the Deployment Runtime ([F1]).

### FedML

Read: the FedML paper, arXiv v4 of 2020-11-08 ([M1]); the repository at `master`, last push 2025-10-28 ([M2]); the PyPI package, 0.9.6 of 2025-02-24 ([M3]).

- **Paradigms.** On-device training for edge devices, distributed computing and single-machine simulation ([M1], abstract and §1). The README now presents the project as TensorOpera Open Source ([M2]).
- **Topology.** A `TopologyManager` lets workers send messages to arbitrary neighbours; the paper draws centralized, decentralized, hierarchical, vertical and split topologies ([M1], §3, Fig. 3).
- **Hierarchy.** The repository has a single-process `HierarchicalFL` example: clients in groups, with group rounds inside each global round ([M4]). Hierarchies of more than two levels are not found.
- **Devices.** FedML-Mobile (Android) and FedML-IoT (Raspberry Pi 4, NVIDIA Jetson Nano), so that the distributed code can be moved to real hardware, reusing nearly all the algorithm implementations ([M1], §2). The communication layer has MPI and MQTT backends ([M1], Fig. 1).
- **Network and device modelling in simulation.** Not found ([M1], [M4]).

### FedScale

Read: the FedScale paper, ICML 2022, arXiv v5 of 2022-06-18 ([S1]); the repository, last push 2023-12-18 ([S2]); the PyPI package, 1.0 of 2024-04-06 ([S3]).

- **Workloads.** 20 real-world federated datasets, partitioned by their real client mapping ([S1], §3.1). Device traces: compute and network speed from AI Benchmark and MobiPerf, and availability from a dataset of 136k users ([S1], §3.2).
- **Simulation.** The simulation mode emulates FL behaviour with a global virtual clock: computation latency is the processed samples × the latency per sample, and communication latency is the traffic size over the client's bandwidth. A client is dropped when its availability slot ends ([S1], §4.2).
- **Deployment.** A deployment mode orchestrates real devices (laptops, mobiles, cloud servers), and a mobile backend runs on Android through Termux; the paper says the same code runs in both modes with little change ([S1], §4.1–4.2).
- **Hierarchy.** Not found: the aggregator simulator is one aggregator ([S1], §4.2, Fig. 8).
- **Per-link network topology.** Not found: the bandwidth is per client ([S1], §4.2).

### Fogify

Read: the README and the documentation site of the repository, last push 2024-01-30 ([G1], [G2], [G3], [G4], [G5]); the paper is Symeonides et al., SEC 2020 ([G1]).

- **What it is.** An emulation framework for modelling, deploying and experimenting with fog testbeds ([G1]). The model extends a docker-compose file with an `x-fogify` section ([G1]).
- **Network.** Network profiles with separate uplink and downlink: delay (with deviation, correlation and distribution, or ping traces), bandwidth and drop rate; `links` set properties between two given nodes ([G3]). Every cluster node needs `iproute` for `tc` ([G5]).
- **Compute.** Node profiles with cores and clock speed, translated into cgroup constraints per container, and memory; disk properties do not affect the deployment in the current version ([G4]).
- **Runtime.** Actions and Scenarios (timestamped sequences of actions) apply faults and topology changes to a running deployment ([G1], [G2]). Metrics are collected per agent and read through the FogifySDK ([G1]).
- **Scale.** Fogify needs a Docker Swarm cluster ([G5]), so an emulation can span several machines.
- **FL.** Not found: Fogify runs whatever services the compose file declares.

### EmuFog

Read: the EmuFog paper, arXiv 1709.07563 ([E1]); the repository, last push 2020-09-28, Kotlin, MIT ([E2]).

- **What it is.** It generates a fog topology and runs it on MaxiNet, the multi-machine extension of Mininet, with the application components as Docker containers on the fog nodes ([E1], §III-B).
- **Topology.** From a generator (BRITE) or a file, such as CAIDA data; links have latency and throughput ([E1], §III-B, §IV). Fog nodes are placed greedily so that every edge router is within a latency bound of one ([E1], §III-C).
- **Compute limits, runtime changes and faults.** Not found. Hierarchical fog support and mobility are listed as future work ([E1], §V).
- **FL.** Not found.

### iFogSim

Read: the repository README for iFogSim2, last push 2025-10-03, Java, MIT ([I1], [I2]); the iFogSim paper, arXiv 1606.02007 ([I3]); the iFogSim2 paper, arXiv 2109.05636, *Journal of Systems and Software* 190 (2022) ([I4]).

- **What it is.** A toolkit for modelling and simulating resource management techniques in IoT and fog environments, measured by latency, network congestion, energy consumption and cost ([I3]). It is built on CloudSim ([I1]).
- **iFogSim2** adds mobility (real datasets and random models) and migration, dynamic distributed clustering and microservice orchestration ([I1], [I4]).
- **FL.** Not found ([I1], [I3], [I4]).

### EdgeSimPy

Read: the repository README, last push 2026-05-01, Python, GPL-3.0 ([P1], [P2]); the paper is Souza et al., *Future Generation Computer Systems* 148 (2023) ([P1]).

- **What it is.** A Python simulator for prototyping resource management policies in edge computing, with models for user mobility, application composition and power consumption ([P1]).
- **Infrastructure.** Edge servers (CPU, RAM, disk, MIPS, power models), base stations with wireless latency, and switches and links with delay and bandwidth, in a NetworkX topology ([P1]).
- **Services.** Containers with their CPU, memory and disk, and container registries whose images are built from layers ([P1]).
- **Time.** Discrete steps of a configurable `tick_duration` ([P1]).
- **FL.** Not found.

## Capability matrix

✅ the sources say so; ◐ partly, as the note says; ❌ the sources say it is not there; — not found in the sources read. Onion-FL cells cite this repository.

| Capability | Flower | FedML | FedScale | Fogify | EmuFog | iFogSim | EdgeSimPy | Onion-FL |
|---|---|---|---|---|---|---|---|---|
| FL algorithms built in | ✅ [F1] | ✅ [M1] §4 | ✅ [S1] §4 | — | — | — | — | ✅ (1) |
| Hierarchical aggregation (fog level) | — [F2] | ◐ two levels, single process [M4] | — [S1] | n/a | n/a | n/a | n/a | ✅ any number of levels (2) |
| Simulation with a virtual clock | — [F1] | — | ✅ [S1] §4.2 | ❌ emulator | ❌ emulator | ✅ simulator [I3] | ✅ time steps [P1] | ✅ (3) |
| Per-link network model (latency, bandwidth, loss) in simulation | — [F1] | — | ◐ bandwidth per client [S1] §4.2 | n/a | n/a | ◐ measures latency and network congestion [I3] | ◐ delay and bandwidth per link [P1] | ✅ (4) |
| Device heterogeneity (compute, availability) | ◐ same resources per ClientApp [F1] | — | ✅ traces [S1] §3.2 | ✅ node profiles [G4] | — | — | ✅ server capacities [P1] | ✅ (5) |
| Emulation: real processes, limited CPU and memory, shaped links | — | — | — | ✅ [G3], [G4] | ◐ containers on MaxiNet; CPU limits not found [E1] | ❌ | ❌ | ✅ one machine (6) |
| Emulation over several machines | n/a | n/a | n/a | ✅ Docker Swarm [G5] | ✅ MaxiNet [E1] | n/a | n/a | ❌ (6) |
| Runtime network changes and faults | — | — | ◐ availability traces [S1] §4.2 | ✅ actions, scenarios [G2] | — [E1] §V | ✅ mobility [I1] | ✅ mobility [P1] | ◐ in `sim` only (7) |
| Deployment on real devices | ✅ Deployment Runtime [F1] | ✅ Android, Raspberry Pi, Jetson [M1] §2 | ✅ [S1] §4.1–4.2 | n/a | n/a | n/a | n/a | ◐ one machine, `local` (8) |
| Same FL code in simulation and on devices | ✅ ClientApps [F1] | ◐ "nearly all" [M1] §2 | ◐ "little changes" [S1] §4.2 | n/a | n/a | n/a | n/a | ✅ same node objects (9) |
| Measured fidelity between backends | — | — | — | — | — | — | — | ✅ (10) |
| Run identity and signature | — | — | — | — | — | — | — | ✅ (11) |
| Declarative experiment | — | ✅ YAML [M4] | ✅ config file [S1] §4.2 | ✅ compose + `x-fogify` [G1] | ◐ fog config file [E1] §III-C | — (experiments are Java code) | — (experiments are Python code) | ✅ YAML (12) |
| Maintenance (last release or push) | 1.43.0, 2026-11-04 [F3] | 0.9.6, 2025-02-24 [M3] | 1.0, 2024-04-06 [S3] | push 2024-01-30 [G1] | push 2020-09-28 [E2] | push 2025-10-03 [I2] | push 2026-05-01 [P2] | — |

*Hierarchical aggregation* is n/a for the emulators and simulators, which run no FL of their own.

**Onion-FL notes.**

1. `src/onion_fl/learning/`: the aggregators and server optimizers listed in `docs/architecture.md` §10, from FedAvg to Krum, Bulyan, DP-FedAvg, FedAdam, SCAFFOLD, FedNova and FedDyn.
2. `docs/architecture.md` §3: a tree with any number of levels, each node with its aggregator, quorum and deadline; sharing scopes decide which keys cross each link (§5).
3. `SimRuntime` (`docs/architecture.md` §7).
4. Link profiles per logical link (fixed, normal or lognormal latency, jitter, bandwidth, loss); with an infrastructure, the path of a link composes its physical segments, and a shared segment is one FIFO queue (`docs/architecture.md` §7a).
5. Compute models (`samples_per_second`, `measured`, `lognormal_rate`, `trace`) and availability models (`always`, `bernoulli`, `schedule`, `crash_at`); infrastructure profiles with `train_rate`, `aggregate_rate`, `cpus` and `memory` (`docs/architecture.md` §7a, §10).
6. The `emulated` backend (issue #195, branch `task/#195`): one container per host with its CPU quota and memory limit, a router per site and netem per physical link, on one Docker host (`docs/architecture.md` §7, *Emulated runs*). The choice of Docker Compose + netem over Containernet was measured in `docs/decisions/2026-11-06-i9-emulator.md`. Several machines are the `distributed` backend (I14, #199), not built.
7. `network.schedule` changes or cuts a link at a time in `sim` (`docs/architecture.md` §7a); the `emulated` backend refuses it.
8. The `local` backend runs one process per aggregator group on one machine, over MQTT (`docs/architecture.md` §7). Over gRPC (issue #196, branch `task/#196`) and over Flower messages (issue #204, branch `task/#204`) as well; both branches are not merged into `develop`. Hosts on real devices are the `distributed` backend (#199), not built.
9. `build_federation` builds the same `Coordinator`, `Aggregator` and `Edge` objects in every backend (spec I1 §3, §10). `tests/test_runtime_equivalence.py` checks that the final model is the same bit for bit across `sim`, `local` (MQTT) and `emulated` with the stub trainer.
10. `onion_fl fidelity A B` measures functional, performance and decision fidelity (`docs/architecture.md` §7, spec I1 §12); the first comparison is in `results/fidelity_emulated/INDEX.md`.
11. Every run has a `topology_id`, `config_id`, `run_id` and `run_hash` (`docs/architecture.md` §8).
12. One experiment YAML, with the topology, data, learning and infrastructure; `onion_fl plan` checks it without training (`docs/architecture.md` §3, §4, §7a).

## What Onion-FL does not do

The matrix shows where the other tools go further:

- **Device and user traces at scale.** FedScale ships real compute, network and availability traces for mobile clients ([S1], §3.2). Onion-FL takes a `trace` compute model and a link `trace` CSV, but ships no trace datasets.
- **Emulation across machines.** Fogify (Docker Swarm) and EmuFog (MaxiNet) spread an emulation over several hosts ([G5], [E1]). The `emulated` backend runs on one Docker host and refuses more than `max_containers` containers (64 by default).
- **Faults during an emulation.** Fogify applies faults and topology changes to a running deployment ([G2]); Onion-FL's emulated backend refuses `network.schedule`.
- **Mobility and energy.** iFogSim2 and EdgeSimPy model user mobility and power ([I1], [P1]); Onion-FL models neither (mobility is listed as later in `docs/architecture.md` §7a).
- **Real devices.** Flower, FedML and FedScale deploy on phones or boards ([F1], [M1], [S1]). Onion-FL's `distributed` backend (#199) is not built.
- **Maturity.** Flower is a released product (1.43.0 on 2026-11-04, [F3]); Onion-FL is a research framework.

## Measurements

The issue asks for a comparable measurement where possible. **No other tool was installed or run for this comparison**, so this section has no head-to-head number. Running the same scenario in another tool would need it rebuilt there by hand: Flower and FedScale have no fog level ([F2], [S1]), and Fogify, EmuFog, iFogSim and EdgeSimPy run no FL. That rebuild is the comparison, and it is left as pending work.

What Onion-FL itself measured on the scenario of `results/fidelity_emulated/` (1 cloud, 2 fogs, 20 SWELL edges on 5 devices, 10 rounds, 3 seeds; commit `67fdded`):

| Quantity | Value | Source |
|---|---|---|
| Lines to declare the scenario (experiment and infrastructure, without comments or blank lines) | 32 + 18 | `experiments/fidelity_emulated.yaml`, `infrastructures/emulated_lab.yaml` |
| Change from simulation to emulation | `--backend emulated` instead of `--backend sim` | `results/fidelity_emulated/INDEX.md` |
| Containers per emulated run, and their setup time | 13; 6–14 s | same |
| Run duration, emulated / simulated (uncalibrated) | 91.6 s / 17.6 s | same |
| Functional fidelity | the same protocol events and macro-F1; final models within 2.1 × 10⁻⁷ (bit for bit with the stub trainer) | same |
| Emulator accuracy per link (RTT, bandwidth, loss) | inside the I9 criteria (≤ 10 % or 1 ms, ≤ 10 %, ≤ 1 point) | `docs/decisions/2026-11-06-i9-emulator.md` |

The simulator's scale, from issue #206 (branch `task/#206`, not merged): with the stub trainer, 10 fogs and a FEMNIST-sized model, about 11 ms and 1.2 MiB per edge and round, up to the 4,000 edges checked on one machine (`results/sim_scale/INDEX.md` on that branch).

## Sources

All read on 2026-11-06.

| Id | Source |
|---|---|
| [F1] | Flower docs, *Run simulations*, v1.43.x. <https://flower.ai/docs/framework/how-to-run-simulations.html> |
| [F2] | Flower docs, *Flower Architecture*, v1.43.x. <https://flower.ai/docs/framework/explanation-flower-architecture.html> |
| [F3] | Flower releases, `framework-1.43.0`, 2026-11-04. <https://github.com/adap/flower/releases> |
| [M1] | C. He et al., *FedML: A Research Library and Benchmark for Federated Machine Learning*, arXiv:2007.13518v4, 2020. <https://arxiv.org/abs/2007.13518> |
| [M2] | FedML repository, `master`. <https://github.com/FedML-AI/FedML> |
| [M3] | `fedml` on PyPI, 0.9.6. <https://pypi.org/project/fedml/> |
| [M4] | FedML, `python/examples/federate/simulation/sp_hierarchicalfl_mnist_lr_example/fedml_config.yaml`, `master`. <https://github.com/FedML-AI/FedML/tree/master/python/examples/federate/simulation/sp_hierarchicalfl_mnist_lr_example> |
| [S1] | F. Lai et al., *FedScale: Benchmarking Model and System Performance of Federated Learning at Scale*, ICML 2022, arXiv:2105.11367v5. <https://arxiv.org/abs/2105.11367> |
| [S2] | FedScale repository. <https://github.com/SymbioticLab/FedScale> |
| [S3] | `fedscale` on PyPI, 1.0. <https://pypi.org/project/fedscale/> |
| [G1] | Fogify repository README; M. Symeonides et al., *Fogify: A Fog Computing Emulation Framework*, ACM/IEEE SEC 2020. <https://github.com/UCY-LINC-LAB/fogify> |
| [G2] | Fogify docs, overview and *Actions and Scenarios*. <https://ucy-linc-lab.github.io/fogify/> |
| [G3] | Fogify docs, *Network QoS*. <https://ucy-linc-lab.github.io/fogify/network-qos.html> |
| [G4] | Fogify docs, *Node profiles*. <https://ucy-linc-lab.github.io/fogify/node-profiles.html> |
| [G5] | Fogify docs, *Installation*. <https://ucy-linc-lab.github.io/fogify/installation.html> |
| [E1] | R. Mayer et al., *EmuFog: Extensible and Scalable Emulation of Large-Scale Fog Computing Infrastructures*, IEEE Fog World Congress 2017, arXiv:1709.07563. <https://arxiv.org/abs/1709.07563> |
| [E2] | EmuFog repository. <https://github.com/emufog/emufog> |
| [I1] | iFogSim2 repository README. <https://github.com/Cloudslab/iFogSim> |
| [I2] | iFogSim repository metadata. <https://api.github.com/repos/Cloudslab/iFogSim> |
| [I3] | H. Gupta et al., *iFogSim: A Toolkit for Modeling and Simulation of Resource Management Techniques in IoT, Edge and Fog Computing Environments*, arXiv:1606.02007; *Software: Practice and Experience* 47(9), 2017. <https://arxiv.org/abs/1606.02007> |
| [I4] | R. Mahmud et al., *iFogSim2*, arXiv:2109.05636; *Journal of Systems and Software* 190, 2022. <https://arxiv.org/abs/2109.05636> |
| [P1] | EdgeSimPy repository README; P. S. Souza, T. Ferreto, R. N. Calheiros, *EdgeSimPy: Python-Based Modeling and Simulation of Edge Computing Resource Management Policies*, *Future Generation Computer Systems* 148, 446–459, 2023, doi:10.1016/j.future.2023.06.013. <https://github.com/EdgeSimPy/EdgeSimPy> |
| [P2] | EdgeSimPy repository metadata. <https://api.github.com/repos/EdgeSimPy/EdgeSimPy> |
