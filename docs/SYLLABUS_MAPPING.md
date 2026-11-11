# Measurable HPC and OS concepts

| Concept | Implemented source | Reproducible observation |
|---|---|---|
| Sequential reference and deterministic algorithms | native/core/simulation.c | simulate with sequential/single-single; compare checksums |
| Shared vs distributed memory | OpenMP and guarded MPI branches in simulation.c, runners/main.c | compare OpenMP locally; MPI requires a built matching runtime |
| Real threads and scheduling | native/core/platform.c, simulation.c | compare single-multi with multi-multi at equal event/group counts |
| Affinity and migratable scheduling | native/core/platform.c | compare single-single with multi-single; inspect applied affinity count |
| Synchronization and race prevention | Work mutex, disjoint group/result arrays in simulation.c | CPU/OpenMP replay parity and cancellation tests |
| IPC and process lifecycle | hftlab/orchestration, engines | lifecycle timeout/cancel tests; inspect process exit and persisted states |
| Producer-consumer coordination | bounded JSONL queue in orchestration | actual native progress while Textual remains responsive |
| Memory management/cache layout | native/core/orderbook.c, market.c; indexed event partitions | bounded allocation checks and documented memory budgets |
| OpenMP loops and data sharing | native/core/simulation.c | real team size and grouped workloads in profiling |
| MPI communication/domain decomposition | native/runners/main.c, simulation.c | rank partition, Bcast/Reduce/Gather timings; unavailable on this host |
| GPU blocks, threads, device memory | native/engines/cuda/compute.cu | separate transfer/kernel measurements on a genuine CUDA build |
| HPC+AI/native inference | hftlab/ai/learning.py, native/ai/inference.c | train/export logistic and MLP; numerical parity tests |
| Speedup/efficiency/statistics | hftlab/benchmarks.py | repeated correctness-gated compare reports |
| Strong and weak scaling | hftlab/benchmarks.py | scaling command with fixed/growing input respectively |
| Amdahl/Gustafson/Karp–Flatt | docs/HPC_METHODOLOGY.md; benchmark inference metric | distinguish analytic predictions from measured times |
| OS resource counters | native/core/simulation.c; hftlab/monitoring.py | peak RSS; Unix context switches; unsupported counters remain null |
| Heterogeneous task roles | native/core/simulation.c | CPU+OpenMP feature/matching phase timings |

Remote/cloud deployment, measured NUMA topology, detailed hardware performance counters, CUDA hardware execution, and distributed remote-node experiments are not claimed as locally verified course demonstrations.
