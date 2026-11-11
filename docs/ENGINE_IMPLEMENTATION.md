# Native engine implementation

The new runtime is C17 with a C-compatible CUDA interface. All execution and trading accounting use native code. The root CMake project and runner do not alter the reference projects under `temp/`.

## Exchange and deterministic work

`native/core/orderbook.c` maintains bid/ask binary heaps ordered by price and arrival sequence. An open-addressed ID map supports lookup/cancellation. Insert/cancel are O(log live orders); best-order lookup is O(1). Matching supports partial/multiple fills and checks duplicate/invalid submissions. Integer ticks avoid floating-point price ordering; the default conversion is 0.01 currency units per tick.

`native/core/simulation.c` groups independent instruments; statistical arbitrage groups adjacent pairs and processes each pair's events in original order. Strategies use the same matching, risk and fill callback across backends. QUOTE replaces modeled exogenous liquidity at mid±1tick. Market making places real resting quotes with inventory/volatility skew. Momentum/mean reversion use causal rolling windows. Pair trading uses rolling32 logspread z-scores and separately filled hedged target legs; execution is not atomic or risk-free. AI uses exported scaler/weights and stable native softmax.

Capital is allocated equally to instruments. Pre-submission checks constrain position/order size, gross exposure to2×allocated capital, cash for buys, maximum loss, and invalid sizes. Fees are notional basis points. Slippage is a configured adverse tick adjustment. Latency is a bounded32 pending-order queue with a second risk check at delivery; pending orders beyond the dataset end are not filled. These are educational execution models, not calibrated market impact.

Each processed event records the whole group's equity change, including both pair legs. Native code reconstructs portfolio equity in the original global event order and calculates portfolio drawdown. Final equity curves are sampled to bounded JSON points; per-instrument curves are not persisted.

## Five engine categories

| Category | Real work | Local verification |
|---|---|---|
| CPU computing | OS threads or one native worker over independent groups | Four modes tested |
| OpenMP shared memory | `omp parallel for` over independent groups; actual runtime team size recorded | Tested with MSVC OpenMP |
| MPI distributed memory | Real `MPI_Init`, Bcast/Allreduce/Reduce/Gather; rank ownership of groups | Source implemented; toolkit/launcher absent here |
| CUDA GPU | `.cu` per-event causal predecessor features and numerical MLP/logistic inference; CPU exchange matching | Source implemented; toolkit absent here |
| Hybrid | Explicit feature/signal/matching phases across selected components | CPU+OpenMP tested; remaining hardware-dependent combinations unverified |

CPU modes are `single-single` (one worker pinned to one eligible logical processor), `single-multi` (real workers share that pinned processor), `multi-single` (one migratable worker), and `multi-multi` (multiple eligible workers). One migratable worker is not simultaneous multicore execution. Worker counts are capped by independent groups and resources. Windows uses native thread APIs; Unix uses pthreads.

CPU+OpenMP parallelizes features/inference then matches serially. CPU+MPI distributes feature work and reduces it to root matching. MPI+OpenMP partitions groups across ranks and OpenMP teams within ranks. CUDA packs only rank-owned events; OpenMP can own signal processing after GPU features when both are selected. Selected components must execute their assigned phases; unavailable components are rejected, never relabeled as CPU execution. Avoid MPI rank×thread oversubscription.

## Timing and protocol

Runner arguments are explicit arrays constructed by `hftlab.engines`. Native stdout emits version1 JSONL progress/error/result events; an atomic result JSON file contains accounting, checksums, curve, latency and profiling fields. `--quiet` suppresses progress writes for benchmarks. CSV parsing, native timed compute region, and total process/orchestration durations have distinct scopes; see [methodology](HPC_METHODOLOGY.md).

CUDA allocation/transfer/kernel errors are checked and resources released; H2D/kernel/D2H timings remain separate. Runtime device probing occurs in the compiled native backend. A driver-visible GPU does not imply the CUDA implementation is built or verified. MPI uses actual collectives and coordinated allocation-failure propagation. Homogeneous in-memory C structs are communicated within MPI; cross-architecture heterogeneous MPI representations have not been validated.

The native exchange loads the complete dataset and allocates event indexes/equity deltas. Hybrid additionally uses contiguous feature/probability arrays. Generation is streaming, but bounded-memory native simulation and NUMA-specific optimization remain future work.
