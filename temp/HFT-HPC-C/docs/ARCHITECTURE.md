# Architecture and ownership

`hft_core` is a C static library. `hft_bench` is the CLI; `hft_studio` is the native desktop; `hft_mpi_worker` is built when MPI C is found. CMake never enables the C++ or CUDA languages.

The event path is background generation → FIFO matching → observable features → signal → risk → order submission → matching/accounting → trace sampling. The same inline functions are compiled as host C and embedded into the OpenCL C program. Host/device structure sizes are checked before simulation. Device execution requires fp64 support and disables floating-point contraction in shared code. Host builds use `-ffp-contract=off` or `/fp:strict`.

Random draws are Philox4x32-10 indexed by seed, symbol, stream and event. Global and sector streams create correlated shocks; a symbol stream contributes idiosyncratic noise. Integer prices and cash make matching/accounting exact. Float model gradients and external math libraries use numerical tolerances instead of false bitwise promises.

A symbol owns a fixed-depth book with direct order-ID slots, intrusive FIFO lists and occupancy bitmaps. The reference selects best orders by scanning slots; the optimized path searches occupied price words then follows FIFO. Capacity is bounded and exhausted/invalid slots can reject an order. Background IDs reserve the final 16 slots for strategy quotes. Prices outside the fixed book window are rejected. IOC orders do not consume resting slots. Self-crossing strategy makers are cancelled before executing the incoming strategy order.

A pair is the parallel ownership unit even for non-pairs strategies. One thread/work item/rank owns both symbols, preserving the two-leg decision order without shared matching locks. Global gross/loss budgets are partitioned equally among symbols. This conservative partition gives deterministic independent risk decisions; it is not a dynamically shared portfolio-credit system. Loss halts cancel quotes and attempt bounded IOC liquidation; insufficient liquidity may leave residual exposure.

CPU workers receive pair indices through one SPSC queue per worker. A release/acquire completion counter protects reductions and snapshots. OpenMP iterations own pairs and thread-local histograms. MPI distributes pair IDs; local symbols keep their global IDs for RNG and order contracts. Idle ranks remain in collectives. OpenCL work items use global state and private arithmetic temporaries; queues operate on disjoint pair ranges.

The UI thread renders only snapshots and completed results. The service thread owns the mutable run result. An acquire/release triple buffer transfers copies without locking a live book. Training commits its model after completion. No UI or file operation occurs in a per-event kernel.

Memory is intentionally bounded per symbol, but not tiny: inspect `sizeof(HftSymbol)` in `examples/contracts.c`. Large configurations need enough host and device RAM; allocation failures report errors. OpenCL checks maximum allocation size. Benchmark timers exclude initialization/JIT setup, include the measured simulation loop and enabled snapshot/transfer work. Live throttling and pauses therefore belong to live wall-clock time; use the CLI for performance claims.
