# Architecture

`hftlab.cli` and `hftlab.tui.app` share `SimulationConfig`, `JobManager`, dataset/model loaders, report exporters, and the same native runner. The UI uses background workers and marshals progress to Textual's event loop; it does not run exchange matching in Python.

`hftlab.datasets.market` streams canonical events and validates types, ordering, IDs, and references using a temporary disk-backed index. It owns native-binary serialization, checksums, import, and scenario generation. `hftlab.ai.learning` owns causal features, temporal labels/splits, training, safe numerical export, and Python inference.

`hftlab.engines` discovers the built executable or `HFTLAB_RUNNER`, probes compiled/runtime capabilities, and builds argument arrays. `hftlab.orchestration` validates requested capabilities, launches the subprocess or MPI launcher, captures JSONL, handles cancellation/timeouts, validates final results, and persists manifests. It never silently substitutes engines. A bounded output queue prevents uncontrolled progress buffering.

`native/core/market.c` parses and bounds native input. `orderbook.c` maintains heap price-time priority and an indexed order-ID map. `simulation.c` groups independent instruments (adjacent pairs for statistical arbitrage), handles strategies, fills, risk, accounting, work scheduling, MPI aggregation, and equity reconstruction. `native/ai/inference.c` loads versioned numerical weights and performs stable softmax and ReLU forward passes. `native/engines/cuda/compute.cu` implements optional device feature/inference processing through a C ABI. `native/core/results.c` emits versioned results and capabilities.

Classical and AI strategies share execution/risk/matching logic. Parallel workers never concurrently modify the same dependent book/group. Pair legs share a synchronized local group; fills remain independent and can expose execution risk. Every recorded event has a deterministic global index. Per-group equity deltas are reduced in input order to reconstruct the global equity curve and drawdown.

Experiments have UUID-suffixed IDs, atomic JSON manifests, input/runner/model hashes, compiler capabilities, environment metadata, raw output, and a transactional SQLite index. Large artifacts stay in files. Registry deletion removes an index entry; it does not recursively delete arbitrary result paths. Inputs are never interpreted as shell code or Python pickle.

The copied reference projects under `temp/` are preserved. The new code uses an event/order-book contract that those indicator-oriented executables did not expose. Their historical logs are not used as evidence for this build.
