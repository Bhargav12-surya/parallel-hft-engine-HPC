# Implementation status

This is a runnable local research implementation, not a claim that every item in the master directive or production-readiness gate is complete. Verification below was performed in this workspace. The reference projects under `temp/` remain unchanged; historical reference logs are not counted as current evidence.

## IMPLEMENTED AND TESTED

- C17 event-driven exchange: price-time priority, indexed cancellation/replacement, partial/multiple fills, invalid/duplicate order checks, dynamically growing books.
- Native cash/inventory, long/short realized and unrealized accounting, fees, adverse slippage, bounded simulated latency, pre-submission/delivery risk checks, chronological portfolio equity and drawdown.
- Market making, momentum, mean reversion, paired statistical arbitrage with leg risk, AI, and HOLD strategies.
- Four CPU execution modes with genuine OS threads/affinity, genuine OpenMP group parallelism, and CPU+OpenMP hybrid feature/matching phases. Same-input classical and AI correctness checks against sequential.
- CPU-only and CPU+OpenMP CMake builds, each passing CTest. Native sources do not require CUDA to build.
- Streaming seeded data generation in eight scenarios, CSV import/validation, explicit little-endian versioned binary/checksum roundtrip, binary-to-native orchestration. Disk-backed ID/reference validation.
- Real Python, sklearn logistic, and PyTorch CPU MLP training; causal six-feature preset; purged chronological train/validation/test splits; train-only scaling; safe native weight export; Python/C probability parity within2e-11.
- Strict held-out trading replay with provenance and equal-cost/capital MLP/logistic/momentum/mean-reversion/HOLD comparisons.
- Installed `hftlab` launcher and `python -m hftlab`; functional headless commands, supervised real native jobs, cancellation/timeouts/process reaping, externally cancellable quiet benchmarks, atomic artifacts, hashes/environment metadata and transactional SQLite registry.
- Textual workflow, exact five engine categories/four CPU modes, navigation/validation, background simulation workers, results, history, training, diagnostics, help/preferences. Headless tests include responsive layouts and real-native worker integration.
- Repeated correctness-gated benchmarks, homogeneous efficiency, confidence intervals, strong/weak scaling, offline HTML/JSON/CSV/Markdown exports and actual equity/runtime charts.
- Editable installation, wheel/sdist creation, isolated wheel import/CSS/native-discovery smoke, dependency consistency and Ruff checks.

See [test evidence](TEST_RESULTS.md) for exact commands/results and [evaluation demo](EVALUATION_DEMO.md) for launch instructions.

## IMPLEMENTED BUT UNTESTED

- Real MPI code paths: initialization, rank ownership, collective distribution/reduction/gather, coordinated failure handling and profiling.
- Real CUDA `.cu` causal feature and logistic/MLP kernels, allocation/transfers/synchronization/error handling and separate transfer/kernel timings.
- Hybrid combinations requiring MPI/CUDA. CPU+OpenMP is the locally verified combination.
- Optional Parquet import/export when PyArrow is installed.
- Linux/pthread execution, GCC/Clang sanitizer builds, Linux/MPI CI configuration and remote-host MPI use. Source/configuration exists; this session ran on Windows.
- Million-event and larger presets are provided, but the largest locally recorded benchmark here is100,000 events. No100-million-event capacity claim is made.

## PARTIALLY IMPLEMENTED

- UI: the full workflow is usable, but some results panes render structured measurements rather than specialized charts. Simulation/training/benchmark cancellation is supported; data-management work completes its current worker operation. Feature presets and classical strategy indicator windows are fixed documented defaults.
- Profiling: native phase times, latency samples, peak memory, affinity count and available OS fields are implemented. Detailed live per-core/GPU utilization, CPU migrations, measured NUMA data, cache counters, CUDA occupancy and per-rank imbalance dashboards are incomplete.
- Trading realism: modeled top-of-book liquidity, deterministic fill mechanics, fees and slippage are implemented. Calibrated market impact, exchange-specific microstructure, atomic pair fills, portfolio margin/leverage models and a configurable family of latency/slippage distributions are not complete.
- Scalability: generation/import are streamed; native simulation holds events, indexes, equity deltas and optional feature buffers in RAM. A bounded-memory streaming native exchange is not implemented.
- AI: configurable one-hidden-layer MLP and multinomial logistic exports are supported. There is no resumable distributed training, arbitrary feature-selection UI, sequence model, reinforcement learning, or full trading attribution/calibrated financial risk suite. Classification metrics and held-out simulated P&L are reported separately.
- Packaging: Python wheel and native CMake runner are separate artifacts. The wheel requires an independently built/discovered runner or `HFTLAB_RUNNER`.
- Diagnostics: capability/runtime statuses are real; CUDA host-compiler/toolkit/device compatibility cannot be certified without compiling/running that backend.

## NOT IMPLEMENTED

- Live exchange/broker connectivity or real-money trading (deliberately outside scope).
- Cloud scheduler/deployment, managed remote cluster provisioning, distributed AI training, advanced sequence/RL models, calibrated real-market profitability claims.
- Fully automatic native binary compilation/bundling inside platform-specific Python wheels.
- Exhaustive production/security certification or proof of absence of all memory/race errors. Sanitizer gates remain unrun on this Windows host.

## BLOCKED BY ENVIRONMENT

- No installed `mpiexec`/MPI development implementation was detected; MPI compilation/execution was not verified.
- NVIDIA RTX5050 Laptop GPU and driver621.01 are visible, but no `nvcc` toolkit was detected. Native CUDA compilation/execution was not verified. CPU PyTorch training does not satisfy the CUDA gate.
- WSL/Linux and supported Linux sanitizer/profiling tools were not available for execution tests.
- Windows sandbox blocks asyncio's internal loopback socket initialization and can block Visual Studio PDB IPC. UI tests and a fresh CPU-only compiler probe passed outside that restriction; application code did not replace or bypass those OS mechanisms.

## Engineering handoff

Architecture: Python modules orchestrate9 C sources,3 public C headers and1 CUDA source, with versioned schemas/configurations, tests, CI and dedicated documentation. `native/core` owns exchange/accounting/work scheduling; `native/ai` owns numerical inference; Python owns data/training/CLI/TUI/persistence/reports. See [engine implementation](ENGINE_IMPLEMENTATION.md) and [architecture](ARCHITECTURE.md).

Primary outputs: `build/hftlab_runner.exe`, `build/cpu/hftlab_runner.exe`, `.venv/Scripts/hftlab.exe`, `dist/hftlab-0.1.0-py3-none-any.whl`, `dist/hftlab-0.1.0.tar.gz`. Generated data/models/experiments/reports are ignored by Git by default.

Measured artifacts: `artifacts/demo/demo.json` and `demo.html`; `reports/performance-smoke.json` and `.html`; `reports/strong.json`/`weak.json`; `reports/heldout-comparison.json`; raw validation evidence in `artifacts/validation/`. Exact numerical results are in [TEST_RESULTS.md](TEST_RESULTS.md).

Launch from `D:\HPC`:

```powershell
.\.venv\Scripts\hftlab.exe
.\.venv\Scripts\hftlab.exe demo --events 10000 --epochs 15 --output artifacts/demo
```
