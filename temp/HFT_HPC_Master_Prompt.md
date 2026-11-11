# MASTER PROMPT — Parallel & GPU-Accelerated High-Frequency Trading Simulation Engine

> **How to use:** Fill in the `[FILL IN]` block in Section 1, then paste everything from **BEGIN PROMPT** to **END PROMPT** into your coding assistant. Best results come from an agentic coding tool that can create files, compile, and fix build errors itself (it will work through the 10 phases in Section 17). In a plain chat tool, send the prompt, then reply `continue with Phase N` after each phase.

---

## BEGIN PROMPT

### 0. Role and mission

You are a principal engineer with deep expertise in low-latency C++ (trading systems), high-performance computing (OpenMP, MPI, CUDA), numerical linear algebra, applied machine learning, and desktop UI design. Your mission is to design and implement, end to end, a complete, compile-ready Windows desktop application named **HFT-HPC Studio** (`HFT_HPC_Studio.exe`) — a "Parallel and GPU-Accelerated High-Frequency Trading Simulation Engine" built as a university HPC course project.

The application lets a user:

1. Watch a **dynamic synthetic market** in which prices move because simulated traders buy and sell through a real limit order book.
2. Choose a **trading strategy** — classical (rule-based) or AI-driven, including an AI "strategy selector" that picks a classical strategy based on the detected market regime.
3. Choose an **execution method** — serial CPU, optimized single-core CPU, multithreaded CPU, OpenMP, MPI, hybrid MPI+OpenMP, CUDA, CUDA + NVIDIA libraries.
4. See **results** — latency, throughput, speedup, efficiency, scaling, GPU/MPI breakdowns, and trading outcomes (live P&L, equity curve, drawdown, Sharpe, fills, inventory) — side by side for every method.

Every method must run the *same workload* and, in Verification Mode, make *identical trading decisions*, so performance comparisons are fair and provably correct.

### 1. Project context

**Course:** High Performance Computing (B.Tech AI & Data Science). The project must visibly demonstrate every unit:

| Unit | Syllabus topics | Where it appears in the app |
|---|---|---|
| 1 | Architecture & OS basics, shared vs distributed memory, parallel algorithms, performance metrics | Serial Naive and Serial Optimized CPU engines; multithreaded pipeline with core pinning and lock-free queues; System Info panel (cores, caches, SIMD support); full metrics suite (speedup, efficiency, cost, Amdahl, Gustafson, Karp–Flatt, strong/weak scaling) |
| 2 | OpenMP essentials, data sharing & synchronization, efficient OpenMP for matrices, MPI & distributed memory, MPI communication, matrix representation & parallel solvers, domain decomposition | OpenMP engine (schedules, reductions, critical vs atomic, false-sharing demo, tiled covariance matmul); MPI worker (symbol domain decomposition, Scatterv/Gatherv/Allreduce/Isend/Irecv, derived datatypes); parallel Conjugate Gradient solver for the minimum-variance portfolio (row-block decomposition); hybrid MPI+OpenMP |
| 3 | GPU architecture, GPGPU & CUDA, thread execution, matrix computing in CUDA, cuBLAS/cuDNN, GPU case studies | CUDA engine (grids/blocks/warps, shared-memory tiling, warp-shuffle reductions, streams, pinned memory, CUDA Graphs); naive vs tiled vs cuBLAS covariance; cuRAND Monte Carlo VaR; Thrust/CUB scans; optional cuDNN LSTM; GPU Deep-Dive dashboard |
| 4 | Cloud computing (IaaS/PaaS/SaaS, architecture, storage, networking, security), HPC + AI synergy, training & inference of AI models on HPC | AI strategies (MLP, Q-learning market maker, regime k-means + contextual-bandit selector); AI training/inference benchmarked across Serial/OpenMP/MPI/CUDA; Cloud & AI Lab (network-latency and cost simulator, service-model explorer, optional real multi-node run on cloud VMs) |

**My environment [FILL IN before sending]:**

- OS: Windows 11 x64
- CPU: `[e.g., Intel Core i7-12700H, 14 cores / 20 threads]`
- RAM: `[e.g., 16 GB]`
- GPU: `[e.g., NVIDIA RTX 3050 Laptop, 4 GB VRAM]` (write "none" if no NVIDIA GPU)
- CUDA Toolkit: `[e.g., 12.6]`
- Visual Studio: `[e.g., 2022 Community 17.x with "Desktop development with C++"]`
- MS-MPI: `[installed / not installed]`
- Python (optional, tooling only): `[e.g., 3.11 / none]`
- Coding tool I'm using: `[e.g., Claude Code / chat]`

Use these specs to choose sensible defaults (thread counts, CUDA architectures, VRAM-safe batch sizes). The app must still detect hardware at runtime and adapt.

**What will be graded:** correct and genuinely parallel implementations of each paradigm, rigorous measurement, honest interpretation of results (including where a method loses), visual quality, and my ability to explain the code in a viva. Optimize for all five.

### 2. Non-negotiable engineering rules

1. **Complete code only.** Every file must be complete and compilable. No pseudo-code, no `// TODO`, no `...`, no "implement similar logic here", no placeholder functions returning dummy values. If a feature is optional, implement it behind a CMake option — do not stub it.
2. **Graceful degradation.** The app must launch and be fully usable on a machine without an NVIDIA GPU, without MS-MPI, or both. Unavailable engines appear greyed out with the exact reason ("No CUDA-capable device detected", "mpiexec not found — install MS-MPI") and never crash the app.
3. **Determinism.** All randomness uses a counter-based Philox4x32-10 generator implemented once in a `__host__ __device__` header, keyed by `(global_seed, symbol_id, stream_id)` and indexed by event counter. The same seed produces the same market and the same decisions on every backend.
4. **Integer prices.** Prices are `int64_t` ticks (tick size configurable, default 0.01), quantities `int32_t`, cash/P&L `int64_t` tick-units. Floating point is used for indicators, ML, and analytics only.
5. **Bit-reproducibility across backends.** Decision-critical math must give identical results on CPU and GPU:
   - Window sums use exact `int64_t` arithmetic; threshold comparisons use integer cross-multiplication where possible (e.g., `fastSum * slowN > slowSum * fastN` instead of comparing two float averages).
   - Any transcendental function that influences generated events or decisions (log, exp, sin, cos) comes from `hft::detmath` — a `__host__ __device__` header implementing them from IEEE basic operations in a fixed evaluation order — never from `std::` or CUDA intrinsics, whose last-bit results differ.
   - Build decision code with MSVC `/fp:precise` and nvcc `--fmad=false -prec-div=true -prec-sqrt=true` so no FMA contraction or reassociation changes results. SIMD in decision paths only for integer or order-preserving operations.
   - Prefer `float` in GPU hot paths (consumer GeForce FP64 throughput is a small fraction of FP32) — determinism comes from operation order, not precision.
   - AI strategies whose inference uses library GEMMs (different summation order) are verified by **decision-agreement rate** (target ≥ 99.9%) rather than strict checksums; say so in the UI.
6. **Hot path hygiene** (optimized engines): no heap allocation, no exceptions, no locks, no virtual calls, no `std::map`, no I/O inside the per-event loop. Pre-allocate everything. The naive engine deliberately violates these (documented and toggleable) so the difference is measurable.
7. **Separation of concerns.** UI never computes; engines never touch UI. Communication is through snapshots and lock-free queues.
8. **Explainability for viva.** Every source file starts with a header comment block: purpose, which syllabus unit/topic it demonstrates, key HPC concepts used, and complexity. Every parallel region has a comment explaining data sharing (shared/private), synchronization, and why it is race-free.
9. **Warnings as signal.** Compile clean at `/W4` (MSVC) and `-Xcompiler /W4` for nvcc host code; treat new warnings as bugs.
10. **Educational honesty.** The market is synthetic. The UI states that results are simulated and not financial advice, and presents GPU latency truthfully (Section 19).
11. **Consistent contracts.** Define the core interfaces (Sections 6, 8.1, 9.1) first. If a contract must change later, state the change explicitly and update every implementer.

### 3. Technology stack (fixed — do not substitute without explaining why)

| Concern | Choice | Notes |
|---|---|---|
| Language | C++20 (host), C++17 for `.cu` files | Keep `.cu` code conservative for nvcc/MSVC compatibility |
| Build | CMake ≥ 3.28 with `CMakePresets.json` | Presets: `windows-release`, `windows-debug`, `windows-release-nocuda`, `linux-release` (core + MPI worker + bench only) |
| Compiler | MSVC (Visual Studio 2022), x64 | Required as the CUDA host compiler on Windows |
| OpenMP | MSVC `/openmp:llvm` | MSVC's default `/openmp` is OpenMP 2.0 (no `collapse`, no tasks, no `min`/`max` reductions). With CMake ≥ 3.30 set `OpenMP_RUNTIME_MSVC=llvm`; otherwise add the flag per-target for CXX only (`$<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>`). Ship `libomp140.x86_64.dll` next to the exe |
| MPI | Microsoft MPI (MS-MPI v10.x SDK + runtime); OpenMPI/MPICH on Linux | `find_package(MPI)`; locate `mpiexec` via `%MSMPI_BIN%`, then `PATH`, then a Settings override. Use MPI-2.2 features plus only MPI-3 calls MS-MPI supports; check `provided` from `MPI_Init_thread` |
| GPU | CUDA Toolkit 12.x; cuBLAS, cuRAND, Thrust, CUB; cuDNN optional (`HFT_ENABLE_CUDNN=OFF` default) | `CMAKE_CUDA_ARCHITECTURES` default `native`, fallback `75;86;89`. Link `cudart_static`. **Delay-load** the cuBLAS/cuRAND DLLs (`/DELAYLOAD:...` + `delayimp.lib`) and probe with `LoadLibraryW` before first use, so the exe starts on non-NVIDIA machines |
| GUI | Dear ImGui (docking branch) + ImPlot + GLFW 3.4 + OpenGL 3.3 backend | Via CMake `FetchContent`, pinned to tags/commits. Chosen for native speed and best-in-class real-time plotting |
| Code viewer | ImGuiColorTextEdit | Syntax-highlighted read-only viewer for embedded engine sources |
| Fonts/icons | Inter (UI), JetBrains Mono (numbers/code), Font Awesome 6 Free Solid + IconFontCppHeaders | Embedded as compressed arrays so the exe is self-contained |
| Utilities | nlohmann/json, spdlog, CLI11, stb_image_write, doctest | All via FetchContent |
| Optional telemetry | NVML (GPU utilization/temperature/power) | `HFT_ENABLE_NVML` auto |
| Optional tooling | Python 3 + NumPy/Matplotlib (+ PyTorch optional) | Only for offline plots/training scripts in `tools/`; the exe never requires Python |

### 4. System architecture

#### 4.1 Processes and binaries

```
┌──────────────────────────────────────────────────────────────────────┐
│ HFT_HPC_Studio.exe  (GUI host, WIN32 subsystem)                       │
│  ├─ Render thread (ImGui/ImPlot @ 60 FPS, never blocks)               │
│  ├─ Simulation thread (live mode: market + engine + strategy)         │
│  ├─ Benchmark thread (headless sweeps, progress + cancel)             │
│  ├─ In-process engines: Serial Naive, Serial Optimized,               │
│  │   Threaded Pipeline, OpenMP, CUDA, CUDA+Libraries                  │
│  └─ MPI Launcher ──CreateProcess──► mpiexec -n N hft_mpi_worker.exe   │
│                     ◄── JSON-lines over stdout pipe + result.json ──  │
├──────────────────────────────────────────────────────────────────────┤
│ hft_mpi_worker.exe  (MPI and Hybrid MPI+OpenMP engines; console)      │
│ hft_bench.exe       (CLI benchmark runner, all engines, CSV/JSON)     │
│ examples/demo_*.exe (small single-file teaching programs per method)  │
│ hft_tests.exe       (doctest unit + cross-backend equivalence tests)  │
└──────────────────────────────────────────────────────────────────────┘
```

MPI ranks cannot live cleanly inside a GUI process, so MPI runs as a separate worker launched through `mpiexec`. The worker emits one JSON object per line (`{"type":"progress",...}`, `{"type":"snapshot",...}`, `{"type":"result",...}`, `{"type":"error",...}`); the GUI parses these to update progress, live charts, and final results. The GUI handles worker crash, timeout, and user cancellation (terminate the whole process tree via a Windows Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`).

#### 4.2 Core libraries

All simulation, strategy, engine, analytics, and AI code lives in a static library `hft_core` (CXX) plus `hft_cuda` (CUDA, optional). The GUI, MPI worker, bench, examples, and tests all link these — one source of truth.

#### 4.3 Per-event pipeline (tick-to-trade)

```
 A. Market Event Gen ─► B. Order Book Update ─► C. Features/Indicators ─► D. Strategy Signal
        ▲                                                                        │
        │                                                                        ▼
 H. Portfolio Analytics ◄─ G. P&L Accounting ◄─ F. Matching/Fill ◄─ E. Risk Check + Order
   (periodic: covariance, CG min-variance, Monte Carlo VaR, regime detection)
```

Each engine times every stage separately (stage-breakdown chart) and end-to-end (tick-to-trade latency = time from event available at stage B to order emitted at stage E).

#### 4.4 Live vs Benchmark modes

- **Live Mode:** visual, throttled to a user-selected simulation speed; the simulation thread publishes UI snapshots at ~30 Hz via a triple buffer. One engine at a time, or **Race Mode**: up to 4 engines on identical event streams in separate threads (cores partitioned among racers), shown as overlaid equity curves.
- **Benchmark Mode:** unthrottled, headless, with warm-up runs, N repetitions, and median/min/p99 reporting; sweeps over symbols, threads, ranks, batch sizes, and CUDA block sizes.

#### 4.5 Verification vs Realistic modes

- **Verification Mode:** compute latency does not affect the market. All engines must produce identical order sequences per symbol. The app computes a 64-bit FNV-1a checksum over every emitted order and fill per symbol and shows ✓/✗ per engine against the Serial Naive reference (decision-agreement rate for GEMM-based AI strategies, per Rule 2.5).
- **Realistic (Latency-Coupled) Mode:** each engine's *measured* decision latency (optionally multiplied by a "latency amplification" slider, 1×–10,000×) is added to the simulated timestamp at which its order reaches the matching engine. Other agents keep trading meanwhile, so slower engines face stale quotes, missed fills, and adverse selection. This makes "speed = money" visible and quantifiable as **latency cost** = P&L(zero latency) − P&L(actual).

### 5. Repository layout

```
HFT-HPC-Studio/
├─ CMakeLists.txt
├─ CMakePresets.json
├─ README.md
├─ LICENSE (MIT) + THIRD_PARTY_NOTICES.md
├─ cmake/            EmbedSources.cmake, EmbedFonts.cmake, Warnings.cmake, CudaConfig.cmake
├─ assets/
│  ├─ fonts/         Inter, JetBrainsMono, fa-solid-900 (embedded at build)
│  ├─ icons/app.ico
│  └─ app.rc, app.manifest (PerMonitorV2 DPI, version info)
├─ configs/          demo.json, latency_showdown.json, gpu_crossover.json, ai_vs_classic.json, full_sweep.json
├─ include/hft/
│  ├─ core/          types.hpp, config.hpp, rng_philox.hpp, detmath.hpp, timer.hpp, spsc_ring.hpp,
│  │                 triple_buffer.hpp, aligned_alloc.hpp, pool.hpp, checksum.hpp, sysinfo.hpp
│  ├─ market/        agents.hpp, order_book_naive.hpp, order_book_ladder.hpp,
│  │                 market_generator.hpp, hawkes.hpp, regimes.hpp, correlation.hpp
│  ├─ strategy/      istrategy.hpp, features.hpp, risk.hpp, traditional/*.hpp, ai/*.hpp
│  ├─ engine/        iengine.hpp, engine_registry.hpp, run_config.hpp, run_result.hpp
│  ├─ analytics/     hpc_metrics.hpp, trading_metrics.hpp, latency_histogram.hpp,
│  │                 covariance.hpp, cg_solver.hpp, montecarlo_var.hpp, kmeans.hpp, lttb.hpp
│  ├─ ai/            mlp.hpp, trainer.hpp, qlearning.hpp, bandit.hpp, regime_detector.hpp
│  ├─ cloud/         latency_model.hpp, cost_model.hpp
│  └─ ipc/           json_lines.hpp, mpi_launcher.hpp
├─ src/
│  ├─ core/ market/ strategy/ analytics/ ai/ cloud/ ipc/        (.cpp files)
│  ├─ engines/
│  │  ├─ serial_naive_engine.cpp
│  │  ├─ serial_optimized_engine.cpp
│  │  ├─ threaded_pipeline_engine.cpp
│  │  ├─ openmp_engine.cpp
│  │  ├─ openmp_kernels.cpp          (covariance tiled, CG, Cholesky, k-means, MLP, sync micro-bench)
│  │  ├─ mpi/mpi_worker_main.cpp     (MPI + hybrid; builds hft_mpi_worker.exe)
│  │  ├─ mpi/mpi_engine.cpp, mpi_cg.cpp, mpi_sgd.cpp, mpi_types.cpp
│  │  └─ cuda/
│  │     ├─ cuda_engine.cu, cuda_kernels.cu, cuda_kernels.cuh
│  │     ├─ cuda_covariance.cu       (naive / tiled shared-memory / cuBLAS)
│  │     ├─ cuda_cg.cu, cuda_montecarlo_var.cu (cuRAND), cuda_kmeans.cu
│  │     ├─ cuda_mlp.cu              (cuBLAS GEMM forward/backward + custom kernels)
│  │     ├─ cuda_qlearning.cu, cuda_backtest_sweep.cu
│  │     ├─ cuda_lstm_cudnn.cu       (only if HFT_ENABLE_CUDNN)
│  │     └─ cuda_utils.cuh           (CUDA_CHECK, CUBLAS_CHECK, event timers, device query)
│  └─ app/
│     ├─ main.cpp, App.hpp/.cpp, Theme.hpp/.cpp, Fonts.cpp, Widgets.hpp/.cpp
│     ├─ screens/  HomeScreen, MarketScreen, StrategyScreen, EngineScreen, ResultsScreen,
│     │            CloudAILabScreen, LearnScreen, SettingsScreen (.hpp/.cpp each)
│     ├─ plots/    Candlestick, DepthChart, OrderBookHeatmap, RadarChart, LatencyPlots, Gauge (.cpp)
│     └─ services/ SimulationService, BenchmarkService, MpiService, ExportService, Toasts
├─ apps/bench/hft_bench_main.cpp
├─ examples/  demo_serial.cpp, demo_openmp.cpp, demo_mpi.cpp, demo_cuda.cu, demo_cublas.cu
├─ tests/     test_rng.cpp, test_detmath.cpp, test_order_book.cpp, test_strategies.cpp,
│             test_equivalence.cpp, test_metrics.cpp, test_cg.cpp, test_mlp.cpp
├─ tools/     plot_results.py, train_lstm_pytorch.py (optional), export_weights.py
├─ scripts/   build.bat, run_mpi_demo.bat, run_benchmarks.bat, package.bat
├─ installer/ HFT_HPC_Studio.iss (Inno Setup) or CPack NSIS config
└─ docs/      ARCHITECTURE.md, REPORT_TEMPLATE.md, VIVA_NOTES.md, BENCHMARK_GUIDE.md, CLOUD_RUN.md, screenshots/
```

### 6. Core data model and utilities

```cpp
namespace hft {
using Price   = int64_t;   // integer ticks
using Qty     = int32_t;
using Cash    = int64_t;   // tick-units * qty
using SimTime = uint64_t;  // nanoseconds of simulated time

enum class Side      : uint8_t { Buy = 0, Sell = 1 };
enum class EventType : uint8_t { LimitAdd, Cancel, MarketOrder, Trade };
enum class AgentType : uint8_t { Noise, MarketMaker, Momentum, Fundamental, Institutional, Strategy };

// Exactly 32 bytes (two per 64-byte cache line) — static_assert(sizeof(MarketEvent) == 32).
struct alignas(32) MarketEvent {
  SimTime   ts;        // 8
  Price     price;     // 8
  uint32_t  order_id;  // 4  (per-symbol id space)
  uint32_t  symbol;    // 4
  Qty       qty;       // 4
  Side      side;      // 1
  EventType type;      // 1
  AgentType agent;     // 1
  uint8_t   flags;     // 1
};

struct Order     { uint32_t id; uint32_t symbol; Side side; Price px; Qty qty; SimTime created, arrives; uint8_t tif; };
struct Fill      { uint32_t order_id; uint32_t symbol; Side side; Price px; Qty qty; SimTime ts; bool aggressive; };
struct TopOfBook { Price bid, ask; Qty bid_qty, ask_qty; Price last; SimTime ts; };
struct Position  { Qty qty; Cash cash; Price avg_px; Cash realized; };
}
```

Also define SoA mirrors (`EventBatchSoA`, `SymbolStateSoA`) for OpenMP-vectorizable and CUDA-coalesced access, with conversion helpers.

Utilities (all complete, all tested):

- **Philox4x32-10** `__host__ __device__` inline, plus `uniform01`, `normal` (Box–Muller via `detmath`), `exponential`, `poisson` (inversion for small λ, normal approximation for large), all deterministic.
- **detmath**: portable deterministic `log`, `exp`, `sin`, `cos`, `pow` built from IEEE basic operations in fixed order; accuracy tests against `std::` (≤ a few ulp) and host/device equality tests.
- **Timers:** `TscClock` (`__rdtsc` calibrated against `QueryPerformanceCounter` at startup, invariant-TSC check), `steady_clock` fallback, `ScopedStageTimer`, `CudaEventTimer`; thin wrappers around `omp_get_wtime` and `MPI_Wtime` used in their engines (and explained in comments).
- **`SpscRing<T, N>`**: lock-free single-producer/single-consumer, power-of-two capacity, `alignas(64)` head/tail to avoid false sharing, acquire/release atomics.
- **`TripleBuffer<T>`** for UI snapshots; **`AlignedVector<T, 64>`**; **`ObjectPool<T>`** for orders.
- **`LatencyHistogram`**: HDR-style log-linear buckets from 10 ns to 10 s with ≤ 1% relative error, mergeable across threads/ranks, percentiles p50/p90/p99/p99.9/p99.99/max, mean, stddev.
- **`SysInfo`**: CPU brand (`__cpuid`), physical/logical cores (`GetLogicalProcessorInformationEx`), L1/L2/L3 sizes, RAM, OS version, AVX2/AVX-512 support, OpenMP version (`_OPENMP` value, `omp_get_max_threads`), MPI availability and version, CUDA devices (name, compute capability, SM count, estimated CUDA cores, clocks, VRAM, bus width, L2, driver/runtime versions).
- **`RunConfig` / `RunResult`** JSON (de)serialization — the single schema used by GUI, bench, and MPI worker (full schema documented in `docs/ARCHITECTURE.md`).

### 7. Module 1 — Dynamic Market Data Generator

#### 7.1 Model — agent-based limit order book where price emerges from order flow

Each symbol has its own order book and population of agents. Price is **not** drawn from a formula: it emerges from matched buy and sell orders, with a latent fundamental value as the anchor.

- **Fundamental value** per symbol — Merton jump-diffusion in log space:
  `ln V(t+Δ) = ln V(t) + (μ − σ²/2 − λκ)Δ + σ√Δ·Z + J·N`, with `N ~ Poisson(λΔ)`, `J ~ Normal(μ_J, σ_J)`, `κ = E[e^J − 1]`.
  Cross-symbol correlation: shocks `Z = L·ε`, where `L` is the Cholesky factor of a configurable correlation matrix (sector blocks, e.g., 4 sectors, intra-sector ρ = 0.6, inter-sector ρ = 0.2). The Cholesky factorization itself is a small matrix-computing demo (serial + OpenMP).
- **Order arrivals:** Hawkes self-exciting process per symbol, `λ(t) = μ₀ + Σᵢ α·e^(−β(t − tᵢ))`, enforcing `α/β < 1`; simulated by Ogata thinning with Philox. Produces realistic bursts of activity (and therefore realistic load imbalance across symbols).
- **Agents** (each with population sliders and parameters):
  - *Noise traders* — random side, market or limit orders, limit offset from the best quote drawn from a geometric distribution, per-order cancellation hazard (zero-intelligence / Santa Fe model).
  - *Market makers* — two-sided quotes around the microprice, inventory-skewed, spread widening with volatility, cancel/replace on book changes.
  - *Momentum traders* — buy after short-horizon up-moves, sell after down-moves (EMA-slope threshold).
  - *Fundamental (value) traders* — buy when price < V·(1 − band), sell when price > V·(1 + band).
  - *Institutional* — occasional large parent orders sliced TWAP-style (creates visible trends and pressure).
  - *The user's strategy* — the agent whose orders come from the chosen strategy + engine.
- **Order book** (two implementations, same semantics, price-time priority):
  - `OrderBookNaive`: `std::map<Price, std::deque<Order>>` per side + `std::unordered_map` id → location. Deliberately allocation-heavy (Unit 1 baseline).
  - `OrderBookLadder`: dense array of price levels centered on a re-centerable anchor, each level an intrusive FIFO list over a preallocated order pool, best-bid/ask indices, O(1) add/cancel, a bitset for fast best-level search. This is how real low-latency books are built.
  - **Ladder depth is a `RunConfig` parameter shared by every engine** (default 1024 levels; smaller on low-VRAM GPUs). Out-of-range limit orders are rejected by an identical rule in both book implementations, so Verification Mode holds.
- **Market regimes** — one-click presets that retune agent mix and parameters: Calm, Bull Trend, Bear Trend, Sideways/Mean-Reverting, High Volatility, Flash Crash (liquidity withdrawal + large sell program + recovery), News Shock (jump). Regimes can also switch automatically via a hidden Markov chain, so the AI regime detector has something real to detect.
- **Optional CSV replay:** load `timestamp,price,volume` (or OHLCV bars) to drive the fundamental value, so real-looking paths can be replayed through the identical pipeline.
- **Scale:** 1 to 4096 symbols; events generated in chunks (e.g., 64K events) so memory stays ≤ 1.5 GB regardless of run length; generation throughput is itself measured.

#### 7.2 Market screen visual requirements (Screen ① "Market")

- **Candlestick chart** (custom ImPlot renderer: bodies and wicks, green up / red down, hover tooltip with OHLCV), timeframe selector in *simulated* time (100 ms, 1 s, 5 s, 1 min bars), linked volume subplot (bars split by buy vs sell aggressor volume), overlay toggles: SMA, EMA, Bollinger bands, VWAP, and the fundamental value (dashed, "reveal hidden truth" toggle).
- **Order book ladder:** 20 levels per side, size bars behind the numbers, bids green / asks red, cells flash on change (fade ≈ 400 ms), spread and microprice in the middle row.
- **Depth chart:** cumulative bid/ask area plot.
- **Liquidity heatmap:** time × price heatmap of resting size (ImPlot heatmap), with trade prints overlaid as bubbles sized by volume (Bookmap-style).
- **Time & Sales tape:** scrolling trades (time, price, size, aggressor side, agent-type icon).
- **Buy/Sell pressure gauge:** semicircular gauge of aggressor-volume imbalance over a rolling window.
- **Market stats strip:** last, change, change %, high, low, VWAP, spread, volume, trades/s, events/s (simulated and wall-clock).
- **Agent control panel:** sliders for every agent population and parameter, volatility, drift, jump intensity, Hawkes μ₀/α/β, with "Apply live".
- **Controls:** symbol selector, Play/Pause/Step/Reset, speed slider (0.1×–1000× and "Max"), seed field, regime preset buttons.
- **Market Overview tab:** grid of symbol tiles (price, % change, colored sparkline, mini volume bar), sortable, click to focus.

### 8. Module 2 — Strategy Lab

#### 8.1 Strategy contract

```cpp
struct FeatureVector { float f[32]; };  // fixed-size, normalized
struct Signal {
  int8_t direction;          // -1, 0, +1
  Qty    qty;
  Price  limit_px;
  bool   is_quote;           // market-making: two-sided quote
  Price  bid_px, ask_px;
  float  confidence;
};

class IStrategy {
public:
  virtual ~IStrategy() = default;
  virtual const char*  name() const = 0;
  virtual StrategyKind kind() const = 0;                       // Traditional / AI / Selector
  virtual void   reset(const StrategyParams&, uint32_t num_symbols) = 0;
  virtual Signal onEvent(uint32_t sym, const TopOfBook&, const FeatureVector&, const Position&) = 0;
  // Batch path for OpenMP/CUDA-friendly engines — must be semantically identical to onEvent:
  virtual void   onBatch(const BatchView&, Signal* out) = 0;
  virtual std::string explain() const = 0;                     // formula + intuition for the UI
};
```

Strategies must be pure functions of (state, inputs), with per-symbol state in SoA arrays owned by the engine, so they run per-symbol in parallel and port to CUDA as `__device__` functions. Put the decision math in shared `__host__ __device__` inline headers so CPU and GPU decisions are identical by construction.

#### 8.2 Traditional strategies (implement all, each with parameter UI, formula card, and defaults)

| Strategy | Core rule |
|---|---|
| SMA Crossover | Long when SMA_fast > SMA_slow, short/flat when below (integer cross-multiplied comparison) |
| EMA Crossover | `EMA_t = α·p_t + (1−α)·EMA_{t−1}`, `α = 2/(N+1)`; crossover signals |
| Mean Reversion (Bollinger / z-score) | `z = (p − μ_N)/σ_N`; enter against the move at `|z| > k`, exit at `|z| < k_exit` |
| Momentum / Breakout | Enter on break of the N-period high/low with volume confirmation |
| RSI | `RSI = 100 − 100/(1 + RS)`; buy < 30, sell > 70 (configurable) |
| MACD | `MACD = EMA12 − EMA26`, signal = EMA9(MACD); crossover signals |
| Order Book Imbalance | `I = (V_bid − V_ask)/(V_bid + V_ask)` over top-k levels; microprice `(P_ask·V_bid + P_bid·V_ask)/(V_bid + V_ask)`; trade when imbalance exceeds a threshold (the most "HFT-native" signal) |
| Market Making (Avellaneda–Stoikov) | Reservation price `r = s − q·γ·σ²·(T − t)`; optimal total spread `δ = γσ²(T − t) + (2/γ)·ln(1 + γ/k)`; quote `r ± δ/2` with inventory limits |
| Statistical Arbitrage (Pairs) | Rolling OLS hedge ratio β, spread `y − βx`, z-score entry ±2, exit at 0; pairs chosen within a sector. **Parallelizes over pairs, not symbols; the MPI partitioner keeps both legs of a pair on the same rank** |
| VWAP / TWAP Execution | Execute a parent order over a horizon; measure implementation shortfall vs arrival price |

#### 8.3 AI strategies (Unit 4)

1. **MLP direction classifier** — 32 input features (Section 8.4); architecture 32→64→32→3 (down/flat/up), ReLU, softmax; label = sign of mid-price change over a horizon of H events with a dead-band. Trained in-app with mini-batch SGD/Adam on simulator-generated data. **Training and inference implemented on Serial CPU, OpenMP, MPI (data-parallel SGD), and CUDA (cuBLAS GEMMs + custom activation/softmax/loss kernels)**, benchmarked against each other (epoch time, samples/s, inference latency at batch 1 vs batch 1024+). Weights saved/loaded as versioned binary + JSON metadata.
2. **Q-learning market maker (reinforcement learning)** — state: discretized (inventory, spread, imbalance, volatility, time-in-episode); actions: quote skew/width combinations (e.g., 9 actions); reward: Δ(mark-to-market P&L) − η·q². Tabular Q with ε-greedy exploration and decay. Training runs thousands of independent simulated episodes **in parallel** (OpenMP: per-thread Q-tables with periodic averaging; CUDA: one episode per thread with per-thread tables in global memory, averaged by a reduction kernel). Show the learning curve, a Q-value heatmap, and the learned policy.
3. **AI Strategy Selector (meta-controller)** — every W events, compute regime features (realized volatility, trend strength via EMA slope, lag-1 return autocorrelation, Hurst-exponent estimate, spread, imbalance). **Regime detection:** k-means (k = 4–6) on historical feature windows (parallel k-means on OpenMP and CUDA). **Selection:** one contextual bandit per regime (UCB1 and Thompson Sampling selectable) over the classical strategies, reward = the chosen strategy's realized P&L over the next window. Display the current regime, distances to each centroid, the chosen strategy, confidence, and a readable "Why this strategy?" explanation.
4. **Optional (`HFT_ENABLE_CUDNN`):** LSTM/GRU next-move model via cuDNN, plus `tools/train_lstm_pytorch.py` that trains offline and exports weights the app can load. The app must work fully without this.

#### 8.4 Feature engineering (shared by AI strategies and regime detection)

Returns over lags {1, 5, 10, 50, 100}; EMA spreads; rolling volatility (short/long); RSI; MACD histogram; Bollinger %b; spread; top-1 and top-5 imbalance; microprice − mid; signed trade-flow imbalance over windows; order-arrival rate; cancel rate; time since last trade; inventory; unrealized P&L. Online normalization (Welford running mean/variance). Implemented serial, OpenMP (across symbols), and CUDA (one thread per symbol, SoA state).

#### 8.5 Risk manager (identical in every engine)

Max position per symbol, max gross exposure, max order size, max orders/second (token bucket), price collar (reject orders more than X ticks from mid), max-loss kill switch (flattens positions, halts, shows a red banner), self-trade prevention.

#### 8.6 Strategy screen visual requirements (Screen ② "Strategy")

- **Left:** strategy list grouped Traditional / AI / Selector, with icons and one-line descriptions.
- **Center:** parameter editor (sliders with sensible ranges and tooltips), formula card (monospace, neatly typeset), "How it works" explanation, and a **quick-backtest preview** (last N events on the Serial Optimized engine, mini equity curve + key stats in under 1 s).
- **Right (AI):** model status (untrained/trained/loaded), training-backend selector (Serial/OpenMP/CUDA), Train button, live loss and accuracy curves, confusion-matrix heatmap, permutation feature-importance bars; for Q-learning, the reward curve and policy heatmap; for the Selector, a regime scatter (2-D PCA projection colored by cluster) and bandit arm statistics.
- **Bottom:** risk settings panel.

### 9. Module 3 — Execution Engines (the HPC core)

#### 9.1 Engine contract

```cpp
enum class EngineKind { SerialNaive, SerialOptimized, ThreadedPipeline, OpenMP, MPI, HybridMPIOpenMP, CUDA, CUDALibs };

struct EngineCaps { bool available; std::string reason; bool supports_live; bool supports_latency_mode; int max_workers; };

class IEngine {
public:
  virtual ~IEngine() = default;
  virtual EngineKind  kind() const = 0;
  virtual const char* name() const = 0;
  virtual const char* syllabusUnit() const = 0;                 // "Unit 1", "Unit 2", ...
  virtual EngineCaps  probe() const = 0;                        // hardware/runtime detection, never throws
  virtual void        configure(const RunConfig&) = 0;          // allocate everything here
  virtual RunResult   run(IStrategy&, MarketSource&, ProgressSink&, const std::atomic<bool>& cancel) = 0;
  virtual void        release() = 0;
  virtual std::vector<std::string> hotPathSourceFiles() const = 0; // shown in the Code Viewer
};
```

`RunResult` contains: per-stage timings, the end-to-end latency histogram, throughput counters, per-worker timings (threads/ranks/blocks), engine-specific breakdowns (GPU H2D/kernel/D2H, MPI compute/communication/wait, OpenMP imbalance), trading results (fills, equity series, positions), checksums, a hardware snapshot, and the full `RunConfig` for reproducibility.

#### 9.2 The engines

**E1 — Serial Naive CPU (Unit 1 baseline).** Single thread, `OrderBookNaive`, AoS data, a virtual strategy call per event, heap-allocated orders, unreserved `std::vector` growth, indicators recomputed from scratch over the whole window (O(N) per event instead of O(1) incremental). Each inefficiency is commented and individually toggleable so the Learn screen can show its individual cost.

**E2 — Serial Optimized CPU ("how real HFT runs on one core", Unit 1).** Single thread pinned to one core (`SetThreadAffinityMask`, `SetThreadPriority(THREAD_PRIORITY_TIME_CRITICAL)`), busy-poll loop, `OrderBookLadder`, preallocated pools, SoA state, incremental O(1) indicators, branch-light code with `[[likely]]`/`[[unlikely]]`, cache-line alignment, software prefetching, and AVX2 intrinsics for batched integer/order-preserving math where supported (runtime-dispatched with scalar fallback). Tick-to-trade measured with calibrated RDTSC. This engine should win the single-symbol latency benchmark; the UI should say so and explain why.

**E3 — Multithreaded Pipeline CPU (Unit 1: shared memory, OS scheduling).** `std::thread` stages — Feed Handler → Book Builder → Strategy → Risk/OMS → Matching/P&L — connected by `SpscRing` queues, each stage pinned to its own core. Demonstrates pipeline vs data parallelism, queueing latency, and cache-coherence traffic. Also a "symbol-sharded" variant (one thread per symbol group) for throughput. Report per-stage queue depths and wait times.

**E4 — OpenMP (Unit 2: shared memory).**
- Data parallelism across symbols: `#pragma omp parallel for schedule(runtime)`, with the schedule (static / dynamic,chunk / guided) selectable in the UI and compared in a chart. Hawkes bursts make symbols unequally busy, so dynamic scheduling visibly matters.
- Data-sharing clauses used deliberately and commented: `default(none)`, `shared`, `private`, `firstprivate`, `lastprivate`, `reduction(+:pnl) reduction(max:peak)`.
- **Synchronization micro-benchmark (Scenario S7):** aggregate portfolio P&L via (a) `critical`, (b) `atomic`, (c) `reduction`, (d) per-thread padded partials; plus a **false-sharing demo** (unpadded vs `alignas(64)` per-thread counters). Chart the cost of each.
- **Efficient OpenMP for matrices:** covariance of returns `Σ = XᵀX/(T − 1)` — naive triple loop vs cache-blocked/tiled loops with `collapse(2)` vs transposed layout; report GFLOPS.
- Parallel Conjugate Gradient (Section 9.3), Cholesky for correlated shocks, parallel k-means, MLP training/inference, Q-learning episodes, Monte Carlo VaR.
- Thread sweep 1 → max logical threads for speedup/efficiency/Karp–Flatt charts. Use `omp_get_wtime()`; show `omp_get_num_procs()`, `omp_get_max_threads()`, and the `_OPENMP` version in System Info.

**E5 — MPI (Unit 2: distributed memory, domain decomposition).** Built as `hft_mpi_worker.exe`.
- **Domain decomposition:** symbols partitioned across ranks, block or cyclic (selectable), pair-aware for the Pairs strategy; per-rank load-balance chart. Each rank owns its symbols' order books and strategy state — no shared memory.
- **Communication (all used, all commented):** `MPI_Bcast` (config, strategy params, model weights), `MPI_Scatterv` (symbol assignments, initial prices), `MPI_Allreduce` (portfolio P&L and gross exposure for global risk limits every K events), `MPI_Gatherv` (fills, equity series, histograms to rank 0), `MPI_Isend`/`MPI_Irecv`/`MPI_Waitall` (Exchange Topology mode), `MPI_Barrier` (only to delimit timed phases).
- **Derived datatypes:** `MPI_Type_create_struct` for `Order`, `Fill`, and `MarketEvent`, committed once and freed at shutdown.
- **Exchange Topology mode:** rank 0 acts as a central exchange/matching gateway; strategy ranks send orders via non-blocking point-to-point, rank 0 matches and returns fills. Demonstrates a communication bottleneck vs the fully decomposed mode; chart both.
- **Parallel CG solver** with row-block decomposition of Σ: the local mat-vec needs the full direction vector → `MPI_Allgatherv`; dot products → `MPI_Allreduce`.
- **Data-parallel SGD** for the MLP: each rank computes gradients on its data shard, `MPI_Allreduce` averages them — the textbook "training AI on HPC" pattern (highlight for Unit 4).
- Timing with `MPI_Wtime()`; per-rank compute vs communication vs wait; strong scaling (fixed symbols, ranks 1 → N) and weak scaling (symbols ∝ ranks).
- The UI explains that on one PC the ranks are separate processes communicating through MS-MPI's shared-memory transport — the programming model is distributed memory even on one machine, and the same binary runs across cloud VMs (Section 11).

**E6 — Hybrid MPI + OpenMP.** `MPI_Init_thread(MPI_THREAD_FUNNELED)` (verify `provided`); ranks decompose symbols, OpenMP threads parallelize within each rank. The UI lets the user choose ranks × threads (e.g., 1×8, 2×4, 4×2, 8×1) and plots a heatmap of time at equal total cores.

**E7 — CUDA custom kernels (Unit 3).**
- Execution model made explicit: grid/block configuration from the UI (block size 64–1024) plus the `cudaOccupancyMaxPotentialBlockSize` suggestion; show theoretical occupancy.
- Kernels: `generateEventsKernel` (Philox per symbol), `bookUpdateKernel` (one thread per symbol, sequential per-symbol ladder in global memory — documented as the branchy, GPU-unfriendly part), `featureKernel` (SoA, coalesced), `signalKernel` (shared `__host__ __device__` strategy math), `riskKernel`, `pnlKernel` with warp-shuffle (`__shfl_down_sync`) + shared-memory block reduction; portfolio totals via CUB `DeviceReduce`; cumulative P&L via Thrust `inclusive_scan`.
- **Matrix computing in CUDA:** covariance via naive global-memory kernel vs shared-memory tiled kernel (TILE = 16/32), compared against cuBLAS in E8.
- **Memory techniques, each toggleable and measured:** pageable vs pinned (`cudaHostAlloc`) transfers; single stream vs multi-stream double-buffered pipeline overlapping H2D/compute/D2H; **CUDA Graphs** to cut launch overhead on the small-batch latency path.
- **Batching semantics:** the GPU processes events in batches (configurable batch size or time window). Per-event latency = time waiting in the batch + H2D + kernels + D2H. The batch-size sweep produces the **CPU-vs-GPU crossover chart**.
- Timing with `cudaEvent` pairs; report kernel time, transfer time, effective bandwidth (GB/s), and GFLOPS where meaningful.
- Robust error handling: `CUDA_CHECK` surfaces the message in the UI, disables the engine, and never crashes the app.

**E8 — CUDA + NVIDIA libraries (Unit 3: cuBLAS, cuRAND, cuDNN).** Same pipeline as E7 with heavy math through libraries: `cublasSgemm`/`cublasSsyrk` for covariance, cuBLAS GEMM for batched MLP inference and training, cuBLAS level-1/2 (`dot`, `axpy`, `gemv`) for CG, cuRAND (Philox or MRG32k3a) for Monte Carlo VaR with 1M+ correlated paths, optional cuDNN LSTM. Shows "library vs hand-written kernel" performance for identical math.

#### 9.3 Shared heavy kernels (each implemented on multiple backends)

| Kernel | Serial | Optimized | OpenMP | MPI | CUDA custom | CUDA libs |
|---|---|---|---|---|---|---|
| Market event generation (Philox, Hawkes, agents) | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| Order book update & matching | ✓ naive | ✓ ladder | ✓ | ✓ | ✓ | — |
| Features / indicators | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| Strategy signals (all strategies) | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| Covariance matrix Σ (N symbols) | ✓ | ✓ blocked | ✓ tiled | ✓ row-block | ✓ naive + tiled | ✓ cuBLAS |
| Min-variance portfolio: solve `(Σ + λI)x = 1`, `w = x/(1ᵀx)` by Conjugate Gradient | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ cuBLAS |
| Monte Carlo VaR/CVaR (correlated paths via Cholesky) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ cuRAND |
| k-means regime detection | ✓ | — | ✓ | ✓ | ✓ | — |
| MLP training + inference | ✓ | ✓ | ✓ | ✓ data-parallel SGD | ✓ | ✓ cuBLAS |
| Q-learning parallel episodes | ✓ | — | ✓ | ✓ | ✓ | — |
| Strategy parameter-sweep backtest | ✓ | ✓ | ✓ | ✓ | ✓ | — |

#### 9.4 Benchmark scenarios

| ID | Scenario | What it proves | Expected winner (hypothesis — report what actually happens) |
|---|---|---|---|
| S1 | Tick-to-trade latency, 1 symbol, per event | Latency vs overhead | Serial Optimized CPU; GPU loses badly at batch = 1 (PCIe + launch cost); CUDA Graphs narrows the gap |
| S2 | Multi-symbol throughput (64 → 4096 symbols) | Data parallelism, scaling | OpenMP/Hybrid at moderate N; CUDA at large N and large batches |
| S3 | Strategy parameter-sweep backtest (e.g., 32×32 SMA grid × 256 symbols) | Embarrassingly parallel throughput | CUDA by a wide margin |
| S4 | Portfolio risk: Σ + CG + Monte Carlo VaR | Matrix computing, parallel solvers, libraries | cuBLAS/cuRAND; OpenMP tiled vs naive shows cache effects |
| S5 | AI inference (batched MLP) | Batch size vs latency | CPU at batch 1, GPU at large batch |
| S6 | AI training (MLP epochs, Q-learning episodes) | HPC for AI (Unit 4) | CUDA; MPI data-parallel SGD shows scaling |
| S7 | OpenMP synchronization & false sharing | Data-sharing and sync costs | reduction / padded partials ≫ critical |
| S8 | MPI strong & weak scaling; decomposed vs central-exchange topology | Distributed memory, communication cost | Decomposed; central exchange saturates |

The dashboard reports what actually happened on this machine; the Learn screen explains any result that contradicts the hypothesis.

#### 9.5 Engine screen visual requirements (Screen ③ "Engine")

- **Card grid of the 8 engines:** icon, name, color-coded unit badge, one-line description, availability (✓ green / ✗ red with reason), expandable config (threads slider, OpenMP schedule + chunk, MPI ranks + partitioning + topology, ranks × threads for hybrid, CUDA block size, batch size, streams on/off, pinned on/off, CUDA Graphs on/off, custom vs library).
- **Multi-select** engines for comparison. **Workload panel:** symbols, events per symbol, batch size, scenario preset (S1–S8), repetitions, warm-up runs, mode (Verification / Latency-Coupled), latency amplification.
- **Buttons:** "Run Live" (single engine or Race of up to 4), "Run Benchmark Sweep", "Quick Demo (≈ 30 s)". Progress bar with ETA, per-engine status lights, Cancel.
- **Hardware panel:** CPU/GPU/MPI summary from `SysInfo`.

### 10. Module 4 — Metrics and Results Dashboard

#### 10.1 Measurement methodology (implement exactly; document in the Learn screen)

- Warm-up runs discarded (default 2); repetitions (default 5); report median, min, and p99 across repetitions with min–max error bars.
- Clocks: calibrated RDTSC for per-event CPU latency; `omp_get_wtime` in OpenMP; `MPI_Wtime` in MPI; `cudaEventElapsedTime` for GPU segments plus host wall-clock end-to-end; `steady_clock` for totals. Record and display each clock's resolution.
- Latency histogram per engine, merged across threads/ranks. Per-event timing overhead is measured, reported, and subtracted.
- Every result records hardware, config, git hash (if available), and timestamp.

#### 10.2 HPC performance metrics (Unit 1)

- Execution time T(p); throughput (events/s, orders/s, symbol-events/s).
- Latency: mean, stddev (jitter), p50, p90, p99, p99.9, p99.99, max.
- Speedup `S(p) = T(1)/T(p)` against both Serial Naive and Serial Optimized (show both and explain why baseline choice matters).
- Efficiency `E(p) = S(p)/p`; cost `C(p) = p·T(p)`; overhead `T_o = p·T(p) − T(1)`.
- Amdahl fit `S(p) = 1/((1 − f) + f/p)`: estimate the parallel fraction f from measurements and overlay the predicted curve.
- Gustafson `S(p) = p − α(p − 1)` for weak-scaling runs.
- Karp–Flatt experimentally determined serial fraction `e = (1/S − 1/p)/(1 − 1/p)` plotted vs p (rising e ⇒ growing overhead).
- Strong and weak scaling curves; load imbalance = max worker time / mean worker time.
- GPU: kernel vs transfer time, effective bandwidth, GFLOPS, occupancy, speedup vs CPU at each batch size.
- MPI: compute / communication / wait per rank, bytes and messages sent.
- Optional energy: GPU power via NVML integrated over the run (J) and J/event.

#### 10.3 Trading metrics

Net P&L (realized + unrealized), gross P&L, fees (configurable maker/taker fees or rebates), cumulative return, Sharpe `mean(r)/std(r)·√periods` (risk-free ≈ 0, period defined in simulated time), Sortino (downside deviation), max drawdown `max_t (peak_t − equity_t)/peak_t`, Calmar, win rate, profit factor (gross profit / gross loss), average win/loss, number of trades, fill ratio (filled/submitted), cancel ratio, average slippage vs decision-time mid, implementation shortfall (VWAP/TWAP), inventory statistics, turnover, historical and Monte Carlo VaR/CVaR (95%, 99%), **latency cost** (Section 4.5), and P&L sensitivity per microsecond of latency (from sweeping latency amplification).

#### 10.4 Results screen visual requirements (Screen ④ "Results")

- **KPI card row** with animated counters and ▲/▼ deltas vs baseline: Net P&L, Sharpe, Max DD, Trades, p99 latency, Throughput, Best speedup.
- **Trading tab:** equity curves (one line per engine, engine colors), drawdown (shaded), price chart with buy ▲ / sell ▼ markers, inventory over time, P&L distribution histogram, per-symbol P&L bars, trade blotter (sortable, filterable).
- **Latency tab:** histogram (log x), percentile ladder (p50 → p99.99, log y), CDF, jitter timeline, stage-breakdown stacked bars per engine.
- **Throughput & Scaling tab:** throughput bars, speedup vs p with ideal line + Amdahl fit, efficiency vs p, Karp–Flatt vs p, strong vs weak scaling, OpenMP schedule comparison, hybrid ranks × threads heatmap.
- **GPU Deep-Dive tab:** H2D/kernel/D2H stacked bars, block-size sweep, pinned vs pageable, streams on/off, CUDA Graphs on/off, custom vs cuBLAS, and the **CPU-vs-GPU crossover chart** (batch size on x; per-event latency and throughput on twin y-axes; crossover point annotated).
- **MPI Deep-Dive tab:** per-rank compute/comm/wait stacked bars, load balance, decomposed vs central-exchange comparison, scaling.
- **AI tab:** training time per backend, inference latency vs batch size, model accuracy, RL learning curve, the Selector's regime timeline (colored band under the price chart) and strategy switches.
- **Leaderboard tab:** sortable table of every metric for every engine, best cell per column highlighted, custom **radar chart** (latency, throughput, efficiency, P&L, Sharpe, cost) per engine.
- **Verification tab:** per-engine checksums vs reference, ✓/✗, decision-agreement rates, max floating-point deviation in analytics.
- **Auto-insights panel:** 3–6 plain-English findings computed from the actual results (e.g., "OpenMP with 8 threads reached 5.9× speedup; Karp–Flatt rose from 0.02 to 0.06, indicating growing synchronization overhead"). Never hard-coded.
- **Export:** CSV (all series), JSON (full `RunResult`), PNG of any plot (render region → `stb_image_write`), and a one-click **Report Pack** folder with all CSVs, PNGs, and a Markdown summary ready for the course report.

### 11. Module 5 — Cloud & AI-on-HPC Lab (Unit 4) (Screen ⑤ "Cloud & AI")

- **Service-model explorer:** interactive cards for IaaS, PaaS, and SaaS showing what the user manages vs what the provider manages *for this application* (IaaS = GPU VM running our MPI worker; PaaS = managed batch/Kubernetes job service running containerized benchmarks; SaaS = a hosted results dashboard), with benefits and challenges (cost, elasticity, latency, data security, vendor lock-in).
- **Network-latency simulator (cloud networking challenges):** deployment presets — Co-located at exchange, Same-city data center, Same-region cloud, Cross-region cloud — each with a configurable one-way latency distribution (defaults clearly labeled "illustrative" and editable). Re-runs the Latency-Coupled simulation with network latency added to compute latency and charts P&L and fill ratio per deployment. Teaches why HFT firms pay for colocation.
- **Cost estimator:** user-editable hourly prices for CPU and GPU instance types; estimates the cost of each benchmark run (runtime × instances) and cost per million events; cost-vs-speed Pareto chart across engines.
- **Cloud storage & security panel:** where simulation data and results would live (object vs block vs file storage), encryption at rest and in transit, IAM least privilege, and why trading data is sensitive — short, accurate text.
- **Real cloud run (optional, fully supported):** the `linux-release` preset builds `hft_mpi_worker` and `hft_bench` on Linux with OpenMPI/MPICH (+ CUDA if present). `docs/CLOUD_RUN.md` gives step-by-step instructions for a multi-node MPI job on cloud VMs — Oracle Cloud Infrastructure as the primary example, with AWS/Azure equivalents: VCN/VPC and subnet setup, security list / security group rules for MPI traffic, SSH keys, hostfile, `mpirun -np N --hostfile hosts ./hft_mpi_worker --config run.json`, and copying `result.json` back. The GUI's **"Import Cloud Result"** loads those results for comparison with local runs.
- **HPC × AI synergy view:** side-by-side AI training time (Serial vs OpenMP vs MPI data-parallel vs CUDA) and inference latency/throughput vs batch size — the concrete answer to "training and inference of AI models using HPC".

### 12. GUI / UX design specification ("cool, aesthetic, professional")

#### 12.1 Visual identity

A dark "trading terminal meets modern HPC dashboard" look: deep navy surfaces, crisp typography, neon accents used sparingly, generous spacing, rounded corners, subtle borders. It must not look like default grey ImGui.

Theme tokens (implement a `Theme` struct; also provide a Light theme and a high-contrast Presentation theme):

| Token | Hex | Use |
|---|---|---|
| bg.base | #0A0E14 | Window background |
| bg.panel | #111722 | Panels, child windows |
| bg.elevated | #182030 | Cards, popups, hovered rows |
| border | #243044 | 1-px borders, separators |
| text.primary | #E6EDF3 | Main text |
| text.secondary | #8B98A9 | Labels, captions |
| accent.primary | #22D3EE | Focus, selection, primary buttons |
| accent.secondary | #A78BFA | Secondary highlights |
| market.up / buy | #22C55E | Up candles, bids, buys, profit |
| market.down / sell | #EF4444 | Down candles, asks, sells, loss |
| warning | #F59E0B | Warnings, kill switch |

**Fixed per-engine colors**, used consistently in every chart and legend: Serial Naive #94A3B8, Serial Optimized #60A5FA, Threaded Pipeline #38BDF8, OpenMP #34D399, MPI #FBBF24, Hybrid #FB923C, CUDA #C084FC, CUDA+Libs #F472B6. Unit badges: U1 blue, U2 amber, U3 purple, U4 cyan.

**Style:** WindowRounding 10, ChildRounding 8, FrameRounding 6, PopupRounding 8, GrabRounding 6, TabRounding 6, FramePadding (10, 6), ItemSpacing (10, 8), WindowPadding (14, 12), ScrollbarSize 10, anti-aliased lines and fills. ImPlot style matched to the theme (transparent plot background on panels, subtle grid, theme fonts, engine color map).

**Typography:** Inter 15 px for UI (13 px captions, 22/28 px headings); JetBrains Mono for all numbers, prices, tables, and code (tabular figures so digits don't jitter). Font Awesome icons merged into both. Rebuild the font atlas on DPI change (`glfwGetWindowContentScale`) and on a UI-scale setting (80–200%).

#### 12.2 Layout and navigation

- **Top bar:** logo + app name; wizard stepper **① Market → ② Strategy → ③ Engine → ④ Results** (clickable, current step underlined in accent), then ⑤ Cloud & AI, ⑥ Learn, ⚙ Settings; right side: LIVE indicator (pulsing dot while running), simulated clock, wall clock, speed.
- **Left sidebar:** collapsible, icons for the same sections.
- **Docking:** enabled, with a well-designed default layout per screen, a "Reset Layout" button, and layouts persisted to `%APPDATA%/HFT-HPC-Studio/imgui.ini`.
- **Status bar:** FPS, events/s, active engine, CPU usage, GPU utilization/temperature (NVML), memory, current seed, mode (Verification / Latency-Coupled).
- **Home screen:** hero title with an animated mini ticker tape, four syllabus-unit cards (each jumps to the relevant part of the app), System Info card, a large "Start Simulation" button, and preset buttons ("Quick Demo", "Latency Showdown", "GPU Crossover", "AI vs Classic").

#### 12.3 Custom widgets and motion

KPI cards (icon, label, large mono number, delta chip), animated numeric tweening (exponential smoothing), price-cell flash fade, gradient progress bars, toggle switches, segmented controls, pill badges, toasts (slide-in, auto-dismiss) for run completion and errors, tooltips containing formulas, shimmer placeholders while results load, smooth tab transitions (alpha lerp ≈ 150 ms). Motion must never cost frame rate.

#### 12.4 Learn screen and Code Viewer (Screen ⑥ "Learn")

- **Per-unit lessons tied to the app:** short explanations with diagrams drawn via `ImDrawList` (shared vs distributed memory, OpenMP fork-join, MPI domain decomposition, CUDA grid/block/warp hierarchy, GPU memory hierarchy, the tick-to-trade pipeline).
- **Code Viewer:** the actual hot-path source of each engine and kernel, embedded into the exe at build time by `cmake/EmbedSources.cmake`, shown read-only with C++ syntax highlighting, a file dropdown, and an annotations side panel explaining the highlighted pragma/kernel/MPI call. This is what I will show in the viva.
- **Viva Quick Facts:** concise Q&A per unit (e.g., "Why does the GPU lose at batch size 1?", "What does Karp–Flatt tell you?", "Why dynamic scheduling here?", "Why is the MPI version slower on one PC?").

#### 12.5 Presentation Mode

F11 full-screen, Presentation theme with larger fonts, and an optional **auto-demo script** that walks through ① → ④ automatically (Calm market → regime switch to Flash Crash → select AI Selector → run Serial vs OpenMP vs CUDA → reveal the Results tabs), pausable with Space.

#### 12.6 UI performance and robustness

- 60 FPS target with VSync; the render thread never blocks on simulation or I/O.
- Plots use ring buffers (e.g., last 5,000 candles, last 100,000 latency samples) and **LTTB downsampling** for long series.
- All long operations run on worker threads with progress and cancel; all errors become toasts or inline messages — never crashes or silent failures.
- Keyboard shortcuts: Space (play/pause), Ctrl+R (run), Ctrl+B (benchmark), Ctrl+E (export), F1 (help overlay), F11 (full-screen), Ctrl+1…6 (screens).
- Settings persisted as JSON (theme, UI scale, default seed, export folder, default threads/ranks, MPI path override).
- `WIN32` subsystem with `/ENTRY:mainCRTStartup` so `int main()` works; a `--console` flag attaches a console for logs; spdlog file log in `%APPDATA%`.
- `app.rc` provides icon, version info, product name; `app.manifest` declares PerMonitorV2 DPI awareness.

### 13. CLI benchmark runner (`hft_bench.exe`)

CLI11-based, e.g.:
`hft_bench --scenario S2 --engines serial_opt,openmp,cuda --symbols 64,256,1024 --events 200000 --threads 1,2,4,8,16 --batch 1,64,1024,16384 --reps 5 --warmup 2 --mode verify --seed 42 --out results/`

Emits CSV (tidy long format: one row per engine × config × metric) and JSON. `--list` prints engines and availability. `scripts/run_benchmarks.bat` runs the full S1–S8 suite (MPI via `mpiexec`). `tools/plot_results.py` produces publication-quality Matplotlib figures from the CSV for the report.

### 14. Testing and correctness

doctest suite (`hft_tests.exe`), run automatically by `scripts/build.bat`:

- Philox: known-answer test vectors; host and device produce identical sequences (CUDA test compiled when available).
- detmath: accuracy vs `std::` within stated ulp bounds; host/device bit-equality.
- Order books: naive and ladder produce identical fills on randomized streams (property test, 10⁶ events); price-time priority; partial fills; cancels; out-of-range rule.
- Strategies: each indicator vs a straightforward reference implementation; `onEvent` ≡ `onBatch`.
- **Cross-backend equivalence (Verification Mode):** for 3 seeds × 3 workload sizes, every available engine's per-symbol order/fill checksums equal Serial Naive's; analytics within tolerance (relative 1e-5 float, 1e-9 double); AI decision agreement ≥ 99.9%.
- Metrics: Amdahl, Karp–Flatt, Sharpe, drawdown, and percentile functions vs hand-computed values.
- CG solver: converges on random SPD matrices, residual below tolerance, all backends agree.
- MLP: finite-difference gradient check on CPU; CPU vs CUDA forward-pass agreement.
- MPI: worker with 1, 2, and 4 ranks gives identical checksums.
- Instructions for MSVC AddressSanitizer (`/fsanitize=address`) in debug and `compute-sanitizer` for CUDA kernels.

### 15. Build, run, and packaging

- **CMake options:** `HFT_ENABLE_CUDA` (auto), `HFT_ENABLE_MPI` (auto), `HFT_ENABLE_OPENMP` (ON), `HFT_ENABLE_CUDNN` (OFF), `HFT_ENABLE_NVML` (auto), `HFT_ENABLE_AVX2` (ON, runtime-dispatched), `HFT_BUILD_TESTS` (ON), `HFT_BUILD_EXAMPLES` (ON), `HFT_BUILD_GUI` (ON on Windows). Print a configure-time summary of what was found and enabled.
- **`scripts/build.bat`:** finds Visual Studio via `vswhere`, configures `windows-release`, builds, runs tests, and copies runtime DLLs (`libomp140.x86_64.dll`; cuBLAS/cuRAND DLLs if CUDA is enabled) next to the exe.
- **`scripts/run_mpi_demo.bat`:** `"%MSMPI_BIN%\mpiexec.exe" -n 4 hft_mpi_worker.exe --config configs\demo.json`.
- **Packaging:** Inno Setup script (or CPack NSIS) producing `HFT_HPC_Studio_Setup.exe` with the app, DLLs, configs, and docs; states that the MS-MPI runtime (`msmpisetup.exe`) and an NVIDIA driver are prerequisites for MPI/GPU features. cuBLAS DLLs are large, so make them an optional installer component ("GPU acceleration components"). Also produce a portable ZIP.
- **README:** exact prerequisite install steps (Visual Studio 2022 workload, CUDA Toolkit, MS-MPI SDK + runtime, CMake), build and run commands, and troubleshooting ("libomp140 DLL missing", "mpiexec not found", "no CUDA device", "cudaErrorInsufficientDriver", "CMake can't find CUDA from Visual Studio"), plus a screenshots section.

### 16. Documentation deliverables

- `docs/ARCHITECTURE.md` — Mermaid diagrams, data flow, threading model, IPC protocol, JSON schemas, design decisions and trade-offs.
- `docs/REPORT_TEMPLATE.md` — course report skeleton: Abstract, Introduction to HFT, Objectives, Syllabus Mapping table, System Design, Implementation per unit (with code excerpts), Experimental Setup (hardware table from the System Info export), Results (figure placeholders matching exported PNG names), Discussion (including where parallelism did *not* help and why), Conclusion, Future Work (FPGA, kernel-bypass networking, real market data), References.
- `docs/VIVA_NOTES.md` — 40+ likely viva questions with crisp answers, grouped by unit, referencing exact files and functions.
- `docs/BENCHMARK_GUIDE.md` — how to reproduce every chart.
- `docs/CLOUD_RUN.md` — Section 11.
- Inline comments in every engine as required by Rule 2.8.

### 17. Generation plan — produce the project in phases

Before writing any implementation, output: (a) the final file tree, (b) all CMake options, and (c) the complete core contract headers (`types.hpp`, `rng_philox.hpp`, `detmath.hpp`, `istrategy.hpp`, `iengine.hpp`, `run_config.hpp`, `run_result.hpp`). These are the contract for everything that follows.

Then generate phase by phase. For each phase, output every file in full as `### File: <path>` followed by one fenced code block, then the command(s) to build/run and verify the phase, then its acceptance criteria. Never truncate a file. If you approach your output limit, stop at a file boundary and write `CONTINUE FROM: <next file path>` so I can reply "continue". If you can execute commands (agentic tool), configure, build, and run the tests after every phase and fix all errors and warnings before moving on.

| Phase | Contents | Acceptance |
|---|---|---|
| 1 | CMake skeleton + presets, FetchContent deps, core utilities (types, Philox, detmath, timers, rings, pools, histogram, sysinfo, JSON config), both order books, market generator (agents, Hawkes, jump-diffusion, correlation, regimes), E1 + E2 engines, 3 traditional strategies, risk manager, minimal `hft_bench`, first tests | Builds clean; `hft_bench --scenario S1 --engines serial_naive,serial_opt` prints latency percentiles; E1/E2 checksums match |
| 2 | GUI shell: bootstrap, theme, embedded fonts/icons, top bar/stepper/sidebar/status bar, Home screen, Market screen fully live (candles, ladder, depth, heatmap, tape, gauge, stats, agent controls, overview grid) driven by E2 on the simulation thread | Exe launches with no console; market animates smoothly at 60 FPS; regime presets visibly change behavior |
| 3 | Remaining traditional strategies, feature engine, Strategy screen (params, formula cards, quick backtest), E3 threaded pipeline | Every strategy runs; quick backtest < 1 s; E3 checksums match |
| 4 | E4 OpenMP engine + OpenMP kernels (covariance naive/tiled, CG, Cholesky, k-means, MC VaR), sync and false-sharing micro-benchmarks, thread sweeps | Speedup/efficiency/Karp–Flatt computed; schedules compared; checksums match |
| 5 | MPI worker (E5) + hybrid (E6), derived datatypes, both topologies, MPI CG, JSON-lines IPC, GUI `MpiService` (launch, live progress, cancel via Job Object, errors) | `mpiexec -n 4` run matches serial checksums; GUI shows live MPI progress; clear message when MS-MPI is absent |
| 6 | E7 CUDA + E8 CUDA libraries: all kernels, naive vs tiled vs cuBLAS covariance, cuRAND VaR, Thrust/CUB, streams, pinned memory, CUDA Graphs, batch semantics, occupancy, delay-load + probing | GPU checksums match in Verification Mode; crossover sweep produces data; app runs on a non-NVIDIA machine with GPU cards disabled |
| 7 | AI: MLP (serial/OpenMP/MPI/CUDA train + infer), Q-learning market maker (parallel episodes), regime k-means + bandit Selector, AI panels on the Strategy screen, optional cuDNN LSTM + PyTorch script | Model trains in-app on each backend; Selector switches strategies with explanations; training-time comparison populated |
| 8 | Engine screen, `BenchmarkService`, full Results dashboard (all tabs, KPI cards, auto-insights, leaderboard + radar, verification), exports + Report Pack, Latency-Coupled mode | Full S1–S8 sweep runs from the GUI; every chart populated from real data; exports open correctly |
| 9 | Cloud & AI Lab, Learn screen with diagrams, Code Viewer with embedded sources + annotations, Viva Quick Facts, Presentation Mode + auto-demo, Settings | Code Viewer shows real engine source; auto-demo runs end to end |
| 10 | Packaging (installer + portable ZIP), scripts, `examples/demo_*` standalone teaching programs (single-file, heavily commented, same toy workload, each prints a results table), all docs, final polish pass | Fresh clone → `scripts\build.bat` → working installer; all tests pass; docs complete |

### 18. Final acceptance checklist

- [ ] App launches on Windows 11 with no console window, looks polished in the dark theme, and scales on high-DPI displays.
- [ ] Market prices emerge from agents buying and selling through a real order book; regimes visibly change market behavior.
- [ ] All 10 traditional strategies, both AI strategies (MLP, Q-learning), and the AI Selector run on every available engine.
- [ ] All 8 engines implemented; unavailable ones disabled with reasons; no crashes.
- [ ] Verification Mode: checksums match across engines (AI: agreement ≥ 99.9%).
- [ ] Latency-Coupled Mode shows measurable latency-cost differences between engines.
- [ ] Every metric in Section 10 is computed from real measurements and visualized.
- [ ] Every syllabus topic in Section 1 is demonstrably present and explained on the Learn screen.
- [ ] GPU crossover and MPI scaling charts generated on my hardware.
- [ ] Exports and Report Pack work; docs complete; installer built.
- [ ] Code is complete, commented for the viva, and warning-free at /W4.

### 19. Realism and honesty rules (reflect these in UI text and docs)

- Real HFT tick-to-trade latency is dominated by network I/O and is achieved with tuned single-core CPU code, kernel-bypass networking, and FPGAs. GPUs are used for throughput work (backtesting, risk, model training, batched inference), not per-tick decisions, because PCIe transfers and kernel launches cost microseconds. The app must demonstrate this with data (S1 and the crossover chart) rather than implying CUDA is fastest at everything.
- MPI on one PC demonstrates the distributed-memory programming model; real distributed speedups need multiple nodes (see the cloud run).
- Windows timer resolution and OS scheduling noise affect sub-microsecond measurements; report clock resolution and measured timing overhead.
- The market is synthetic; profitability in the simulator says nothing about real markets. Show a small persistent label: "Educational simulation — not financial advice."

**Now begin:** output the file tree, the CMake option list, and the core contract headers, then proceed with Phase 1.

## END PROMPT
