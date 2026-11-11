# HFT-LAB user guide

HFT-LAB runs offline research simulations. Market data, models, native engine
results, logs and experiment manifests remain local. Start the terminal
workspace from the project directory:

![Authentic HFT-LAB terminal capture](../artifacts/tui.svg)

```powershell
.venv\Scripts\python.exe -m hftlab
```

On Linux or WSL use `.venv/bin/python -m hftlab`. After installing the package,
`hftlab` starts the same interface. A Textual-capable terminal and the optional
TUI dependencies are required; the headless commands remain available without
the UI dependencies.

## Navigation

The sidebar follows the workflow from data to results. Use Tab and Shift+Tab
to focus controls, Enter or Space to activate them, and arrow keys in selects,
tables and selection lists. Mouse interaction is supported by Textual.

| Shortcut | Action |
| --- | --- |
| F1 | Help and settings |
| F2 | Market data |
| F5 | Validate configuration and start simulation |
| F6 | Final results |
| Escape | Home |
| Ctrl+P | Textual command palette |
| Ctrl+C | Request cancellation of the active simulation or training job |
| Ctrl+Q | Quit; asks before cancelling a running job |

The layout reduces sidebar width and stacks form controls on narrow terminals.
The Help page also offers a compact layout. Save preferences to retain dataset,
strategy, engine, CPU mode, output directory and compact layout in
`.hftlab-preferences.json`.

## Home

The home page shows detected engine statuses, recent experiments, recent model
artifacts and shortcuts into the workflow. Diagnostic discovery runs in a
background worker. Opening a recent experiment reloads its persisted result.
Only an actual native runner capability probe can enable a native engine.

## Market data

Choose an output path, event count, instrument count, random seed and scenario,
then select **Generate dataset**. Generation uses a background worker. The log
shows the returned manifest and validation statistics after the operation.

Available scenarios are normal, high volatility, high liquidity, low liquidity,
shock, correlated, stress and predictable. The predictable scenario is a
synthetic teaching aid for model learning; it supplies no evidence about real
market profitability.

**Validate / inspect** checks the selected dataset and displays statistics.
**Browse files** refreshes the dataset table. Select a row to use that dataset
for both simulation and AI training. To import a file, enter its source path
and choose a different destination in the selected dataset field. Supported
formats depend on the installed data dependencies; CSV is always supported.

**Delete selected** requires confirmation and only permits deletion within
the project `data` directory. It removes the selected dataset file; inspect
associated manifests separately if also removing them.

## Trading strategy

Choose market making, momentum, statistical arbitrage, mean reversion or AI
trading. Each selection displays a brief description. AI trading enables the
native model path field. Train or select an exported model before running it.
Statistical arbitrage needs a suitable multi-instrument dataset.

## Computation engine

Exactly five engine categories are presented:

1. **CPU Computing**: single-core single-threaded, single-core multithreaded,
   multi-core single-threaded, or multi-core multithreaded. A single worker
   allowed to migrate across cores is still one worker.
2. **OpenMP Shared-Memory Computing**: uses the available native OpenMP build
   without an additional engine-selection submenu.
3. **MPI Distributed-Memory Computing**: uses a compatible MPI runner and
   launcher without an additional engine-selection submenu.
4. **GPU-Accelerated CUDA Computing**: requires a CUDA-enabled native build and
   a compatible actual GPU.
5. **Hybrid Computing**: select two or more components. The capability planner
   rejects unsupported compositions.

The UI chooses a conservative automatic worker count. The headless CLI exposes
advanced worker settings for benchmark studies. Selecting unavailable hardware
causes a clear error and never silently substitutes another engine.

## Simulation configuration

The summary uses the dataset, strategy and engine selected on their pages.
Set capital, transaction fee basis points, slippage ticks, simulated latency
nanoseconds, position limit, maximum order quantity, maximum loss, timeout and
output directory. Fields are validated before starting native computation.

**Run simulation** or F5 starts the native subprocess through the job manager.
The interface remains available while the job runs. **Benchmark available CPU
engines** performs repeated sequential, threaded and available OpenMP runs and
saves `reports/benchmark.json`. Measurements are collected without live UI
updates inside the measured native work.

## Live monitor

Progress uses native processed-event counters rather than a timer animation.
The monitor shows job lifecycle states, event counts, trades, orders, elapsed
time, simulated market timestamp when reported, P&L, equity and throughput.
Resident memory is shown when process sampling supplies it. CPU/GPU utilization
is only shown when actual telemetry is supplied; missing telemetry is not
estimated. Logs contain real backend messages.

**Cancel job** asks for confirmation before terminating an active native job or
benchmark. The job manager cleans up its subprocess tree. AI training can be
cancelled at a completed epoch boundary. Dataset operations finish their current
operation; they do not expose an immediate stop control in this interface.

## Results center

The results tabs expose HPC overview, engine timing, scaling, CPU/OS profiling,
OpenMP, MPI, CUDA, hybrid, trading, AI and correctness information. The available
measurements depend on the run. Empty hardware-specific tabs explicitly say
that no measurements were collected. No scalability is inferred from a single
run. Benchmark results add an engine comparison table with actual median
runtime, throughput, speedup, efficiency and status.

The Export tab writes the current result to an HTML, JSON or CSV path. Reports
use the measurements in the current result. Reopen another experiment to export
its report instead.

## AI training center

Choose a dataset, logistic regression or MLP, epochs, random seed and exported
model path. **Train & export native model** starts a training worker and shows
actual epoch callback records. Training returns validation and held-out test
metrics, preprocessing metadata and a safe native numerical artifact. A `.json`
sidecar records model provenance and compatibility metadata.

**Browse models** lists recent artifacts under `models`. Enter a model path in
the output field to select an existing artifact. **Use selected model for AI
trading** copies that path to Strategy, selects the AI strategy and opens engine
selection. Classification metrics describe direction prediction, while the
simulation applies execution costs and risk controls separately.

## Experiment history

The SQLite registry survives application restarts. Filter by experiment ID,
engine or strategy, refresh the list, sort by runtime and open a selected row.
The table displays persisted status and available runtime/P&L values. Enter on
a selected row reopens its results. Headless commands provide additional
comparison and experiment operations.

## Diagnostics

Refresh diagnostics to inspect host details, Python environment, optional
packages, native capabilities, compiler/build tools, MPI, CUDA and profiling
tools. Statuses distinguish AVAILABLE, UNAVAILABLE, INCOMPATIBLE and NOT TESTED.
An NVIDIA driver utility alone does not establish a compiled CUDA backend.
See `docs/TROUBLESHOOTING.md` and `docs/INSTALLATION.md` for build setup.

## Headless use

Each command offers validated arguments and `--help`:

```text
python -m hftlab doctor
python -m hftlab generate --help
python -m hftlab simulate --help
python -m hftlab train --help
python -m hftlab benchmark --help
python -m hftlab compare --help
python -m hftlab report --help
python -m hftlab models --help
python -m hftlab experiments --help
```

Use `docs/EVALUATION_DEMO.md` for the tested demonstration and
`docs/IMPLEMENTATION_STATUS.md` for the exact implemented and verified scope.
