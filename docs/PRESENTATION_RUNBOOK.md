# Presentation runbook (Windows)

Fresh verification: **13 November 2026**. Run commands from `D:\HPC` in PowerShell.
The active application is the root **Parallel and GPU-Accelerated High-Frequency Trading Simulation Engine** project. Use its existing Python
3.11 environment. The machine's default Python and the projects in `temp/` are
separate from this application.

## 1. What each folder does

| Location | Purpose |
|---|---|
| `hftlab/cli.py`, `hftlab/__main__.py` | Command line and application entry points |
| `hftlab/tui/` | Textual terminal interface, display labels, styling |
| `hftlab/datasets/` | Seeded generation, CSV/binary import and validation |
| `hftlab/ai/`, `ai_training/` | Training, numerical model exports and training notes |
| `hftlab/engines/`, `hftlab/orchestration/` | Native discovery, subprocesses, cancellation and validation |
| `hftlab/benchmarks.py`, `experiments.py`, `reports.py`, `monitoring.py` | Comparisons/scaling, SQLite history, exports, diagnostics |
| `native/core/`, `native/ai/`, `native/include/` | C17 matching, strategies, risk, accounting, inference and public types |
| `native/engines/cuda/` | Optional CUDA feature/inference kernels; CPU builds use an explicit unavailable stub |
| `native/runners/`, `native/tests/` | Native executable and C regression tests |
| `tests/` | Python unit, integration, AI/native correctness and terminal UI tests |
| `configs/`, `schemas/` | Example configurations and versioned data/protocol contracts |
| `scripts/` | Installation, CMake build, demo and cleanup utilities |
| `docs/`, `README.md` | User/developer guides and verification evidence |
| `.github/workflows/` | CI definitions; local success does not prove CI ran |
| `build/`, `build/cpu/` | Default CPU+OpenMP runner and separate CPU-only runner |
| `build-full/` | IDE/preset build tree; the verified full executable is under `build/full/` |
| `.venv/`, `dist/` | Installed dependencies/launchers and generated Python packages |
| `data/`, `models/` | Datasets and exported native models |
| `results/`, `experiments/` | Run artifacts and persistent experiment registry |
| `reports/`, `artifacts/` | Reports, end-to-end demonstrations and test evidence |
| `temp/HFT-HPC-C/` | Separate older C desktop/benchmark reference project |
| `temp/Mini-HFT-HPC-MidSem/` | Separate smaller POSIX/WSL teaching reference project |

Python handles data, training and presentation. Simulation, strategy decisions,
matching and accounting execute in native C/CUDA. Related architecture details
are in [ENGINE_IMPLEMENTATION.md](ENGINE_IMPLEMENTATION.md).

## 2. Start and check this machine

```powershell
Set-Location D:\HPC
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\hftlab.exe --version
.\.venv\Scripts\hftlab.exe doctor
.\.venv\Scripts\hftlab.exe
```

The last command opens the terminal UI. `Ctrl+Q` quits. An equivalent launcher is
`.\.venv\Scripts\python.exe -m hftlab`. Use a normal terminal with approximately
130 columns and 45 rows for the presentation.

Current verified engines include sequential, native threads, OpenMP,
CPU+OpenMP hybrid, local Microsoft MPI and RTX 5050 CUDA execution. CUDA 13.5
and Microsoft MPI runtime/SDK are now installed. Select the full runner in the
current shell before launching diagnostics, tests or the presentation:

```powershell
$env:HFTLAB_RUNNER = 'D:\HPC\build\full\hftlab_runner.exe'
.\.venv\Scripts\hftlab.exe doctor
```

Without virtual-environment activation, `doctor` may say CMake is not on PATH.
`scripts/build.py` also checks beside the selected Python executable, so the
tested build command below still works. To expose the environment tools to
diagnostics for the current shell:

```powershell
$env:Path = "D:\HPC\.venv\Scripts;$env:Path"
```

## 3. Run the complete local automated checks

Run each command and inspect its result before continuing:

```powershell
.\.venv\Scripts\python.exe scripts/build.py --test
.\.venv\Scripts\python.exe scripts/build.py --no-openmp --build-dir build/cpu --test
.\.venv\Scripts\python.exe -m pytest -q -ra
.\.venv\Scripts\python.exe -m ruff check hftlab tests scripts
.\.venv\Scripts\python.exe -m pip check
```

The earlier CPU verification passed 101 pytest tests with 2 hardware skips.
After installing CUDA/MPI and selecting `build/full`, the suite passed **102
tests with 1 skip**. That skip covers unavailable-engine rejection because all
optional engines are compiled. Both CUDA and MPI replay tests executed.
CTest passed 2/2 cases in the full, fresh CUDA-default and CPU+OpenMP builds.
Ruff passed. One sklearn/SciPy optimizer deprecation warning is nonfatal.

CTest covers the order book, accounting, features, inference, native replay and
capability protocol. Pytest covers data/scenarios/corruption, purged training,
Python/C inference parity, classical and AI replay against sequential, risk and
result contracts, registry/reports, timeouts/cancellation and the terminal UI.
All four CPU modes and the CPU+OpenMP hybrid are exercised.

The Windows sandbox needs loopback socket support for Textual/asyncio tests.
This verification ran the full pytest suite with that access. If a restricted
agent session stalls during socket initialization, run the suite from normal
PowerShell with loopback access. Native compiler probes can also need Visual
Studio PDB IPC access. See [TESTING.md](TESTING.md).

For a subset while changing text:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/tui -q
.\.venv\Scripts\python.exe -m pytest tests/integration tests/correctness -q -ra
```

Packaging is a separate check. The local temporary folder avoids the previously
observed setuptools temporary-path issue. These environment settings apply only
to the current PowerShell session:

```powershell
New-Item -ItemType Directory -Force -Path .tmp | Out-Null
$env:TEMP = 'D:\HPC\.tmp'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m build --no-isolation
```

The wheel contains Python/UI code; the native executable is a separate build.
Both package formats and an isolated wheel import/CSS/native-discovery smoke
check passed during this verification.

## 4. Prepare an offline presentation pack

```powershell
.\.venv\Scripts\hftlab.exe demo --events 10000 --epochs 15 --output artifacts/rehearsal/demo
```

This command generates data, benchmarks engines, trains logistic and MLP models,
exports a strictly held-out dataset, replays AI and classical baselines and
creates offline reports. With the installed extras, automatic training selected
sklearn logistic and PyTorch CPU MLP. A dependency-free Python training backend
is also supported and was checked separately.

Open `D:\HPC\artifacts\rehearsal\demo\demo.html` and `benchmark.html` in a
browser. Important companion files are `demo.json`, `market.csv`, `heldout.csv`,
`mlp/model.hftmodel`, `mlp/metadata.json` and `logistic/model.hftmodel`.

The already-tested presentation pack is in
[`artifacts/presentation-2026-11-13/demo/`](../artifacts/presentation-2026-11-13/demo/).
Commands here use `artifacts/rehearsal/` to retain that dated evidence.

## 5. Suggested live demonstration

1. Launch the UI and open **Diagnostics**. Show actual backend availability.
2. Open **Market Data** (F2). Set the selected dataset to
   `artifacts/rehearsal/demo/market.csv` and choose **Validate / inspect**.
3. Choose **Momentum trading**, then **CPU Computing / Single-core,
   single-threaded**. Leave the cost/risk defaults consistent between runs.
4. Press F5. On **Results** (F6), show event count, checksum, timing, fees,
   equity and P&L. A 10,000-event job can finish too quickly for a lengthy live
   monitor demonstration.
5. Repeat with **Multi-core, multithreaded**, **OpenMP**, and **Hybrid** with
   CPU+OpenMP selected. Use the benchmark report for repeated timing comparisons.
6. Open **AI Training** or select the exported MLP model in **Strategy**. For
   a held-out trading demonstration, use `artifacts/rehearsal/demo/heldout.csv`
   and `artifacts/rehearsal/demo/mlp/model.hftmodel`.
7. Show **Experiments** to reopen a result, and export it as HTML/CSV/JSON.
   Finish with the saved offline demo report.

Use Tab/Shift+Tab, arrow keys, Enter, scrolling and mouse navigation. F1 opens
help, Escape goes home, Ctrl+C requests cancellation. A saved report provides
a presentation fallback if a terminal/display setup differs.

## 6. Individual commands for data, training and simulation

```powershell
.\.venv\Scripts\hftlab.exe generate --events 10000 --instruments 4 --scenario predictable --seed 42 --output artifacts/rehearsal/market.csv
.\.venv\Scripts\hftlab.exe data inspect artifacts/rehearsal/market.csv
.\.venv\Scripts\hftlab.exe data validate artifacts/rehearsal/market.csv
.\.venv\Scripts\hftlab.exe data import artifacts/rehearsal/market.csv --output artifacts/rehearsal/market.hftdata
.\.venv\Scripts\hftlab.exe train --dataset artifacts/rehearsal/market.csv --kind logistic --backend python --epochs 15 --output artifacts/rehearsal/python-logistic
.\.venv\Scripts\hftlab.exe train --dataset artifacts/rehearsal/market.csv --kind mlp --backend torch --device cpu --epochs 15 --output artifacts/rehearsal/mlp
.\.venv\Scripts\hftlab.exe models --validate artifacts/rehearsal/mlp/model.hftmodel
.\.venv\Scripts\hftlab.exe simulate --config configs/default.yaml --dataset artifacts/rehearsal/market.csv --progress
.\.venv\Scripts\hftlab.exe simulate --dataset artifacts/rehearsal/market.hftdata --engine threaded --cpu-mode multi-multi --threads 4
.\.venv\Scripts\hftlab.exe simulate --dataset artifacts/rehearsal/market.csv --engine openmp --threads 4
.\.venv\Scripts\hftlab.exe simulate --dataset artifacts/rehearsal/market.csv --engine hybrid --components cpu,openmp --threads 4
.\.venv\Scripts\hftlab.exe simulate --dataset artifacts/rehearsal/demo/heldout.csv --strategy ai --model artifacts/rehearsal/demo/mlp/model.hftmodel
.\.venv\Scripts\hftlab.exe experiments list
```

Supported CLI strategies: `market-making`, `momentum`, `mean-reversion`,
`stat-arb`, `ai`, `hold`. Statistical arbitrage needs at least two instruments.
The four CPU modes are `single-single`, `single-multi`, `multi-single` and
`multi-multi`. A single migratable worker remains one worker.

`experiments show ID`, `experiments repeat ID`, and
`experiments export ID --output artifacts/rehearsal/experiment.html` operate on
an ID from the list. `report --experiment ID --output ...` supports `.html`,
`.json`, `.csv`, and `.md`. `benchmark` is an alias for the comparison workflow.
Each subcommand has `--help`.

## 7. Repeated benchmarks and scaling

```powershell
.\.venv\Scripts\hftlab.exe generate --events 100000 --instruments 16 --scenario correlated --output artifacts/rehearsal/benchmark.csv
.\.venv\Scripts\hftlab.exe compare --dataset artifacts/rehearsal/benchmark.csv --all --threads 4 --repeats 3 --output-dir artifacts/rehearsal/runs --output artifacts/rehearsal/engines.json
.\.venv\Scripts\hftlab.exe scaling --dataset artifacts/rehearsal/benchmark.csv --engine threaded --workers 1,2,4 --repeats 3 --output-dir artifacts/rehearsal/runs --output artifacts/rehearsal/strong.json
.\.venv\Scripts\hftlab.exe scaling --dataset artifacts/rehearsal/benchmark.csv --engine threaded --workers 1,2,4 --weak --events-per-worker 10000 --repeats 3 --output-dir artifacts/rehearsal/runs --output artifacts/rehearsal/weak.json
.\.venv\Scripts\hftlab.exe report artifacts/rehearsal/strong.json --output artifacts/rehearsal/strong.html
.\.venv\Scripts\hftlab.exe report artifacts/rehearsal/weak.json --output artifacts/rehearsal/weak.html
```

`compare` writes HTML alongside JSON. Compare every candidate against the same
sequential oracle. Use measured medians and retain slower/noisy results. Native
compute time excludes dataset loading and process orchestration; it differs
from the time spent waiting for a CLI command. Timings vary between rehearsals.

Explain strong scaling as fixed work with more workers. Weak scaling grows work
with workers; each workload gets its own sequential reference. Keep at least as
many independent instrument groups as workers. The native simulator holds its
input/work arrays in RAM; the tested scale is 100,000 events.

## 8. Optional backends and reference projects

Build the installed toolchains in a separate directory and select the runner
explicitly in the current shell:

```powershell
.\.venv\Scripts\python.exe scripts/build.py --cuda --cuda-arch 120 --mpi --build-dir build/full --test
$env:HFTLAB_RUNNER = 'D:\HPC\build\full\hftlab_runner.exe'
.\.venv\Scripts\hftlab.exe doctor
```

An MPI-only build uses `--mpi --build-dir build/mpi`. A CUDA-only build uses
`--cuda --cuda-arch 120 --build-dir build/cuda` on this GPU. The architecture
argument overrides the CMake cache. Confirm capabilities and matching runtimes, rerun
`pytest tests/correctness -q -ra`, then execute correctness-gated comparisons.
Return to the normal runner with
`$env:HFTLAB_RUNNER = 'D:\HPC\build\hftlab_runner.exe'`.
Hardware hybrids beyond CPU+OpenMP, Parquet and Linux/sanitizer gates remain
unverified here.

The root suite does not build `temp/`. Its reference source files were preserved.
The bundled `temp/HFT-HPC-C/windows/hft_tests.exe` was executed freshly from an
isolated artifact working directory: **1,502 checks, 0 failures**. That checks
the supplied executable, not a fresh source build or the desktop UI.
Use that project's [README](../temp/HFT-HPC-C/README.md) for its separate GUI,
benchmark and CMake workflow. Its default source rebuild was not tested.

The mini reference needs a POSIX environment with GCC, make and pthreads. In
Linux/WSL, its documented commands are `make all`, `make data`, then
`./bin/sequential data/market.csv`, `./bin/pthreads data/market.csv 2` and
`./bin/openmp data/market.csv 2`. MPI additionally uses `make mpi` and
`mpirun -np 2 ./bin/mpi data/market.csv`. These commands were inspected, not
executed on this Windows host. Historical reference logs are not test evidence.

## 9. Display name and presentation text

The display name is **Parallel and GPU-Accelerated High-Frequency Trading
Simulation Engine**. Python UI, CLI and report text share `APP_NAME` in
`hftlab/__init__.py`. Native help/error text uses `HFT_APP_NAME` in
`native/runners/main.c`. The launcher remains `hftlab`.

| Text | Edit location |
|---|---|
| Window title, subtitle, sidebar brand, home text, navigation, strategy/engine display names | `hftlab/tui/app.py` (`TITLE`, `SUB_TITLE`, `PAGES`, `STRATEGIES`, `ENGINES`, `CPU_MODES`, page composition methods) |
| UI colors, spacing and layout | `hftlab/tui/styles.tcss` |
| CLI description, version label and errors | `hftlab/cli.py` |
| Exported HTML/Markdown titles and explanatory text | `hftlab/reports.py` |
| Native help/error/capability text | `native/runners/main.c`, `native/core/results.c` |
| Package description and author credit | `pyproject.toml` |
| Project introduction and documentation | `README.md`, `docs/` |

Engine IDs,
strategy IDs, package imports, widget IDs, protocol/schema versions, feature
names and BUY/HOLD/SELL model mappings are functional identifiers; renaming
them requires corresponding code/tests/data migrations. UI label changes can
also require test expectation updates. Rebuild the native runner for C changes,
restart the UI for Python text changes, and regenerate reports to update their
saved text. The presentation reports and authentic terminal capture were
regenerated after the display-name change; historical logs retain their
original contents.

Fresh logs, JUnit, CLI checks and a machine-readable summary are in
[`artifacts/validation/presentation-2026-11-13/`](../artifacts/validation/presentation-2026-11-13/).
The numerical results and remaining gates are recorded in
[TEST_RESULTS.md](TEST_RESULTS.md) and [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).
