# Executed test and demonstration evidence

## CUDA/MPI build and execution verification: 13 November 2026

CUDA 13.5.37 and Microsoft MPI 10.1.3 runtime/SDK are now installed. The full
runner was built with MSVC 19.51 for the RTX 5050, driver 621.01, CUDA
architecture `120` (`compute_120`/`sm_120`). CMake previously created
`hftlab_core` before enabling CUDA, so its initial `CUDA_ARCHITECTURES` property
was empty even though compiler detection succeeded. CUDA language/default
initialization now precedes target creation. `scripts/build.py --cuda-arch`
also supports explicit selection and cache overrides.

```powershell
.\.venv\Scripts\python.exe scripts/build.py --cuda --cuda-arch 120 --mpi --build-dir build/full --test
$env:HFTLAB_RUNNER = 'D:\HPC\build\full\hftlab_runner.exe'
.\.venv\Scripts\hftlab.exe doctor
.\.venv\Scripts\python.exe -m pytest -q -ra
```

| Executed check | Actual outcome |
|---|---|
| Full CUDA/MPI build for architecture 120 | CUDA source compiled, executables linked, 2/2 CTest cases passed |
| Fresh `--cuda --build-dir build/cuda-default --test`, with no architecture override | Compiler default 75 initialized correctly on the first configuration; build and 2/2 CTest cases passed |
| Default CPU+OpenMP rebuild after the CMake change | 2/2 CTest cases passed |
| Full Python suite with `HFTLAB_RUNNER=build/full/hftlab_runner.exe` | 102 passed, 1 skipped, 1 existing sklearn/SciPy deprecation warning; 49.33 seconds |
| CUDA and MPI AI replay gates | Both executed and matched the sequential oracle; MPI used 2 processes |
| Windows MPI diagnostics | `mpiexec -help` succeeds and reports AVAILABLE; the former `--version` probe was unsupported by Microsoft MPI |
| Ruff | All checks passed |
| Six-engine comparison, 100,000 correlated events / 16 instruments | Every engine completed; every measured repeat passed the sequential correctness gate |

The single pytest skip is `test_unavailable_engine_rejected`, because both
optional engines are compiled and usable. The compiler/build tests and full
pytest run used the required Windows driver/process/loopback access outside the
sandbox. MPI verification is local multi-process execution. Hardware hybrids
beyond CPU+OpenMP, the GPU logistic inference case, remote-node MPI, Parquet,
Linux/sanitizers and fault injection remain unverified.

The comparison used 4 requested workers, 1 warmup and 3 measured repeats per
engine. Actual median native timings and speedups are:

| Engine | Median native seconds | Speedup |
|---|---:|---:|
| Sequential | 0.0169058 | 1.000 |
| Native threads | 0.0049599 | 3.408 |
| OpenMP | 0.0046574 | 3.630 |
| MPI, 4 local ranks | 0.0057198 | 2.956 |
| CUDA, RTX 5050 | 0.0884469 | 0.191 |
| CPU+OpenMP hybrid | 0.0173509 | 0.974 |

CUDA kernel times were 0.0105453 / 0.0106305 / 0.0105347 seconds for the measured
runs. The full native CUDA region includes allocation, transfer and matching
overhead and was slower on this workload. Every MPI repeat recorded 4 ranks
and positive communication time. These short timings remain sensitive to host
noise; larger-workload GPU speedups have not been established.

One capability probe returned an empty response during the initial comparison,
causing its hybrid stage to fail. That partial report is preserved as
`comparison-initial.json`. Eight subsequent native capability probes succeeded,
the focused sequential/hybrid comparison passed, and one complete six-engine
retry passed. The transient response's underlying cause remains undetermined.
The table above uses only the successful complete retry.

Fresh [JUnit](../artifacts/validation/cuda-mpi-2026-11-13/pytest.xml),
[diagnostics](../artifacts/validation/cuda-mpi-2026-11-13/doctor.json),
[validated summary](../artifacts/validation/cuda-mpi-2026-11-13/summary.json),
build/test/probe logs and the initial partial report are under
[`artifacts/validation/cuda-mpi-2026-11-13/`](../artifacts/validation/cuda-mpi-2026-11-13/).
The presentation comparison is available as [HTML](../reports/full-check.html)
and [JSON](../reports/full-check.json).

## Display-name update verification: 13 November 2026

The application now displays **Parallel and GPU-Accelerated High-Frequency
Trading Simulation Engine**. Python UI/CLI/report branding shares `APP_NAME`;
native help/error branding uses `HFT_APP_NAME`. The sidebar and home heading
grow with wrapped text. Display text, descriptive schema/package metadata,
project credits and documentation were updated. The `hftlab` launcher,
internal package/build identifiers, schema versions, model formats and native
simulation behavior remain unchanged.

| Re-executed check | Actual outcome |
|---|---|
| Default CPU+OpenMP and separate CPU-only builds | Renamed native runners rebuilt; 2/2 CTest cases pass in each tree |
| Full `pytest -q -ra` | 101 passed, 2 MPI/CUDA skips, 1 existing sklearn/SciPy deprecation warning; 18.91 seconds |
| Ruff and `pip check` | All checks passed; no broken requirements |
| Actual Textual rendering at 140x48, 130x45 and 60x30 | Full home/sidebar title retained without truncation in rendered widget lines; automatic wrapping verified |
| CLI version/help/errors and native help/errors | Requested display name present; old branding absent |
| Existing saved HTML/Markdown reports | 14 reports regenerated from retained JSON measurements with the new branding |

The native compiler stalled under the documented Windows sandbox PDB IPC
restriction and was interrupted. Both rebuilds then succeeded with that access.
The complete pytest run and authentic Textual captures used loopback-capable
access. No new timing samples were inferred from the refreshed reports; their
underlying measurements are unchanged. Reference projects and original logs
were preserved.

The [documentation terminal capture](images/tui.svg) was regenerated from the
actual renamed application after diagnostics completed. Evidence is under
[`artifacts/validation/rename-2026-11-13/`](../artifacts/validation/rename-2026-11-13/):
[JUnit](../artifacts/validation/rename-2026-11-13/pytest.xml),
[test log](../artifacts/validation/rename-2026-11-13/pytest.log),
and [rendered title/report checks](../artifacts/validation/rename-2026-11-13/branding.json).
MPI/CUDA, Linux/sanitizer execution and manual desktop inspection remain outside
this verification. The earlier presentation benchmarks below retain their
original execution scope.

## Earlier CPU verification: 13 November 2026

Freshly executed in `D:\HPC`, using Python 3.11.9, MSVC 19.51, Windows
10.0.26300 and the default native CPU+OpenMP runner. There are 10 physical / 16
logical processors and approximately 15.7 GiB RAM. The driver sees an RTX 5050;
the native CUDA toolkit/compiler and MPI implementation/launcher remain absent.
The existing `build-full` cache records `CMAKE_CUDA_COMPILER-NOTFOUND` and is not
a verified full build. No optional backend was substituted.

| Executed check | Actual outcome |
|---|---|
| `.venv\Scripts\python.exe scripts/build.py --test` | Default CPU+OpenMP configuration builds; 2/2 CTest cases pass |
| `.venv\Scripts\python.exe scripts/build.py --no-openmp --build-dir build/cpu --test` | Separate CPU-only configuration builds; 2/2 CTest cases pass |
| `.venv\Scripts\python.exe -m pytest -q -ra --junitxml=artifacts/validation/presentation-2026-11-13/pytest.xml` | 101 passed, 2 skipped, 0 failures/errors; 19.45 seconds |
| `.venv\Scripts\python.exe -m ruff check hftlab tests scripts` | All checks passed |
| `.venv\Scripts\python.exe -m pip check` | No broken requirements |
| `.venv\Scripts\python.exe -m build --no-isolation` | Wheel and source distribution built, with TEMP/TMP under `.tmp` |
| Wheel installed with `--no-index --no-deps --target artifacts/presentation-2026-11-13/wheel-smoke` | Isolated import originates in target directory; TUI CSS exists; explicit native capability probe succeeds |
| `hftlab demo --events 10000 --epochs 15 --output artifacts/presentation-2026-11-13/demo` | Generation, actual sklearn/PyTorch CPU training, held-out replay, engine comparison, JSON/HTML export complete |
| Additional CLI smoke driver, 23 invocations | Data inspect/validate/list, binary import/replay, YAML+CLI overrides/progress, models, Python training, AI replay, history show/repeat/export, four report formats, benchmark alias, diagnostics and explicit MPI/CUDA rejection pass |
| Bundled `temp/HFT-HPC-C/windows/hft_tests.exe`, isolated working directory | 1,502 checks, 0 failures; supplied executable only, separately from Parallel and GPU-Accelerated High-Frequency Trading Simulation Engine |

The 2 pytest skips explicitly identify unavailable MPI and CUDA replay gates.
One nonfatal sklearn/SciPy L-BFGS optimizer deprecation warning remains. The
complete pytest run used loopback-capable access outside the Windows sandbox
restriction, as required for asyncio/Textual initialization. Native builds were
incremental configurations of the existing build trees. UI verification in this
run is automated/headless, including real native worker integration; the
interactive desktop/terminal was not manually inspected again.

Fresh evidence: [JUnit](../artifacts/validation/presentation-2026-11-13/pytest.xml),
[pytest log](../artifacts/validation/presentation-2026-11-13/pytest.log),
[diagnostics](../artifacts/validation/presentation-2026-11-13/doctor.json),
[CLI invocations](../artifacts/validation/presentation-2026-11-13/cli-smoke.json),
and [validated numerical summary](../artifacts/validation/presentation-2026-11-13/summary.json).
Build, packaging, wheel, lint, dependencies and reference-executable logs are in
that same dated evidence directory. No old reference logs were used as current
evidence. Generated artifacts remain ignored by Git.

### Fresh demo and performance measurements

The demo trained logistic and MLP models from 10,000 seeded events. MLP held-out
classification: 1,482 samples, accuracy 0.75843455, macro F1 0.63492679,
cross-entropy 0.63637304. Its standalone trading dataset contains 1,374 events,
with 124 earlier-order cancellations omitted, zero training/embargo events,
empty initial book and cold-start features. Held-out simulated net P&L:
MLP 263.366861; logistic 243.808088; momentum 217.251203; mean reversion
-289.531206; HOLD 0. These are synthetic educational outputs, not real-market
profitability evidence.

The independent 100,000-event / 16-instrument correlated-data comparison used
4 requested workers, 1 warmup and 3 measured repeats per available engine. Every
reported repeat passed the sequential correctness gate:

| Engine | Median native seconds | Speedup |
|---|---:|---:|
| Sequential | 0.0194122 | 1.000 |
| Native threads | 0.0058335 | 3.328 |
| OpenMP | 0.0047506 | 4.086 |
| CPU+OpenMP hybrid | 0.0179986 | 1.079 |
| MPI | UNAVAILABLE | N/A |
| CUDA | UNAVAILABLE | N/A |

These are stored observations, not thresholds or promised future timings. Native
compute time excludes input loading and process orchestration. Three short
samples remain sensitive to scheduling/frequency noise; do not infer a precise
parallel efficiency or guaranteed improvement from the point estimates.

Strong scaling held the 100,000-event input fixed at 1/2/4 workers, with 3 repeats
and 1 warmup each. Threaded medians were 0.0158800 / 0.0084125 / 0.0045394 seconds;
speedup relative to its one-worker run was 1.000 / 1.888 / 3.498. Weak scaling
used 10,000 events per worker; medians were 0.0019378 / 0.0021355 / 0.0022077
seconds and weak efficiencies 1.000 / 0.907 / 0.878. All embedded sequential
correctness comparisons passed; timing noise remains visible.

Current presentation artifacts: [demo HTML](../artifacts/presentation-2026-11-13/demo/demo.html),
[engine HTML](../artifacts/presentation-2026-11-13/engines.html),
[engine JSON](../artifacts/presentation-2026-11-13/engines.json),
[strong JSON](../artifacts/presentation-2026-11-13/strong.json),
[weak JSON](../artifacts/presentation-2026-11-13/weak.json).
See the [presentation runbook](PRESENTATION_RUNBOOK.md) for commands to rehearse
in a separate output directory. MPI/CUDA/hardware hybrids, Parquet,
Linux/sanitizers, reference source rebuilds, and the reference desktop GUI were
not executed. The mini POSIX reference was inspected but not run.

## Earlier verification: 11 November 2026

Verified locally on 11 November 2026. Environment: Windows kernel 10.0.26300, Python 3.11.9 in `.venv`, MSVC 19.51, 10 physical / 16 logical processors, approximately 15.7 GiB RAM. Driver reports NVIDIA RTX 5050 Laptop GPU, 8151 MiB, compute 12.0; native CUDA toolkit absent. MPI launcher absent. [Actual diagnostics](../artifacts/validation/doctor.json) retain the probe output.

## Automated checks

| Executed command | Actual outcome |
|---|---|
| `.venv\Scripts\python.exe scripts/build.py --test` | CPU+OpenMP build succeeds;2/2 CTest cases pass |
| `.venv\Scripts\python.exe scripts/build.py --no-openmp --build-dir build/cpu --test` | Separate CPU-only build succeeds;2/2 CTest cases pass |
| `.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/validation/all-tests.xml` | 101 passed, 2 skipped; 0 failures/errors; latest run 18.73 seconds |
| `.venv\Scripts\python.exe -m ruff check hftlab tests scripts` | All checks passed |
| `.venv\Scripts\python.exe -m pip check` | No broken requirements |
| `pip install --no-build-isolation --no-deps -e .` | Editable installation and `hftlab --version` succeed |
| `python -m build --no-isolation` | Wheel and source distribution built |
| Local wheel installation into `artifacts/wheel-smoke` | Import originates from installed wheel directory; TUI CSS exists; explicit native runner capability probe succeeds |
| `.venv\Scripts\hftlab.exe` in a terminal | Actual interface opens, detects native capabilities, and exits cleanly with Ctrl+Q |

Full [JUnit XML](../artifacts/validation/all-tests.xml) and [test log](../artifacts/validation/all-tests.log) preserve the final suite. The2 skips are genuine unavailable MPI/CUDA replay gates. One nonfatal sklearn/SciPy L-BFGS deprecation warning remains; it is not hidden. CTest logs and Ruff output are in `artifacts/validation/`.

An authentic [terminal SVG capture](images/tui.svg) was exported from the real app after diagnostics completed. It can be reproduced with `python -m hftlab.tui.capture`.

Python/native probability tests cover Python, sklearn and PyTorch exported logistic/MLP weights, scalers and stable softmax within absolute tolerance2e-11. Replay tests compare event counts, orders/fills/checksums/accounting across four CPU configurations, OpenMP and CPU+OpenMP. Other tests cover malformed/corrupt input, binary import to native execution, purged splits, transactional persistence, escaped exports, cancellation/timeouts and TUI workers/navigation.

The initial sandboxed UI attempt blocked in Windows `asyncio` socketpair creation before application initialization. The same tests passed outside that restriction. A fresh CPU-only compiler probe similarly required Visual Studio PDB IPC outside the sandbox. A local setuptools temporary-path issue with braces was resolved by placing TEMP/TMP in `D:\HPC\.tmp` for packaging. These environment failures were diagnosed rather than hidden.

## End-to-end demo actually executed

```powershell
.\.venv\Scripts\hftlab.exe demo --events 10000 --epochs 15 --output artifacts/demo
```

Generated10,000 seeded predictable educational events over4 instruments. Trained real sklearn logistic and PyTorch CPU MLP models; exported weights and validation/test metadata. The demo MLP held-out classification test has1,482 samples, accuracy0.75843455, macroF1 0.63492679 and cross-entropy0.63637304. This is synthetic directional classification, not real-market evidence.

`export_holdout` selected source indices8502..9999, omitted124 cancellations referencing pre-test orders, and produced1,374 standalone events. Provenance records0 training and0 embargo events replayed, with empty book/cold-start indicators. Same initial capital100,000 and fee1bp across all execution baselines:

| Held-out strategy | Fills | Net simulated P&L |
|---|---:|---:|
| PyTorch MLP |61|263.366861 |
| Logistic |61|243.808088 |
| Momentum |106|217.251203 |
| Mean reversion |65|-289.531206 |
| HOLD |0|0 |

See [demo JSON](../artifacts/demo/demo.json), [offline demo report](../artifacts/demo/demo.html), and [engine comparison](../artifacts/demo/benchmark.html). Model weights and metadata are under `artifacts/demo/mlp/` and `logistic/`. Separate5,000-event reference training artifacts and their actual metrics remain under `models/` and are documented in [AI training](AI_TRAINING.md).

## Performance smoke actually executed

Generated100,000 correlated events/16 instruments, then ran3 measured repeats per available engine after1 warmup. Every measured candidate passed its sequential correctness gate. Reported values below are the actual stored medians, not guaranteed future performance:

| Engine | Median native seconds | Speedup |
|---|---:|---:|
| Sequential |0.0176992|1.000 |
| Native threads |0.0048029|3.685 |
| OpenMP |0.0056900|3.111 |
| CPU+OpenMP hybrid |0.0192579|0.919 |
| MPI |UNAVAILABLE|— |
| CUDA |UNAVAILABLE|— |

The hybrid is slower on this workload and is reported accordingly. [Raw samples/CI/results](../reports/performance-smoke.json) and [HTML comparison](../reports/performance-smoke.html) include phase timings and accounting. Native timing excludes CSV loading/setup and process orchestration; see [measurement scope](HPC_METHODOLOGY.md).

Strong scaling with fixed10,000-event input and weak scaling at2,000events/worker were executed at1,2,4 workers,2 measured repeats each plus warmup. Outputs: [strong JSON](../reports/strong.json), [weak JSON](../reports/weak.json), and corresponding HTML files. Small-workload timing noise remains visible; no absolute runtime threshold is asserted.

## Gates not passed or not executed

CUDA compilation/execution, MPI compilation/communication, optional hardware-dependent hybrids, Parquet/PyArrow, Linux/pthread execution, sanitizers and remote-node CI were not verified. There is no hardware-gate substitute from the CPU tests. The available hardware did not support executing those software paths in this session. No claim of production readiness or complete fulfillment of every master acceptance item is made.
