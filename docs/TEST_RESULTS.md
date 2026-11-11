# Executed test and demonstration evidence

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
