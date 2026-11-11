# Validation record

Validation was performed in a Linux execution environment using GCC 13 and the supplied C sources. This record is limited to the checks actually run. Raw outputs and example result exports are in `validation/`.

| Area | Result | Evidence |
|---|---|---|
| C-only project | CMake enables C17 only; authored source/exemplar suffixes are `.c`/`.h`; no `.cpp`, `.hpp`, `.cu` | CMakeLists.txt and package manifest |
| CPU behavioral regression | **1,502 checks, 0 failures**; all 13 strategies agree across reference, optimized serial, threads and OpenMP | cpu-tests.log |
| Address/undefined behavior sanitizers | Both CTest cases pass; leak detection disabled because of the execution environment | asan-tests.log |
| OpenCL program | Builds and executes on a **POCL CPU OpenCL device**; not physical GPU validation | opencl-regression.log |
| OpenCL simulation equivalence | All 13 strategy hashes/accounting match serial; device tests vary pinned staging and one/two queues | opencl-regression.log |
| OpenCL analytics / AI | Covariance and inference match tolerances; CPU versus device trained weights match tolerance; Q kernel executes | tests/device_tests.c and opencl-regression.log |
| MPI two ranks | 3,000 events; final hash matches serial | mpi-test.log and mpi-test.json |
| Hybrid / central exchange | Four ranks, two threads per rank, including an idle rank; final hash matches serial | hybrid-central.log and JSON |
| Distributed CG | Converges in 19 iterations; residual approximately 3.26e-11 | mpi-cg.log |
| Distributed MLP SGD | Two-rank run saves a trained model | mpi-train.log |
| CLI S1–S7 | Smoke-size scenario suite completes; unsupported analytic engine combinations are explicitly skipped | scenario-suite.log and scenarios/ |
| Latency mode | CPU latency-coupled scenarios execute and export separate results | scenario-suite.log |
| C examples | Contract layout, engine equivalence, solver/risk and training examples run | example-*.log |
| Linux desktop | Compiles. All eight screens rendered via the same C drawing/layout path in headless mode | gui-build.log and preview/ |
| Native display interaction | Not executed here: X server socket/network-interface setup is restricted | xvfb.log |
| Windows x64 | GCC/MinGW C cross-build succeeds; executable DLL imports inspected and runtime DLLs bundled | windows-build.log and windows-imports.log |
| MSVC / Windows execution | **Not run** | Run scripts/build_windows.bat and windows/hft_tests.exe on Windows |
| Physical GPU / NVIDIA libraries | **Not run**; no physical GPU/NVIDIA SDK validation | Build/run on the target machine |
| Multi-node network / installer | **Not run** | Recipes and instructions supplied |

The initial MPI TCP attempt could not use network interfaces in the restricted environment. The successful tests used local shared-memory MPI transport; the hybrid/CG/training runs disabled the process-vm-copy shortcut and completed cleanly. This is not evidence of inter-node performance.

No GPU speedup or production trading performance is claimed. The short included scenarios are correctness/smoke runs, not statistically strong performance studies. Use the repeated benchmark commands and real target hardware for a report.
