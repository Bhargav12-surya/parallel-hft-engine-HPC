# HFT / HPC Studio — C edition

A C17 research and teaching application for synthetic limit-order markets, strategy experiments, parallel execution, portfolio risk and small AI models. All application and example sources are `.c` / `.h`. The CMake project enables **C only**. There are no `.cpp`, `.hpp`, `.cu`, or Python application scripts.

## Start on Windows

1. Extract the entire ZIP to a writable folder.
2. Open `windows/hft_studio.exe`. Keep the bundled DLLs beside the executables. These bundled x64 binaries include the serial, threaded and OpenMP engines.
3. Choose a strategy and engine, then **Run**. Use **Results → Export report pack** to save CSV, JSON, SVG, PNG, configuration and model files into `results/` relative to the working directory.
4. For a command-line check, open a terminal in `windows/` and run:

```bat
hft_tests.exe
hft_bench.exe --scenario S1 --engines serial_naive,serial_opt,threaded,openmp --symbols 2 --events 2000 --reps 3 --warmup 1 --out results
```

The Windows binaries were cross-compiled; their interactive behavior has not been executed on a Windows desktop here. Rebuild on your Windows machine for MPI, OpenCL or NVIDIA library support.

## Build from source

Requirements: a C17 compiler and CMake 3.22+. No internet fetch occurs during CMake configuration; cJSON is vendored. Optional packages are detected locally.

Windows: open an **x64 Native Tools Command Prompt for Visual Studio 2022** and run:

```bat
scripts\build_windows.bat
build\Release\hft_studio.exe
```

Linux:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/hft_studio
```

The Linux native desktop requires X11 development libraries and a display. `hft_bench` works headlessly. `hft_studio --render` writes eight software-rendered screen previews without a display when the X11 target was built. The previews use the same C layout/drawing functions and a bitmap font; native Windows uses Segoe UI.

Useful options: `HFT_GUI`, `HFT_TESTS`, `HFT_OPENMP`, `HFT_MPI`, `HFT_OPENCL`, `HFT_NVIDIA`, `HFT_SANITIZE`. CUDA Toolkit is optional and is required only when `HFT_NVIDIA=ON`.

## What is included

- Deterministic Philox streams, synthetic Hawkes/jump-diffusion market, integer price-time books, position accounting and risk checks.
- Ten traditional strategies plus MLP, tabular Q-learning and a k-means/UCB selector.
- Serial reference, optimized serial, native threads, OpenMP, MPI and hybrid execution.
- OpenCL C simulation, covariance, inference, MLP training and Q-learning kernels; pinned staging and one/two compute queues.
- Optional cuBLAS/cuRAND through their C interfaces for covariance and Gaussian risk inputs.
- Native C desktop, benchmark scenarios S1–S8, model/config persistence, exports, lessons, code viewer and network/cost experiments.
- Tests, small C examples, build/run scripts, installer recipe, report template and viva questions.

## C-only adaptations and limits

The original specification named C++ ImGui/ImPlot and custom CUDA `.cu` kernels. This edition replaces the UI with native C drawing and custom device kernels with **OpenCL C**. It does not rename OpenCL as CUDA. cuBLAS/cuRAND remain optional C-API services. There is no cuDNN integration or CUDA Graph implementation.

This is a bounded synthetic educational simulator, not an exchange gateway or live trading system. The desktop is a compact implementation, not every decorative widget from the original design specification. Exact phase coverage, differences, metric definitions and hardware validation limits are in [docs/PHASES.md](docs/PHASES.md) and [docs/VALIDATION.md](docs/VALIDATION.md).

## Where to go next

- [Quick start](docs/QUICKSTART.md)
- [Architecture and ownership](docs/ARCHITECTURE.md)
- [Strategies and model equations](docs/STRATEGIES.md)
- [Benchmark methodology](docs/BENCHMARKS.md)
- [GPU and MPI setup](docs/BACKENDS.md)
- [Cloud/network lab](docs/CLOUD.md)
- [Report template](docs/REPORT_TEMPLATE.md)
- [Viva questions](docs/VIVA.md)

Use `hft_bench --list` to inspect actual build/runtime availability. Missing backends are reported explicitly; their measurements are not fabricated.
