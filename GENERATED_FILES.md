# Where the generated files are saved

Everything created for HFT-LAB is inside **`D:\HPC`**.

| Location | Contents |
|---|---|
| `D:\HPC\hftlab` | Python CLI, terminal UI, orchestration, datasets, AI, experiments, benchmarks and reports code |
| `D:\HPC\native` | C simulation/order book, CPU/OpenMP/MPI paths, native AI inference, CUDA kernels and C tests |
| `D:\HPC\tests` | Python unit, integration, correctness and terminal UI tests |
| `D:\HPC\configs` | Simulation, benchmark, AI and scenario presets |
| `D:\HPC\schemas` | Versioned configuration, event, model, progress, result and experiment contracts |
| `D:\HPC\scripts` | Installation, build, demo and developer cleanup scripts |
| `D:\HPC\docs` | Architecture, installation, user guide, engine/AI/data documentation, implementation status and test evidence |
| `D:\HPC\docs\images\tui.svg` | Authentic terminal screenshot |
| `D:\HPC\data\generated` | Useful generated demo, benchmark and learning-reference datasets with manifests |
| `D:\HPC\models` | Trained reference logistic and MLP models, metadata and numerical fixtures |
| `D:\HPC\results` | Persisted native simulations, configurations, manifests and logs |
| `D:\HPC\experiments\registry.sqlite3` | Searchable experiment history database |
| `D:\HPC\reports` | Engine comparisons, held-out evaluation and strong/weak scaling reports |
| `D:\HPC\artifacts\demo` | Complete demonstration: dataset, trained models, held-out replay, results and HTML/JSON reports |
| `D:\HPC\artifacts\validation` | Build/test/package evidence and cleanup record |
| `D:\HPC\build` | Built native runner and native build files; CPU-only variant is under `build\cpu` |
| `D:\HPC\dist` | Python wheel and source-distribution packages |
| `D:\HPC\.venv` | Installed project environment and dependencies; required for the existing launcher |
| `D:\HPC\.github\workflows` | Continuous-integration workflow |

Root project files include `README.md`, `AGENTS.md`, `pyproject.toml`, `CMakeLists.txt`, `CMakePresets.json`, `Makefile`, `MANIFEST.in`, `LICENSE`, `.gitignore` and `.env.example`.

The detailed [file inventory](artifacts/generated-files.json) lists individual paths relative to `D:\HPC`. It excludes installed dependencies in `.venv`, caches, and your original `temp` reference folder.

## Cleanup performed

Removed 105 disposable files across 19 locations: Python/test/lint caches, an empty temporary directory, generated packaging metadata, a duplicate wheel-install test copy, and the unused initial `agent-smoke` dataset/model. This recovered 1,308,418 bytes (approximately 1.25 MiB).

Moved the two loose root diagnostic/benchmark logs into `artifacts\validation` as `initial-doctor.json` and `initial-engine-comparison.log`. Useful models, datasets, results, reports, packages, native executables and the Python environment were retained. Your original `D:\HPC\temp` folder was preserved.

The exact removed/moved paths are recorded in [cleanup.json](artifacts/validation/cleanup.json). The disposable `artifacts/wheel-smoke` copy mentioned in historical test evidence has been removed after its successful verification.

## Launch

From `D:\HPC`:

```powershell
.\.venv\Scripts\hftlab.exe
```

Start reading [README.md](README.md), [implementation status](docs/IMPLEMENTATION_STATUS.md), or the [demo report](artifacts/demo/demo.html).
