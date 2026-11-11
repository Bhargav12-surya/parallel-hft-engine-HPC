# Installation

Use Python 3.11 or newer with matching optional-package wheels, a C17 compiler, CMake>=3.22, and Ninja. CPU-only builds do not need CUDA or MPI. On Debian/Ubuntu, install `build-essential python3-venv`; MPI additionally requires `libopenmpi-dev openmpi-bin`. On Windows install Visual Studio C++ Build Tools; `scripts/build.py` discovers and initializes its x64 environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,science]'
python scripts/build.py --test
hftlab doctor
```

Windows PowerShell uses `.\.venv\Scripts\Activate.ps1` or directly `.\.venv\Scripts\python.exe`. The current workspace's environment was created from `C:\Users\bharg\AppData\Local\Programs\Python\Python311\python.exe`; the default Python 3.15 lacks compatible wheels for some optional libraries.

Build variants:

```bash
python scripts/build.py --no-openmp --build-dir build/cpu --test
python scripts/build.py --build-dir build --test
python scripts/build.py --mpi --build-dir build/mpi --test
python scripts/build.py --cuda --build-dir build/cuda --test
python scripts/build.py --cuda --mpi --build-dir build/full --test
```

Set `HFTLAB_RUNNER` to an absolute executable path when selecting a nondefault build directory. MPI needs a launcher on PATH (`HFTLAB_MPI_LAUNCHER` can override its name). Use the MPI implementation discovered by CMake, not an unrelated launcher. CUDA requires `nvcc`, a supported host compiler, GPU architecture selection when needed, and a compatible driver. Installing PyTorch does not build the native CUDA backend.

Optional packages:

```bash
python -m pip install -e '.[science,parquet]'
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Training can use the dependency-free Python backend, sklearn logistic, or PyTorch MLP. CPU PyTorch is sufficient to train native C-compatible weights. See `hftlab train --help` for explicit backend/device options.

The Python wheel contains orchestration and UI code. The native runner is a separate CMake artifact, discovered in the source checkout or through `HFTLAB_RUNNER`; installing the Python wheel alone does not compile it.
