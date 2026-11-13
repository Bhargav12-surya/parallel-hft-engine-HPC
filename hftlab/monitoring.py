"""Read-only diagnostics. Compiled capability and runtime probes stay distinct."""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from hftlab.engines import capabilities, engine_available


def _probe(arguments: list[str]) -> dict:
    try:
        completed = subprocess.run(arguments, capture_output=True, text=True, timeout=10)
        return {"status": "AVAILABLE" if completed.returncode == 0 else "INCOMPATIBLE",
                "detail": (completed.stdout or completed.stderr).strip()[:4000]}
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "UNAVAILABLE", "detail": str(error)}


def diagnose() -> dict:
    capability = capabilities()
    result = {"schema_version": 1, "os": platform.platform(), "kernel": platform.release(),
              "python": sys.version.split()[0], "python_executable": sys.executable,
              "virtual_environment": sys.prefix != sys.base_prefix, "cpu": platform.processor(),
              "logical_processors": os.cpu_count(), "native": capability,
              "engines": {name: "AVAILABLE" if engine_available(capability, name) else "UNAVAILABLE"
                          for name in ("sequential", "threaded", "openmp", "mpi", "cuda", "hybrid")},
              "numa": {"status": "NOT TESTED", "detail": "Use numactl --hardware on Linux for node topology"}}
    try:
        import psutil
        result.update(physical_processors=psutil.cpu_count(logical=False),
                      memory_bytes=psutil.virtual_memory().total,
                      eligible_cpus=psutil.Process().cpu_affinity())
    except (ImportError, AttributeError, OSError):
        result.update(physical_processors=None, memory_bytes=None,
                      eligible_cpus=list(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
    for name, args in (("cmake", ["cmake", "--version"]), ("compiler", ["cc", "--version"]),
                       ("mpi", ["mpiexec", "-help" if os.name == "nt" else "--version"]),
                       ("cuda_toolkit", ["nvcc", "--version"]),
                       ("gpu", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"])):
        executable = shutil.which(args[0])
        adjacent = Path(sys.executable).parent / (args[0] + (".exe" if os.name == "nt" else ""))
        if executable is None and adjacent.is_file():
            executable = str(adjacent)
        result[name] = _probe([executable] + args[1:]) if executable else {"status": "UNAVAILABLE", "detail": f"{args[0]} is not on PATH"}
    if result["compiler"]["status"] == "UNAVAILABLE" and os.name == "nt":
        vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
        if vswhere.is_file():
            result["compiler"] = _probe([str(vswhere), "-latest", "-products", "*", "-property", "installationPath"])
            result["compiler"]["detail"] += " (Visual Studio installation; actual compiler is recorded in native capabilities)"
    result["optional_dependencies"] = {name: "AVAILABLE" if importlib.util.find_spec(name) else "UNAVAILABLE"
                                       for name in ("numpy", "sklearn", "torch", "plotly", "pyarrow", "textual", "psutil")}
    result["profilers"] = {name: shutil.which(name) for name in ("perf", "valgrind", "nsys", "ncu", "numactl")}
    return result
