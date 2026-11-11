"""Native executable discovery and capability-aware command construction."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from hftlab.configuration import SimulationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_runner() -> Path:
    configured = os.environ.get("HFTLAB_RUNNER")
    candidates = [Path(configured)] if configured else []
    suffix = ".exe" if os.name == "nt" else ""
    for directory in ("build/Release", "build", "build/cpu/Release", "build/cpu", "build/full/Release", "build/full", "bin"):
        candidates.append(PROJECT_ROOT / directory / ("hftlab_runner" + suffix))
    installed = shutil.which("hftlab_runner")
    if installed:
        candidates.append(Path(installed))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError("Native runner is not built. Run: python scripts/build.py (inside the project virtual environment)")


def capabilities() -> dict:
    try:
        runner = find_runner()
        completed = subprocess.run([str(runner), "--capabilities"], capture_output=True, text=True,
                                   timeout=15, check=True)
        value = json.loads(completed.stdout)
        if not isinstance(value, dict) or value.get("schema_version", 1) != 1:
            raise ValueError("Invalid native capability schema")
        value["runner"] = str(runner)
        return value
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        return {"schema_version": 1, "engines": {}, "error": str(error)}


def engine_available(capability: dict, name: str) -> bool:
    value = capability.get("engines", {}).get(name, False)
    return value.get("available", False) if isinstance(value, dict) else bool(value)


def validate_engine(config: SimulationConfig, capability: dict | None = None) -> dict:
    capability = capability or capabilities()
    if not engine_available(capability, config.engine):
        raise RuntimeError(f"Engine {config.engine} is UNAVAILABLE: {capability.get('error', 'not compiled or no compatible runtime')}")
    if config.engine == "hybrid":
        for component in config.components:
            name = "sequential" if component == "cpu" else component
            if not engine_available(capability, name):
                raise RuntimeError(f"Hybrid component {component} is UNAVAILABLE; no substitution is permitted")
        supported = capability.get("hybrid_combinations")
        if supported is not None and sorted(config.components) not in [sorted(c) for c in supported]:
            raise RuntimeError(f"Hybrid combination {'+'.join(config.components)} is not implemented by this runner")
    if config.engine == "mpi" or (config.engine == "hybrid" and "mpi" in config.components):
        if not mpi_launcher():
            raise RuntimeError("MPI launcher is missing; install OpenMPI, MPICH, or MS-MPI and add mpiexec to PATH")
    return capability


def mpi_launcher() -> str | None:
    return shutil.which(os.environ.get("HFTLAB_MPI_LAUNCHER", "mpiexec")) or shutil.which("mpirun")


def command(config: SimulationConfig, output: Path) -> list[str]:
    arguments = [str(find_runner()), "--input", str(config.dataset), "--output", str(output),
                 "--engine", config.engine, "--cpu-mode", config.cpu_mode, "--threads", str(config.threads),
                 "--strategy", config.strategy, "--capital", str(config.capital),
                 "--fee-bps", str(config.fee_bps), "--slippage-ticks", str(config.slippage_ticks),
                 "--latency-ns", str(config.latency_ns), "--position-limit", str(config.position_limit),
                 "--max-order", str(config.max_order), "--max-loss", str(config.max_loss)]
    if config.model:
        arguments.extend(["--model", str(config.model)])
    if config.engine == "hybrid":
        arguments.extend(["--components", ",".join(config.components)])
    if config.engine == "mpi" or (config.engine == "hybrid" and "mpi" in config.components):
        arguments = [str(mpi_launcher()), "-n", str(config.threads)] + arguments
    return arguments
