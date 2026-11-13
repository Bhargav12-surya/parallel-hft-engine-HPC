"""Portable CMake driver, including Visual Studio tool discovery on Windows."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_tool(name: str) -> str:
    adjacent = Path(sys.executable).parent / (name + (".exe" if os.name == "nt" else ""))
    found = str(adjacent) if adjacent.is_file() else shutil.which(name)
    if not found:
        raise RuntimeError(f"Missing {name}; install the dev extra: python -m pip install -e '.[dev]'")
    return found


def build_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    if os.name == "nt" and not shutil.which("cl") and not shutil.which("gcc"):
        vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
        if not vswhere.is_file():
            raise RuntimeError("Install Visual Studio C++ Build Tools or MinGW-w64")
        location = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
                                           "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
        batch = Path(location) / "VC/Auxiliary/Build/vcvars64.bat"
        if not batch.is_file():
            raise RuntimeError("Visual Studio C/C++ workload not found")
        # cmd's /s /c consumes an outer quote pair itself. Passing this as an argv
        # list would add C-runtime escaping, which cmd does not understand.
        output = subprocess.check_output(f'cmd.exe /d /s /c ""{batch}" >nul && set"', text=True)
        for line in output.splitlines():
            name, separator, value = line.partition("=")
            if separator and name:
                environment[name] = value
        environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("Path", environment.get("PATH", ""))
    return environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cuda-arch", help="CMake CUDA architecture list, e.g. 120 or 86;89")
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--no-openmp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sanitize", choices=("address", "thread", "undefined"))
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.cuda_arch and not args.cuda:
        parser.error("--cuda-arch requires --cuda")
    environment = build_environment()
    cmake = find_tool("cmake")
    configure = [cmake, "-S", str(ROOT), "-B", str(args.build_dir), "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=" + ("Debug" if args.debug else "Release"),
                 "-DHFT_ENABLE_OPENMP=" + ("OFF" if args.no_openmp else "ON"),
                 "-DHFT_ENABLE_MPI=" + ("ON" if args.mpi else "OFF"),
                 "-DHFT_ENABLE_CUDA=" + ("ON" if args.cuda else "OFF"),
                 "-DCMAKE_MAKE_PROGRAM=" + find_tool("ninja"),
                 "-DHFT_SANITIZER=" + (args.sanitize or "")]
    if args.cuda_arch:
        configure.append("-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_arch)
    subprocess.run(configure, env=environment, check=True, cwd=ROOT)
    subprocess.run([cmake, "--build", str(args.build_dir), "--parallel"], env=environment, check=True, cwd=ROOT)
    if args.test:
        subprocess.run([find_tool("ctest"), "--test-dir", str(args.build_dir), "--output-on-failure"],
                       env=environment, check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
