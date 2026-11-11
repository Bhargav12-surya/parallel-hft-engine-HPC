@echo off
setlocal
cd /d "%~dp0.."
set "HFT_BIN=build\Release"
if exist windows\hft_bench.exe set "HFT_BIN=windows"
if not exist results mkdir results
for %%S in (1 2 3 4 5 6 7) do (
 "%HFT_BIN%\hft_bench.exe" --scenario S%%S --events 2000 --symbols 16 --engines serial_naive,serial_opt,threaded,openmp,opencl --warmup 1 --reps 3 --out results
 if errorlevel 1 exit /b 1
)
if exist "%HFT_BIN%\hft_mpi_worker.exe" (
 mpiexec -n 2 "%HFT_BIN%\hft_mpi_worker.exe" --config configs\S8.json --out results\S8-two-ranks
) else (
 echo MPI worker unavailable. Rebuild with an MPI C SDK to execute S8.
)
