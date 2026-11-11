#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p results
for scenario in 1 2 3 4 5 6 7; do
 ./build/hft_bench --scenario "S$scenario" --events 2000 --symbols 16 --engines serial_naive,serial_opt,threaded,openmp,opencl --warmup 1 --reps 3 --out results
done
if [ -x build/hft_mpi_worker ]; then
 mpiexec -n 2 ./build/hft_mpi_worker --config configs/S8.json --out results/S8-two-ranks
else
 echo 'MPI worker unavailable: rebuild with an MPI C SDK for S8.'
fi
