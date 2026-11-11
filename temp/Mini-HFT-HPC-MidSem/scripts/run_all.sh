#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results

make all data

./bin/sequential data/market.csv | tee results/sequential.txt
./bin/pthreads data/market.csv 2 | tee results/pthreads_2.txt
./bin/openmp data/market.csv 2 | tee results/openmp_2.txt

if command -v mpicc >/dev/null 2>&1 && command -v mpirun >/dev/null 2>&1; then
    make mpi
    mpirun -np 2 ./bin/mpi data/market.csv | tee results/mpi_2.txt
else
    echo "MPI compiler/runtime not installed; skipping MPI run."
fi
