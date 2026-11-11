#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p results
for ranks in 1 2 4; do
 mpiexec -n "$ranks" ./build/hft_mpi_worker --symbols 64 --events 5000 --out "results/S8-strong-$ranks"
 symbols=$((16 * ranks))
 mpiexec -n "$ranks" ./build/hft_mpi_worker --symbols "$symbols" --events 5000 --out "results/S8-weak-$ranks"
 mpiexec -n "$ranks" ./build/hft_mpi_worker --symbols 64 --events 5000 --topology central --out "results/S8-central-$ranks"
done
mpiexec -n 2 ./build/hft_mpi_worker --cg
mpiexec -n 2 ./build/hft_mpi_worker --train --config configs/S6.json --out results/S6-mpi
