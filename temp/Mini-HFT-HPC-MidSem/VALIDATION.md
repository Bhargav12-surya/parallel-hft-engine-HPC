# Validation notes

The project was compiled with GCC 14.2.0 using `-O2 -std=c17 -Wall -Wextra -pedantic`.

For the included 4-symbol x 10,000-event dataset, the executed sequential, pthreads and OpenMP versions all produced:

```text
events=40000
trades=27999
final_position=-1
pnl=-207.625000
```

The measured runtime is intentionally not treated as a portable benchmark because it depends on hardware, OS scheduling, compiler/runtime versions and workload size.

A larger one-run demonstration using 8 symbols x 100,000 events showed that parallel versions can become faster once enough useful work exists, but formal results should use repeated runs on the target machine.

MPI source is included and was syntax-checked, but the packaging environment did not have `mpicc` / `mpirun`, so no MPI runtime result is claimed here. Install Open MPI or MPICH and test `make mpi` plus `mpirun -np 2 ./bin/mpi data/market.csv` before presenting MPI timing numbers.
