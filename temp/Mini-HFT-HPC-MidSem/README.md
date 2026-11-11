# Mini HFT + HPC Mid-Sem Project (C)

This is a deliberately small teaching version of the larger HFT/HPC project.
It keeps only four ideas:

1. Generate synthetic quote data.
2. Store the latest best bid/ask in a tiny Level-1 order book.
3. Run one simple momentum strategy.
4. Execute the same independent-symbol workload using sequential C, pthreads, OpenMP and MPI.

## Pipeline

`CSV data -> Level-1 order book -> momentum signal -> buy/sell -> P&L -> timing`

The same `simulate_symbol()` function is used by every backend. Only the way symbols are assigned to workers changes. That makes result comparison easy and keeps the viva simple.

## Files

- `src/generate_data.c` - deterministic synthetic data generator.
- `src/mini_hft.c` - CSV loader, order-book update, strategy, P&L and timer.
- `src/sequential.c` - one CPU thread, symbols one after another.
- `src/pthreads_version.c` - manual POSIX threads.
- `src/openmp_version.c` - OpenMP parallel-for version.
- `src/mpi_version.c` - MPI process-level version.
- `include/mini_hft.h` - small shared data structures and function declarations.
- `Makefile` - build commands.
- `scripts/run_all.sh` - build, generate data and run the available backends.

## Linux / WSL quick start

```bash
make all
make data
./bin/sequential data/market.csv
./bin/pthreads data/market.csv 2
./bin/openmp data/market.csv 2
```

Or simply:

```bash
./scripts/run_all.sh
```

For MPI, install an MPI implementation such as Open MPI or MPICH, then:

```bash
make mpi
mpirun -np 2 ./bin/mpi data/market.csv
```

## Change project size

```bash
./bin/generate_data 4 10000 data/market.csv
```

The arguments are `number_of_symbols events_per_symbol output_file`.
For a tiny viva demo, try 4 symbols and 10,000 events per symbol. For timing experiments, increase events to 100,000 or 1,000,000 per symbol.

## Strategy

At each event we compute:

`mid = (best_bid + best_ask) / 2`

- If the mid-price is higher than the previous mid-price, buy one unit at the ask.
- If it is lower, sell one unit at the bid.
- Do nothing when unchanged.
- Position is capped between -10 and +10.

Final mark-to-market P&L is:

`PnL = cash + final_position * last_mid`

This is intentionally a toy strategy. The project demonstrates computing and parallelization, not profitable trading.

## Expected correctness check

All backends should report the same event count, trade count, final position and P&L. Runtime can differ.

## Important limitation

This project uses a **Level-1 quote book**, not a full exchange matching engine. It stores only the current best bid, best ask and their quantities. This is intentional so that the whole project can be understood during a mid-sem viva.
