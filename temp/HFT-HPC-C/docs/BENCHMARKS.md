# Measurement guide

Run from the project root. Use optimized Release builds, fixed configuration and seed, and record available hardware. Repeated simulation runs store a sorted timing CSV in addition to the last-run JSON/CSV and SVG equity report. The printed median comes from the repetitions; the JSON `seconds`, trace and latency histogram belong to the final measured run. They are intentionally distinct observations. Warm-up runs are discarded. Compare like-for-like compiler/build/configuration, not a live throttled desktop to the CLI.

```sh
build/hft_bench --scenario S1 --engines serial_naive,serial_opt,threaded,openmp --symbols 1 --events 5000 --warmup 2 --reps 5 --out results/S1
build/hft_bench --scenario S2 --engines serial_opt,openmp,opencl --symbols 64,256 --threads 1,2,4,8 --batch 1,64,1024 --events 5000 --warmup 1 --reps 3 --out results/S2
build/hft_bench --scenario S3 --grid 32 --engines serial_opt,openmp --symbols 16 --events 2000 --warmup 0 --reps 1 --out results/S3
build/hft_bench --scenario S4 --engines serial_naive,serial_opt,openmp,opencl,nvidia_libs --out results/S4
build/hft_bench --scenario S5 --engines serial_opt,openmp,opencl --samples 2048 --out results/S5
build/hft_bench --scenario S6 --engines serial_opt,openmp,opencl --samples 2048 --epochs 10 --out results/S6
build/hft_bench --scenario S7 --threads 4 --out results/S7
mpiexec -n 4 build/hft_mpi_worker --config configs/S8.json --out results/S8
```

Create parent result directories before supplying nested output paths. S3 defaults to a modest 4×4 grid; `--grid 32` makes 1024 complete backtests and can take substantial time. S4 uses 256 observations and 16 assets with 10,000 Monte Carlo paths. Its matrix timing and subsequent CPU factorization/CG/risk timing are separate. Unsupported combinations are skipped explicitly. `nvidia_libs` is not a tick engine.

## Metrics

- Throughput = actual processed events / measured wall seconds.
- CPU latency histogram = elapsed time of one owned pair step (one or two symbols), including generation, book, features, decision, risk and fills. It is **not** pure feed-arrival-to-order latency. Timer-call overhead is measured and subtracted consistently on CPU paths. It is clamped at zero.
- Histogram bins have bounded logarithmic precision; reported percentiles use bin upper bounds clipped by observed maximum. GPU/MPI paths do not synthesize per-event percentiles; zero means no such observations. Device batch latency is separate.
- OpenCL H2D/kernel/D2H durations use profiling events. Wall time includes transfer/copy/synchronization work. With two queues, kernel duration is the sum of queue durations and may exceed elapsed device wall time due to overlap. Context/JIT creation is excluded from simulation timing.
- Speedup uses the optimized serial reference or the measured naive baseline when present. Efficiency divides by reported workers. Karp–Flatt is meaningful for comparable strong-scaling runs and more than one worker. The current serial reference is a separate single run, so uncertainty in speedup must be considered.
- Cash, fees and marked P&L are integer ticks. Total cost basis follows proportional integer allocation on partial closes. Profit factor and win rate currently describe **sampled equity increments**, not completed round-trip trades. Sharpe and Sortino are per-sample ratios, not annualized. Do not compare them to annualized financial performance claims.
- Slippage is recorded relative to the prior mark and reported as total signed share-ticks divided by fill count; it is not a volume-weighted bps metric.
- Max drawdown is sampled portfolio equity drawdown. Sampling may miss intrabatch extremes. `latency_cost` is latency-mode P&L minus a zero-delay reference with the same seed.

Exact verification compares strategy order/fill hashes, market stream hashes, integer P&L, per-symbol positions/cash where full states are present. Device gradients/matrices use tolerances. Hash agreement alone is not a mathematical proof; behavioral book/accounting tests supplement it.

S7 separately measures correct OpenMP critical/atomic/reduction and native-thread shared-atomic/adjacent/padded counters. Work descriptions differ and are printed in the CSV; compare synchronized alternatives on equal work and interpret false-sharing as a memory-layout experiment. There is no intentionally undefined shared increment.

S8 strong scaling fixes total symbols/events; weak scaling increases symbols with ranks. Compare `--topology central` with decomposed mode. Central mode transfers books to rank zero every batch and acknowledges each peer, exposing a communication bottleneck. Use `--cg` and `--train --config ...` for distributed solvers/training.
