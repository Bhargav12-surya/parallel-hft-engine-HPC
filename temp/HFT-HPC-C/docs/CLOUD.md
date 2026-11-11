# Cloud and network lab

This lab models the effect of delayed order arrival. It does not provision or contact a cloud provider. Presets are explicit educational assumptions: local 0 ns, colocated 50 microseconds, same-region 500 microseconds and cross-region 20 milliseconds. These are not provider guarantees or measured network latencies.

Verification mode has zero simulated delay and targets identical discrete results. Latency-coupled mode estimates prior-batch compute time, multiplies it by the configured amplification, adds simulated network delay/jitter, queues orders by arrival timestamp and rechecks risk at arrival. Delays depend on measured compute and scheduling, so this mode is deliberately not bitwise reproducible. MPI currently adds the configured network component; its compute amplification/jitter model is not the same as CPU/OpenCL.

Run paired experiments with identical seed, market, strategy and limits:

```sh
build/hft_bench --config configs/latency.json --engines serial_opt,openmp --warmup 0 --reps 1 --out latency-results
```

Compare P&L, fill counts, rejected/dropped delayed orders, and the zero-delay reference. A slower engine does not necessarily produce lower P&L on every synthetic path; report what happened.

Illustrative cost = measured seconds / 3600 × entered hourly price. Prices in configs and the UI are assumptions, not current market quotes. For an actual deployment, obtain the selected provider's current price and include CPU/GPU memory, network/egress, storage, minimum billing and idle time separately.

For a remote MPI experiment, install the same C build and MPI runtime on every node, ensure the normal MPI launcher can reach the nodes using your existing secured administration setup, and run the supplied worker. Do not expose an unauthenticated desktop/worker service to the internet. Record instance family, vCPU count, memory, region, rank placement and transport alongside results. The project includes no credentials and asks for none.
