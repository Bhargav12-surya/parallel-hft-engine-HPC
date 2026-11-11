# Benchmarking commands and artifacts

Run `hftlab compare --dataset data/generated/demo.csv --all --repeats 3 --output reports/engines.json`. Sequential is always first and serves as the correctness oracle. Requested unavailable backends are recorded as UNAVAILABLE, not silently substituted. Failed or incorrect backends make compare return a failure exit code. Native progress output and live UI callbacks are suppressed inside repeat runs.

Each successful row includes every timing sample, median/mean/stddev/CI, throughput, speedup, homogeneous efficiency where applicable, raw native result, checksum, and persisted experiment paths. `hftlab report reports/engines.json --output reports/engines.html` exports an offline comparison chart. CSV/JSON/Markdown export is also available.

Strong scaling holds the dataset fixed. Weak scaling streams a different seeded dataset with event and instrument count proportional to workers. Both execute a sequential correctness reference for every workload. The scaling output includes `scaling_speedup` and `scaling_efficiency` relative to its smallest tested worker count; weak efficiency is T(reference)/T(p), while strong efficiency also divides by the worker-count ratio.

```bash
hftlab scaling --dataset data/generated/demo.csv --engine threaded --workers 1,2,4 --repeats 3 --output reports/strong.json
hftlab scaling --dataset data/generated/demo.csv --engine threaded --workers 1,2,4 --weak --events-per-worker 2000 --repeats 3 --output reports/weak.json
```

Avoid tiny workloads when drawing speedup conclusions. Use at least as many independent instrument groups as workers. Pair trading has fewer groups. Hardware counter availability, compiler flags, timing boundaries, and memory costs are explained in [HPC methodology](HPC_METHODOLOGY.md).
