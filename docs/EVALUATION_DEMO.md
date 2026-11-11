# Evaluation demonstration

Activate the virtual environment. `hftlab` launches the real terminal UI; `hftlab doctor` prints capabilities and detected hardware.

One local demonstration:

```bash
hftlab demo --events 10000 --epochs 15 --output artifacts/demo
```

It saves dataset+hash, both model families, native simulations, repeated engine comparisons, and offline HTML. Predictable data is deliberately synthetic and must not be presented as real-world profitability evidence. Optional unavailable engines appear explicitly. Inspect `artifacts/demo/demo.json`, `demo.html`, and `benchmark.html`.

Interactive demonstration: Home → Data (generate/inspect/validate) → Strategy → Engine (four CPU options or other categories) → Configure → Monitor → Results → History. Training and Diagnostics have dedicated pages. Benchmarks run actual backends outside UI rendering measurements. Use Cancel to stop a long simulation, then inspect its recorded state.

Larger workload and scalability:

```bash
hftlab generate --events 1000000 --instruments 32 --scenario correlated --output data/generated/benchmark.csv
hftlab compare --dataset data/generated/benchmark.csv --all --threads 4 --repeats 5 --output reports/large.json
hftlab scaling --dataset data/generated/benchmark.csv --engine threaded --workers 1,2,4 --repeats 3 --output reports/strong.json
hftlab scaling --dataset data/generated/benchmark.csv --engine threaded --workers 1,2,4 --weak --events-per-worker 10000 --output reports/weak.json
hftlab report reports/large.json --output reports/large.html
```

Explain native vs end-to-end timing, measured throughput, event-tail latency scope, correctness checksums, homogeneous efficiency, and memory/partition bottlenecks. OpenMP uses genuine parallel loops; MPI and CUDA require their actual toolchains/runtime and should be shown only if `doctor` and execution tests verify them. Do not claim that optional paths passed on this host.

AI evaluation uses chronological train/validation/test boundaries with purge and train-only scaling. The model's validation/test metrics are stored with native weights. The demo exports `heldout.csv` from the test interval, repairs references to earlier orders, starts book/features cold, and replays MLP/logistic/momentum/mean-reversion/HOLD with equal costs and capital. Its manifest records zero training/embargo events replayed. Cold-start execution features can differ from classification test features that retained causal prehistory. Synthetic held-out profitability is still not real-market evidence.
