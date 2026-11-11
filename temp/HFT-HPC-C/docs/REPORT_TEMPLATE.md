# Project report template

## Title
Parallel Synthetic Trading and Risk Simulation in C

## Abstract
Describe the bounded agent-based market, deterministic book/strategy pipeline, C-only architecture and CPU/MPI/OpenCL experiments. State the actual hardware used and distinguish completed validation from unavailable devices.

## Problem and objectives
Explain latency versus throughput, serial reference correctness, parallel decomposition, memory locality, distributed communication and AI workloads. Do not imply that GPU batching makes a real per-tick trading gateway faster.

## Method
Include the architecture and ownership diagram, Philox indexing, integer contracts, fixed book bounds, risk-budget partition, strategy formulas and AI temporal split. Explain why native C and OpenCL C replace the original C++ UI/CUDA source stack.

## Experimental setup
Record OS, CPU, memory, compiler/options, runtime versions, actual device, rank/thread placement, config/seed, warm-ups, repetitions and sample count. Attach JSON/CSV and the validation logs. Separate JIT/setup, transfer, kernel and end-to-end time.

## Results tables
Use one row per engine/configuration. Show serial equivalence, measured median/min/max, throughput, CPU pair-step p50/p99, transfer/communication fractions, scaling efficiency and dataset accuracy. Mark unavailable observations as unavailable, never zero-speed measurements.

## Discussion
Discuss overhead crossover, memory limits, bounded-book effects, finite sample noise, synchronization costs, central state traffic, nonannualized risk metrics and model leakage controls. Explain any engine that was slower than the reference.

## Limitations
Copy the relevant boundaries from PHASES.md. Note physical GPU/Windows execution/multi-node validation status. Do not claim a live exchange, financial profitability, exact Avellaneda–Stoikov optimization or production HFT latency.

## Conclusion
Answer which implementations improved which measured workloads, and what evidence supports the conclusion. Describe remaining experiments, not invented projected speedups.

## Appendices
Configuration files, full result exports, compiler logs, test logs, relevant C code, third-party licenses and the viva questions.
