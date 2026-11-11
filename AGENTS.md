# HFT-LAB development

- Keep all simulation/strategy matching and accounting in native C/CUDA. Python owns orchestration, datasets, training, reporting, and presentation.
- Preserve `temp/` reference projects. Do not use historical reference logs as evidence for current tests.
- Preserve deterministic per-group event ordering; compare every backend against the sequential oracle.
- Never silently substitute missing MPI/CUDA/hybrid components, fabricate metrics, or report unavailable counters as measured zeros.
- Keep protocol/schema versions explicit. Native models are numerical artifacts, never pickle.
- Validate meaningful changes using CTest, relevant pytest tests, and Ruff. UI tests on this Windows sandbox require loopback socket support for asyncio initialization.
- Document implemented/tested/untested/partial/unavailable scope honestly in docs/IMPLEMENTATION_STATUS.md and docs/TEST_RESULTS.md.
- Do not commit generated datasets, models, experiment databases, or large benchmark artifacts by default.
