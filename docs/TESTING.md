# Testing

```bash
python scripts/build.py --test
python -m pytest -q
python -m ruff check hftlab tests scripts
python -m build --no-isolation
```

CTest validates price-time matching, partial fills, cancellation, duplicate IDs, heap growth, long/short accounting reversals, causal indicators, stable logistic/MLP inference, and the capability protocol. Explicit CHECK macros remain active in Release builds.

Pytest validates generation reproducibility, scenarios, malformed input/reference IDs, checksum corruption, versioned binary roundtrip, temporal leakage boundaries, scalers, model loading/training, Python/C inference parity, real native replay parity, configuration boundaries, transactional registry updates, escaped reports, timeout/process cancellation, TUI navigation/validation/results, and benchmark correctness gates. Native/CUDA/MPI skips identify unavailable prerequisites explicitly.

On GCC/Clang, build a separate sanitizer tree:

```bash
python scripts/build.py --debug --sanitize address --build-dir build/asan --test
python scripts/build.py --debug --sanitize undefined --build-dir build/ubsan --test
python scripts/build.py --debug --sanitize thread --no-openmp --build-dir build/tsan --test
```

Do not combine AddressSanitizer and ThreadSanitizer. MSVC sanitizer flags are not implemented by the generic option. Linux sanitizers and optional-engine execution are separate unverified gates on this Windows host. CI config describes future Linux/MPI coverage; a workflow file is not evidence it has run.

CUDA hardware gate: configure with `--cuda`, select that runner using `HFTLAB_RUNNER`, run CTest then `pytest tests/correctness -q` and real comparisons. A working driver or CPU CI job does not satisfy this gate. MPI likewise requires a matching compiled implementation and launcher, tested through actual multi-process replay.
