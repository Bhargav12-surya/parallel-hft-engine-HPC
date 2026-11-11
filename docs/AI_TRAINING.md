# Offline AI training and native inference

HFT-LAB trains actual three-class classifiers and exports their learned numerical
weights for C inference. AI is a trading strategy; it uses the same sequential,
threaded, OpenMP and other available engine adapters as classical strategies.
Python does not supply precomputed BUY/SELL labels to the native simulation.

```powershell
.venv\Scripts\python.exe -m hftlab generate --events 5000 --instruments 4 --scenario predictable --output data/generated/learning-reference.csv
.venv\Scripts\python.exe -m hftlab train --dataset data/generated/learning-reference.csv --kind logistic --backend python --epochs 20 --output models/logistic-reference
.venv\Scripts\python.exe -m hftlab train --dataset data/generated/learning-reference.csv --kind mlp --backend python --epochs 20 --output models/mlp-reference
.venv\Scripts\python.exe -m hftlab simulate --dataset data/generated/learning-reference.csv --strategy ai --model models/mlp-reference/model.hftmodel
```

On Linux replace `.venv\Scripts\python.exe` with `.venv/bin/python`.

## Features and target

`hftlab/ai/learning.py::FeatureState` maintains separate bounded histories for
each instrument. The native core implements the same order and double precision
formulas. At an event with price `p`, the features are:

| Index | Name | Formula |
|---|---|---|
| 0 | return1 | `(p - previous_price) / previous_price`; initially zero |
| 1 | momentum4 | `(p - price_4_events_ago) / price_4_events_ago`; zero until available |
| 2 | momentum16 | Same expression at a 16-event lag |
| 3 | volatility16 | Population standard deviation of the last 16 one-event returns, including the current return; initial return zero |
| 4 | imbalance | `side_sign * quantity / (quantity + 100)`; BUY +1, SELL -1 |
| 5 | spread | `2 / max(p, 1)` |

All event types update history. Cancellation events use the previous price and
zero quantity. Prices here are event prices, including passive LIMIT prices,
rather than exclusively midprice observations. The last two features are
event-flow and two-tick spread proxies; they are not multi-level order-book
imbalance or a measured dynamic book spread.

For a configurable per-instrument event horizon (default 4), the target is BUY
when the future event-price return is greater than `threshold` (default .00015),
SELL below the negative threshold, and HOLD otherwise. Future prices are used
only in labels. Training direction classification is distinct from order
execution, fees, latency, inventory and risk checks performed during simulation.

## Leakage controls

Samples remain in global chronological event order; future targets are formed
within the same instrument. Default splits are 70% train, 15% validation and 15%
test. Training and validation labels whose future event index crosses their
partition boundary are removed. An embargo of at least `max(16, horizon)` global
events separates later partitions. Means and population standard deviations are
fit only on retained training features. The same means and scales are exported.
Training examples may be shuffled inside the training partition for optimization;
validation and test membership remain chronological.

Training reads at most 20,000 events by default and records `events_read` and
`truncated`. The Python API exposes `max_samples` up to one million. The entire
source dataset is validated first. It is not accurate to describe a model trained
on a truncated prefix as trained on the full dataset.

## Implemented models and optimizers

The logistic baseline is a learned 6-to-3 softmax linear classifier. The MLP is
6-to-H-to-3, with one ReLU hidden layer (default H=12; maximum 128), and stable
softmax output. Class order is SELL, HOLD, BUY. The dependency-free backend uses
minibatch Adam, capped inverse-frequency class weighting, gradient clipping,
learning-rate decay, validation loss early stopping and best-checkpoint selection.

`--backend auto` selects installed scikit-learn for logistic regression and
installed PyTorch for the MLP, otherwise the dependency-free trainer. Explicit
`--backend python`, `sklearn`, or `torch` chooses a backend without substitution.
The scikit-learn logistic backend uses balanced LBFGS and requires examples of
all three classes. PyTorch uses float64 Adam and supports `--device cuda` only
when that installation can actually use a compatible CUDA device. Optional
backend code is not evidence that that backend or GPU has been tested locally.
The PyTorch CPU path was executed with version 2.6.0+cpu; GPU training remains
an explicit unavailable hardware gate on this environment. Tiny networks use at
most four PyTorch CPU workers to avoid unnecessary oversubscription.

## Artifact contract

An output directory contains `model.hftmodel`, `metadata.json`, `history.json`,
and `model.hftmodel.fixture.json`. An explicit `name.hftmodel` output stores its
metadata as `name.hftmodel.json`. Metadata includes the dataset SHA256, model
SHA256, feature order, labels, scaler, architecture, split boundaries, sample
counts, seed, actual backend/device/library versions, best epoch and held-out
metrics. Model loading verifies a present metadata checksum. Safe numerical text
is used instead of Python pickle.

The whitespace-delimited native numerical format is:

```text
HFTMODEL 1
6 H 3
mean[6]
scale[6]
W1[H][6]       # omitted for H=0
b1[H]         # omitted for H=0
W2[3][H or 6]
b2[3]
```

The comments and brackets above explain the layout; exported files contain only
the header and numerical values. Weights are output-major. Logistic regression
uses H=0. Scales must be at least 1e-15; dimensions, numerical length and finite values
are validated. Models over 2 MiB are rejected by the Python loader.

## Numerical and simulation verification

`tests/unit/test_ai.py` checks causal/instrument-isolated features, purging,
train-only scaling, reproducible learned weights, a separable three-class
learning problem, artifact corruption and inference round trips.

`tests/correctness/test_ai_native.py` trains logistic and MLP models, sends held-out
raw features to the compiled runner, and compares native probabilities with the
Python forward pass to a double-precision tolerance. The runner interface is:

```text
hftlab_runner --infer --model model.hftmodel --features features.txt
```

`features.txt` contains one whitespace-delimited six-double row per prediction.
The JSON output contains `schema_version: 1` and `probabilities`. Further tests
replay the exported MLP strategy across available CPU modes, OpenMP, Hybrid,
MPI and CUDA and compare orders, trades, fees, equity and deterministic checksums
against the sequential oracle. Unavailable backends are explicit skipped
hardware gates, never successful GPU/MPI executions. Consult `TEST_RESULTS.md`
for the actual current environment's verification.

## Measured reference training

The reference files under `models/logistic-reference` and `models/mlp-reference`
were actually trained on `data/generated/learning-reference.csv`: 5,000
`predictable` events, four instruments, seed 42, horizon 4, threshold .00015,
Python minibatch Adam, maximum 20 epochs, seed 42.

| Model | Completed epochs | Test accuracy | Test macro F1 |
|---|---:|---:|---:|
| Logistic regression | 19 | 0.8046448087 | 0.6232251640 |
| One-hidden-layer MLP | 17 | 0.7240437158 | 0.5746964472 |

An additional actual scikit-learn LBFGS logistic artifact is saved under
`models/sklearn-logistic-reference`, using the same dataset/splits/seed. Its test
accuracy is 0.7076502732 and macro F1 is 0.5740314636; backend differences are
preserved in metadata rather than relabeling these results as Python Adam.

`models/torch-mlp-reference` contains a genuine PyTorch 2.6.0+cpu-trained MLP,
using the same 5,000-event data, chronological splits, seed and architecture,
with 15 training epochs. Its held-out accuracy is 0.7199453552 and macro F1 is
0.5723404255. Its exported numerical artifact is covered by the same Python/C
probability parity gate as the Python and scikit-learn artifacts.

## Strictly held-out trading comparisons

Training a model and replaying its entire source dataset includes training data
in the trading run. Such a run is an integration demonstration, not held-out
performance. The Python API provides an explicit trading evaluation export:

```python
from hftlab.ai import export_holdout

manifest = export_holdout(
    "data/generated/learning-reference.csv",
    "models/torch-mlp-reference/model.hftmodel",
    "data/processed/learning-heldout.csv",
)
```

The function checks that the input SHA256 matches the model's actual training
source. It selects original event indices from `split.test_first_index` inclusive
to `training.events_read` exclusive, which excludes all training, validation and
embargo events. Trailing events needed to form future test labels are included
as ordinary held-out market observations. It verifies the standalone CSV before
atomic publication and writes a provenance manifest with exact source boundaries.

An empty exchange cannot cancel a resting order from an excluded earlier period.
Such CANCEL events are omitted. A MODIFY referencing an excluded prior order is
converted to a new LIMIT with the same new order ID, price and quantity. Both
transformations are counted in the manifest. Native risk, capital, fees and
execution assumptions should then be identical for AI MLP, logistic, momentum,
mean reversion and HOLD comparisons on this exported file.

This trading evaluation uses a cold-start book, portfolio and causal indicator
history. Classification test features were computed with preceding causal
history, so their initial features and scores are not identical to a cold-start
trading run. The distinction and any reference repair remain visible in the
manifest. No earlier training events are silently added as warmup.

An actual same-market, same-capital, same-cost held-out run is preserved in
`reports/heldout-comparison.json`. It uses `data/processed/learning-heldout.csv`,
657 replayable events from source indices 4252..4999 of the 5,000-event reference
dataset. Its manifest reports 91 excluded cancellations of prior-period orders.
The empty-book, cold-start results on this deliberately predictable synthetic
period were:

| Strategy | Executed trades | Net P&L | Fees |
|---|---:|---:|---:|
| PyTorch MLP | 29 | 132.03111 | 2.86889 |
| scikit-learn logistic | 31 | 131.12438 | 3.07562 |
| Momentum | 58 | 94.480419 | 5.529581 |
| Mean reversion | 27 | -133.95823 | 2.65823 |
| HOLD | 0 | 0 | 0 |

All five used initial capital 100,000, one basis-point transaction fees, zero
additional slippage/latency and common position/order limits. These numbers
describe only that saved synthetic experiment and its book initialization.

These are measured classification results from those saved artifacts. Full
confusion matrices, losses, per-class precision/recall, class counts and expected
calibration error are in their metadata. HOLD, training-majority, momentum and
mean-reversion direction classifiers are evaluated on the same held-out samples.
Those comparisons are not a trading backtest; use the native simulator to compare
trading outcomes under common costs and risk settings.

The predictable generator deliberately exposes a learnable sinusoidal synthetic
pattern. Its scores and any simulated profit do not establish real-market
profitability. Multi-layer deep networks, recurrent networks, reinforcement
learning, dynamic feature selection and model calibration fitting are not
implemented.
