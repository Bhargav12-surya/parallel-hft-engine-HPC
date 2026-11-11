# Reproducible offline training

The implementation lives in `hftlab.ai.learning`; training does not require a
separate service. `python -m hftlab train --help` lists supported options. Both
class-weighted logistic regression and a one-hidden-layer ReLU MLP produce the
same native HFTMODEL1 format. Python's dependency-free backend trains genuine
minibatch Adam models; optional PyTorch supports CPU/CUDA MLP training and
scikit-learn supports logistic regression.

Six causal per-instrument features are computed before future labels are formed.
Splits are chronological and purge labels whose horizons cross a split boundary.
A configurable embargo separates later splits. Means and scales use only the
training partition. Test classification and calibration metrics, training history,
dataset SHA256, architecture, seed, versions and a native inference fixture are
saved beside each exported model. Training is bounded by `max_samples`, and any
truncation is recorded explicitly. Evaluation direction baselines do not include
execution costs; use native simulation for trading outcomes.

The `predictable` synthetic scenario is an educational learning demonstration,
not evidence of live trading profitability. No brokerage connectivity exists.
