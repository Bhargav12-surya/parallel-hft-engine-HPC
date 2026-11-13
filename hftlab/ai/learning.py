"""Small reproducible classifiers with exactly the native C forward pass.

The dependency-free backend is real class-weighted minibatch Adam training.
Optional torch/sklearn backends export the same numerical artifact; no pickle
or arbitrary deserialization enters the native engine.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Callable, Iterable

from hftlab.datasets.market import EVENT_FIELDS, file_hash, iter_events, validate_dataset
from hftlab.utilities import atomic_json as _atomic_json, temporary_path

FEATURE_NAMES = ("return1", "momentum4", "momentum16", "volatility16", "imbalance", "spread")
LABELS = ("SELL", "HOLD", "BUY")


class ModelError(ValueError):
    """An unsafe, incompatible, or invalid model/training input."""


class FeatureState:
    """Six native-compatible causal features; per-instrument bounded history."""

    def __init__(self) -> None:
        self.prices: dict[int, deque] = defaultdict(lambda: deque(maxlen=17))
        self.returns: dict[int, deque] = defaultdict(lambda: deque(maxlen=16))

    def update(self, event: dict) -> tuple[list[float], float]:
        instrument = event["instrument_id"]
        history = self.prices[instrument]
        price = float(event["price_ticks"])
        if event["event_type"] == "CANCEL" or price <= 0:
            price = history[-1] if history else 10000.0
        previous = history[-1] if history else price
        change = (price - previous) / previous
        momentum4 = (price - history[-4]) / history[-4] if len(history) >= 4 else 0.0
        momentum16 = (price - history[-16]) / history[-16] if len(history) >= 16 else 0.0
        returns = self.returns[instrument]
        returns.append(change)
        mean = sum(returns) / len(returns)
        volatility = math.sqrt(sum((value - mean) ** 2 for value in returns) / len(returns))
        quantity = event["quantity"]
        imbalance = (1 if event["side"] == "BUY" else -1) * quantity / (quantity + 100)
        features = [change, momentum4, momentum16, volatility, imbalance, 2.0 / max(price, 1.0)]
        history.append(price)
        return features, price


def extract_features(events: Iterable[dict]) -> list[list[float]]:
    state = FeatureState()
    return [state.update(event)[0] for event in events]


def build_samples(dataset: str | Path, horizon: int = 4, threshold: float = .00015,
                  max_samples: int = 20000) -> dict:
    if type(horizon) is not int or not 1 <= horizon <= 10000:
        raise ModelError("horizon must be in [1, 10000]")
    if not math.isfinite(threshold) or threshold < 0:
        raise ModelError("threshold must be finite and nonnegative")
    if type(max_samples) is not int or not 100 <= max_samples <= 1_000_000:
        raise ModelError("max_samples must be in [100, 1000000]")
    state = FeatureState()
    rows = []
    by_instrument: dict[int, list[int]] = defaultdict(list)
    truncated = False
    for index, event in enumerate(iter_events(dataset)):
        if index >= max_samples:
            truncated = True
            break
        features, price = state.update(event)
        rows.append({"features": features, "price": price, "timestamp_ns": event["timestamp_ns"],
                     "instrument_id": event["instrument_id"], "index": index})
        by_instrument[event["instrument_id"]].append(index)
    for indices in by_instrument.values():
        for position in range(max(0, len(indices) - horizon)):
            current, future = indices[position], indices[position + horizon]
            movement = (rows[future]["price"] - rows[current]["price"]) / rows[current]["price"]
            rows[current]["label"] = 2 if movement > threshold else 0 if movement < -threshold else 1
            rows[current]["target_index"] = future
            rows[current]["target_timestamp_ns"] = rows[future]["timestamp_ns"]
    usable = [row for row in rows if "label" in row]
    if len(usable) < 60:
        raise ModelError("Too few usable samples; generate more events or reduce horizon")
    return {"rows": usable, "events_read": len(rows), "truncated": truncated,
            "horizon": horizon, "threshold": threshold}


def _split(rows: list[dict], train_fraction: float, validation_fraction: float, embargo: int) -> tuple[list, list, list, dict]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ModelError("Train/validation fractions must leave a positive test fraction")
    if type(embargo) is not int or embargo < 0:
        raise ModelError("embargo must be nonnegative")
    first_boundary = rows[int(len(rows) * train_fraction)]["index"]
    second_boundary = rows[int(len(rows) * (train_fraction + validation_fraction))]["index"]
    train = [row for row in rows if row["index"] < first_boundary and row["target_index"] < first_boundary]
    validation = [row for row in rows if row["index"] >= first_boundary + embargo and row["target_index"] < second_boundary]
    test = [row for row in rows if row["index"] >= second_boundary + embargo]
    if min(len(train), len(validation), len(test)) < 10:
        raise ModelError("Chronological purging leaves fewer than 10 samples in a split")
    assert max(row["target_index"] for row in train) < min(row["index"] for row in validation)
    assert max(row["target_index"] for row in validation) < min(row["index"] for row in test)
    metadata = {"method": "chronological-per-instrument-label-purging", "embargo_events": embargo,
                "train_boundary_index": first_boundary, "validation_boundary_index": second_boundary,
                "train_samples": len(train), "validation_samples": len(validation), "test_samples": len(test),
                "train_last_target_index": max(row["target_index"] for row in train),
                "validation_first_index": min(row["index"] for row in validation),
                "validation_last_target_index": max(row["target_index"] for row in validation),
                "test_first_index": min(row["index"] for row in test),
                "scaler_fit": "training-only"}
    return train, validation, test, metadata


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _forward(model: dict, scaled: list[float]) -> tuple[list[float], list[float]]:
    hidden_size = model["hidden_size"]
    if hidden_size:
        hidden = [max(0.0, sum(weight * value for weight, value in zip(row, scaled)) + bias)
                  for row, bias in zip(model["w1"], model["b1"])]
    else:
        hidden = scaled
    logits = [sum(weight * value for weight, value in zip(row, hidden)) + bias
              for row, bias in zip(model["w2"], model["b2"])]
    if not all(math.isfinite(value) for value in logits):
        raise ModelError("Model inference overflowed; inspect features and weight magnitudes")
    return hidden, _softmax(logits)


def predict_proba(model: dict | str | Path, features: list[float] | list[list[float]]) -> list:
    if not isinstance(model, dict):
        model = load_model(model)
    if not features:
        return []
    if isinstance(features[0], (list, tuple)):
        return [predict_proba(model, row) for row in features]
    if len(features) != 6 or not all(math.isfinite(float(value)) for value in features):
        raise ModelError("Prediction requires six finite features")
    scaled = [(float(value) - mean) / scale for value, mean, scale in zip(features, model["mean"], model["scale"])]
    return _forward(model, scaled)[1]


def predict(model: dict | str | Path, features: list[float] | list[list[float]]) -> int | list[int]:
    probabilities = predict_proba(model, features)
    if probabilities and isinstance(probabilities[0], list):
        return [max(range(3), key=row.__getitem__) for row in probabilities]
    return max(range(3), key=probabilities.__getitem__) if probabilities else []


def _metrics(labels: list[int], probabilities: list[list[float]]) -> dict:
    confusion = [[0] * 3 for _ in range(3)]
    for actual, row in zip(labels, probabilities):
        confusion[actual][max(range(3), key=row.__getitem__)] += 1
    count = len(labels)
    per_class = {}
    f1s, weighted_f1 = [], 0.0
    for index, name in enumerate(LABELS):
        tp = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        weighted_f1 += f1 * support / max(count, 1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    loss = -sum(math.log(max(row[label], 1e-15)) for label, row in zip(labels, probabilities)) / max(count, 1)
    # Ten-bin expected calibration error, with explicitly held-out probabilities.
    calibration = 0.0
    for bin_index in range(10):
        members = [(label, row) for label, row in zip(labels, probabilities)
                   if min(9, int(max(row) * 10)) == bin_index]
        if members:
            confidence = sum(max(row) for _, row in members) / len(members)
            accuracy = sum(max(range(3), key=row.__getitem__) == label for label, row in members) / len(members)
            calibration += len(members) / max(count, 1) * abs(confidence - accuracy)
    return {"samples": count, "accuracy": sum(confusion[i][i] for i in range(3)) / max(count, 1),
            "macro_f1": sum(f1s) / 3, "weighted_f1": weighted_f1, "cross_entropy": loss,
            "confusion_matrix": confusion, "per_class": per_class,
            "class_distribution": dict(Counter(LABELS[label] for label in labels)),
            "expected_calibration_error": calibration}


def _initial_model(hidden: int, rng: random.Random) -> dict:
    return {"schema_version": 1, "feature_names": list(FEATURE_NAMES), "labels": list(LABELS),
            "hidden_size": hidden, "mean": [0.0] * 6, "scale": [1.0] * 6,
            "w1": [[rng.gauss(0, math.sqrt(2 / 6)) for _ in range(6)] for _ in range(hidden)],
            "b1": [0.0] * hidden,
            "w2": [[rng.gauss(0, math.sqrt(1 / (hidden or 6))) for _ in range(hidden or 6)] for _ in range(3)],
            "b2": [0.0] * 3}


def _parameter_refs(model: dict) -> list[tuple[list, int]]:
    refs = []
    for name in ("w1", "w2"):
        for row in model[name]:
            refs.extend((row, index) for index in range(len(row)))
    for name in ("b1", "b2"):
        refs.extend((model[name], index) for index in range(len(model[name])))
    return refs


def _python_train(model: dict, train: list, validation: list, epochs: int,
                  learning_rate: float, batch_size: int, seed: int, patience: int,
                  callback: Callable | None) -> tuple[dict, list, int]:
    rng = random.Random(seed)
    refs = _parameter_refs(model)
    first, second = [0.0] * len(refs), [0.0] * len(refs)
    distribution = Counter(label for _, label in train)
    weights = [min(5.0, len(train) / (3 * max(distribution[index], 1))) for index in range(3)]
    best_loss, stale, step = math.inf, 0, 0
    best_model, history = None, []
    for epoch in range(1, epochs + 1):
        order = list(range(len(train)))
        rng.shuffle(order)
        lr = learning_rate * (.5 ** ((epoch - 1) // max(epochs // 3, 1)))
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            gradients = {"w1": [[0.0] * 6 for _ in range(model["hidden_size"])],
                         "w2": [[0.0] * (model["hidden_size"] or 6) for _ in range(3)],
                         "b1": [0.0] * model["hidden_size"], "b2": [0.0] * 3}
            for index in batch:
                features, label = train[index]
                hidden, probabilities = _forward(model, features)
                epoch_loss -= math.log(max(probabilities[label], 1e-15))
                delta = [(probability - int(out == label)) * weights[label] for out, probability in enumerate(probabilities)]
                for out in range(3):
                    gradients["b2"][out] += delta[out]
                    for inp, value in enumerate(hidden):
                        gradients["w2"][out][inp] += delta[out] * value
                for node in range(model["hidden_size"]):
                    error = sum(delta[out] * model["w2"][out][node] for out in range(3)) if hidden[node] > 0 else 0.0
                    gradients["b1"][node] += error
                    for inp, value in enumerate(features):
                        gradients["w1"][node][inp] += error * value
            gradient_refs = _parameter_refs(gradients)
            step += 1
            for position, ((container, index), (gradient_container, gradient_index)) in enumerate(zip(refs, gradient_refs)):
                gradient = max(-10.0, min(10.0, gradient_container[gradient_index] / len(batch)))
                first[position] = .9 * first[position] + .1 * gradient
                second[position] = .999 * second[position] + .001 * gradient * gradient
                corrected_first = first[position] / (1 - .9 ** step)
                corrected_second = second[position] / (1 - .999 ** step)
                container[index] -= lr * corrected_first / (math.sqrt(corrected_second) + 1e-8)
        validation_probabilities = [_forward(model, features)[1] for features, _ in validation]
        metric = _metrics([label for _, label in validation], validation_probabilities)
        record = {"epoch": epoch, "train_loss": epoch_loss / len(train),
                  "validation_loss": metric["cross_entropy"], "validation_accuracy": metric["accuracy"],
                  "validation_macro_f1": metric["macro_f1"], "learning_rate": lr}
        history.append(record)
        if callback:
            callback(record)
        if metric["cross_entropy"] < best_loss - 1e-7:
            best_loss, stale = metric["cross_entropy"], 0
            best_model = json.loads(json.dumps(model))
        else:
            stale += 1
            if stale >= patience:
                break
    return best_model, history, min(history, key=lambda item: item["validation_loss"])["epoch"]


def _torch_train(model: dict, train: list, validation: list, epochs: int,
                 learning_rate: float, batch_size: int, seed: int, patience: int,
                 callback: Callable | None, device: str) -> tuple[dict, list, int]:
    import torch
    torch.manual_seed(seed)
    # Tiny native-compatible networks do not benefit from dozens of BLAS
    # workers, and a bounded team avoids oversubscribing simulation jobs.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if device == "cuda" and not torch.cuda.is_available():
        raise ModelError("CUDA training requested but PyTorch cannot use a CUDA device")
    if device not in ("cpu", "cuda"):
        raise ModelError("device must be cpu or cuda")
    torch.use_deterministic_algorithms(True)
    if model["hidden_size"]:
        network = torch.nn.Sequential(torch.nn.Linear(6, model["hidden_size"]), torch.nn.ReLU(), torch.nn.Linear(model["hidden_size"], 3))
    else:
        network = torch.nn.Sequential(torch.nn.Linear(6, 3))
    network.to(device)
    x = torch.tensor([features for features, _ in train], dtype=torch.float64, device=device)
    y = torch.tensor([label for _, label in train], dtype=torch.long, device=device)
    xv = torch.tensor([features for features, _ in validation], dtype=torch.float64, device=device)
    network.double()
    distribution = Counter(label for _, label in train)
    weights = torch.tensor([min(5.0, len(train) / (3 * max(distribution[index], 1))) for index in range(3)], dtype=torch.float64, device=device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=.5)
    best_loss, stale = math.inf, 0
    best_state, history = None, []
    for epoch in range(1, epochs + 1):
        network.train()
        order = torch.randperm(len(train), device=device)
        total_loss = 0.0
        for start in range(0, len(train), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_function(network(x[indices]), y[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 10)
            optimizer.step()
            total_loss += loss.item() * len(indices)
        network.eval()
        with torch.no_grad():
            probabilities = torch.softmax(network(xv), dim=1).cpu().tolist()
        metric = _metrics([label for _, label in validation], probabilities)
        record = {"epoch": epoch, "train_loss": total_loss / len(train),
                  "validation_loss": metric["cross_entropy"], "validation_accuracy": metric["accuracy"],
                  "validation_macro_f1": metric["macro_f1"], "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        if callback:
            callback(record)
        if metric["cross_entropy"] < best_loss - 1e-7:
            best_loss, stale = metric["cross_entropy"], 0
            best_state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
        else:
            stale += 1
        scheduler.step()
        if stale >= patience:
            break
    network.load_state_dict(best_state)
    layers = [layer for layer in network if isinstance(layer, torch.nn.Linear)]
    if model["hidden_size"]:
        model["w1"], model["b1"] = layers[0].weight.detach().cpu().tolist(), layers[0].bias.detach().cpu().tolist()
    model["w2"], model["b2"] = layers[-1].weight.detach().cpu().tolist(), layers[-1].bias.detach().cpu().tolist()
    return model, history, min(history, key=lambda item: item["validation_loss"])["epoch"]


def _sklearn_train(model: dict, train: list, validation: list, epochs: int, seed: int) -> tuple[dict, list, int]:
    from sklearn.linear_model import LogisticRegression
    estimator = LogisticRegression(max_iter=max(100, epochs * 10), class_weight="balanced", random_state=seed, solver="lbfgs")
    estimator.fit([features for features, _ in train], [label for _, label in train])
    if list(estimator.classes_) != [0, 1, 2]:
        raise ModelError("sklearn training requires SELL, HOLD, and BUY examples; change threshold")
    model["w2"], model["b2"] = estimator.coef_.tolist(), estimator.intercept_.tolist()
    metric = _metrics([label for _, label in validation], [_forward(model, features)[1] for features, _ in validation])
    history = [{"epoch": int(estimator.n_iter_[0]), "validation_loss": metric["cross_entropy"],
                "validation_accuracy": metric["accuracy"], "validation_macro_f1": metric["macro_f1"],
                "optimizer": "lbfgs"}]
    return model, history, history[0]["epoch"]


def train_model(dataset: str | Path, output: str | Path, kind: str = "mlp", epochs: int = 30,
                seed: int = 42, hidden_size: int = 12, horizon: int = 4,
                threshold: float = .00015, learning_rate: float = .01, batch_size: int = 64,
                train_fraction: float = .7, validation_fraction: float = .15, embargo: int = 16,
                patience: int = 8, max_samples: int = 20000, backend: str = "auto",
                device: str = "cpu", callback: Callable | None = None, **kwargs) -> dict:
    if kwargs:
        raise ModelError(f"Unknown training parameters: {', '.join(kwargs)}")
    if kind not in ("mlp", "logistic"):
        raise ModelError("kind must be mlp or logistic")
    if type(seed) is not int:
        raise ModelError("seed must be an integer")
    if type(embargo) is not int or embargo < 0:
        raise ModelError("embargo must be a nonnegative integer")
    if type(epochs) is not int or not 1 <= epochs <= 10000:
        raise ModelError("epochs must be in [1, 10000]")
    if type(hidden_size) is not int or not 1 <= hidden_size <= 128:
        raise ModelError("hidden_size must be in [1, 128]")
    if not 0 < learning_rate <= 1 or not math.isfinite(learning_rate):
        raise ModelError("learning_rate must be finite in (0,1]")
    if type(batch_size) is not int or not 1 <= batch_size <= 10000 or type(patience) is not int or patience < 1:
        raise ModelError("batch_size must be in [1,10000]; patience positive")
    if backend not in ("auto", "python", "torch", "sklearn"):
        raise ModelError("backend must be auto, python, torch, or sklearn")
    if device not in ("cpu", "cuda"):
        raise ModelError("device must be cpu or cuda")
    if backend == "auto":
        preferred = "sklearn" if kind == "logistic" else "torch"
        backend = preferred if importlib.util.find_spec(preferred if preferred != "sklearn" else "sklearn") else "python"
    if device == "cuda" and backend != "torch":
        raise ModelError("CUDA training requires the torch backend")
    if backend == "sklearn" and kind != "logistic":
        raise ModelError("sklearn backend is available for logistic regression")
    if backend in ("torch", "sklearn") and importlib.util.find_spec(backend) is None:
        raise ModelError(f"Optional {backend} backend is not installed")
    data_info = validate_dataset(dataset)
    samples = build_samples(dataset, horizon, threshold, max_samples)
    training, validation, testing, split_metadata = _split(samples["rows"], train_fraction, validation_fraction, max(embargo, horizon))
    model = _initial_model(hidden_size if kind == "mlp" else 0, random.Random(seed))
    model["mean"] = [sum(row["features"][column] for row in training) / len(training) for column in range(6)]
    model["scale"] = [max(1e-8, math.sqrt(sum((row["features"][column] - model["mean"][column]) ** 2 for row in training) / len(training))) for column in range(6)]
    def transform(rows):
        return [([(value - mean) / scale for value, mean, scale in zip(row["features"], model["mean"], model["scale"])], row["label"]) for row in rows]
    train_values, validation_values, test_values = map(transform, (training, validation, testing))
    if backend == "torch":
        model, history, best_epoch = _torch_train(model, train_values, validation_values, epochs, learning_rate, batch_size, seed, patience, callback, device)
    elif backend == "sklearn":
        model, history, best_epoch = _sklearn_train(model, train_values, validation_values, epochs, seed)
        if callback:
            callback(history[0])
    else:
        model, history, best_epoch = _python_train(model, train_values, validation_values, epochs, learning_rate, batch_size, seed, patience, callback)
    destination = Path(output).resolve()
    if destination.suffix.lower() in (".hftmodel", ".model", ".txt"):
        native_path, folder = destination, destination.parent
        metadata_path = destination.with_suffix(destination.suffix + ".json")
        history_path = destination.with_suffix(destination.suffix + ".history.json")
    else:
        folder = destination
        native_path, metadata_path, history_path = folder / "model.hftmodel", folder / "metadata.json", folder / "history.json"
    folder.mkdir(parents=True, exist_ok=True)
    _export_model(model, native_path)
    validation_metrics = _metrics([label for _, label in validation_values], [_forward(model, features)[1] for features, _ in validation_values])
    test_metrics = _metrics([label for _, label in test_values], [_forward(model, features)[1] for features, _ in test_values])
    majority = Counter(label for _, label in train_values).most_common(1)[0][0]
    baselines = {}
    for name, labels in {
        "hold": [1] * len(testing),
        "training-majority": [majority] * len(testing),
        "momentum": [2 if row["features"][2] > threshold else 0 if row["features"][2] < -threshold else 1 for row in testing],
        "mean-reversion": [0 if row["features"][2] > threshold else 2 if row["features"][2] < -threshold else 1 for row in testing],
    }.items():
        baselines[name] = _metrics([row["label"] for row in testing], [[float(index == label) for index in range(3)] for label in labels])
    versions = {"python": sys.version.split()[0]}
    if backend in ("torch", "sklearn"):
        module = __import__(backend)
        versions[backend] = module.__version__
    metadata = {"schema_version": 1, "export_format": "HFTMODEL1", "kind": kind,
                "architecture": [6, hidden_size, 3] if kind == "mlp" else [6, 3],
                "feature_names": list(FEATURE_NAMES), "labels": list(LABELS),
                "class_mapping": {str(index): label for index, label in enumerate(LABELS)},
                "scaler": {"mean": model["mean"], "scale": model["scale"]},
                "dataset": str(Path(dataset).resolve()), "dataset_sha256": data_info["sha256"],
                "native_model": str(native_path), "sha256": file_hash(native_path),
                "timestamp": datetime.now(timezone.utc).isoformat(), "seed": seed,
                "backend": backend, "device": device, "versions": versions,
                "split": split_metadata, "validation": validation_metrics, "test": test_metrics,
                "baselines": baselines, "best_epoch": best_epoch, "epochs_completed": len(history),
                "training": {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                             "horizon": horizon, "threshold": threshold, "max_samples": max_samples,
                             "events_read": samples["events_read"], "truncated": samples["truncated"],
                             "optimizer": "LBFGS" if backend == "sklearn" else "Adam",
                             "class_weight": "balanced" if backend == "sklearn" else "balanced-capped-5"},
                "native_inference": "one-hidden-ReLU-softmax" if kind == "mlp" else "softmax-linear",
                "limitations": ["Synthetic directional classification does not establish trading profitability.",
                                "Baselines are direction classifiers; execution costs and profitability require simulation."]}
    _atomic_json(metadata_path, metadata)
    _atomic_json(history_path, history)
    # Export an inference fixture independent of training history for C/CUDA parity.
    fixture = {"schema_version": 1, "model_sha256": metadata["sha256"],
               "features": [row["features"] for row in testing[:32]],
               "probabilities": [predict_proba(model, row["features"]) for row in testing[:32]],
               "labels": [row["label"] for row in testing[:32]]}
    _atomic_json(native_path.with_suffix(native_path.suffix + ".fixture.json"), fixture)
    return metadata


def export_holdout(dataset: str | Path, model_path: str | Path, output: str | Path) -> dict:
    """Export the model's strictly held-out chronological event interval.

    A new exchange starts empty. Cancels referring to pre-holdout orders are
    excluded; replacements referencing them become ordinary new LIMIT orders.
    Every transformation is counted, and neither training nor embargo events are
    replayed. Classification features may have prior causal history, whereas this
    standalone execution benchmark intentionally uses a cold-start indicator/book.
    """
    source, destination = Path(dataset).resolve(), Path(output).resolve()
    if source == destination:
        raise ModelError("Holdout output must differ from the source dataset")
    if destination.suffix.lower() != ".csv":
        raise ModelError("Native holdout export requires a .csv destination")
    model = load_model(model_path)
    metadata = model.get("metadata")
    if not isinstance(metadata, dict):
        raise ModelError("Held-out export requires training metadata beside the model")
    if file_hash(source) != metadata.get("dataset_sha256"):
        raise ModelError("Holdout dataset SHA256 differs from the model training dataset")
    try:
        start = metadata["split"]["test_first_index"]
        stop = metadata["training"]["events_read"]
    except KeyError as exc:
        raise ModelError("Model metadata has no chronological held-out boundaries") from exc
    if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start < stop <= 1_000_000:
        raise ModelError("Invalid held-out chronological boundaries in model metadata")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(destination)
    known_limits = set()
    dropped_cancels = converted_replacements = selected_events = written_events = 0
    first_source_index = last_source_index = None
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS, lineterminator="\n")
            writer.writeheader()
            for index, source_event in enumerate(iter_events(source)):
                if index >= stop:
                    break
                if index < start:
                    continue
                selected_events += 1
                event = dict(source_event)
                kind = event["event_type"]
                reference = event["reference_order_id"]
                if kind in ("CANCEL", "MODIFY") and reference not in known_limits:
                    if kind == "CANCEL":
                        dropped_cancels += 1
                        continue
                    event["event_type"], event["reference_order_id"] = "LIMIT", 0
                    converted_replacements += 1
                if event["event_type"] in ("LIMIT", "MODIFY"):
                    known_limits.add(event["order_id"])
                writer.writerow(event)
                written_events += 1
                first_source_index = index if first_source_index is None else first_source_index
                last_source_index = index
        if not written_events:
            raise ModelError("Held-out interval contains no replayable events")
        # Validate before replacing an existing export; an invalid managed file
        # never becomes the visible final output.
        validate_dataset(temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    manifest = validate_dataset(destination)
    manifest.update(source=str(source), source_sha256=metadata["dataset_sha256"],
                    model_sha256=file_hash(model["path"]), purpose="held-out-trading-evaluation",
                    source_start_index_inclusive=start, source_stop_index_exclusive=stop,
                    first_written_source_index=first_source_index, last_written_source_index=last_source_index,
                    selected_source_events=selected_events, dropped_prior_order_cancels=dropped_cancels,
                    converted_prior_order_modifications=converted_replacements,
                    training_events_replayed=0, embargo_events_replayed=0,
                    initial_exchange_state="empty", initial_feature_state="cold-start",
                    note="Strictly held-out chronology. Earlier-order cancels are omitted; earlier-order replacements become new limits. Cold-start features can differ from classification test features with causal prehistory.")
    _atomic_json(Path(str(destination) + ".manifest.json"), manifest)
    return manifest


def _export_model(model: dict, destination: Path) -> None:
    values = ["HFTMODEL 1", f"6 {model['hidden_size']} 3"]
    for row in (model["mean"], model["scale"]):
        values.append(" ".join(format(value, ".17g") for value in row))
    if model["hidden_size"]:
        values.extend(" ".join(format(value, ".17g") for value in row) for row in model["w1"])
        values.append(" ".join(format(value, ".17g") for value in model["b1"]))
    values.extend(" ".join(format(value, ".17g") for value in row) for row in model["w2"])
    values.append(" ".join(format(value, ".17g") for value in model["b2"]))
    temporary = temporary_path(destination)
    try:
        temporary.write_text("\n".join(values) + "\n", encoding="ascii")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(path: str | Path, verify_checksum: bool = True) -> dict:
    source = Path(path).resolve()
    if source.is_dir():
        source = source / "model.hftmodel"
    if not source.is_file():
        raise ModelError(f"Model file not found: {source}")
    if source.stat().st_size > 2 * 1024 * 1024:
        raise ModelError("Model file exceeds the 2 MiB safety limit")
    metadata_paths = [source.with_suffix(source.suffix + ".json")]
    if source.name == "model.hftmodel":
        metadata_paths.append(source.parent / "metadata.json")
    metadata = None
    for candidate in metadata_paths:
        if candidate.exists():
            try:
                metadata = json.loads(candidate.read_text(encoding="utf-8"))
            except (UnicodeError, ValueError) as error:
                raise ModelError(f"Invalid model metadata: {error}") from error
            if not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
                raise ModelError("Invalid model metadata schema")
            if metadata.get("feature_names") != list(FEATURE_NAMES) or metadata.get("labels") != list(LABELS):
                raise ModelError("Model metadata features/classes are incompatible")
            break
    if verify_checksum and metadata is not None and file_hash(source) != metadata.get("sha256"):
        raise ModelError("Model checksum does not match metadata")
    try:
        tokens = source.read_text(encoding="ascii").split()
        if tokens[:2] != ["HFTMODEL", "1"] or len(tokens) < 5:
            raise ModelError("Unsupported model header/version")
        inputs, hidden, outputs = map(int, tokens[2:5])
        if inputs != 6 or outputs != 3 or not 0 <= hidden <= 128:
            raise ModelError("Model must be 6 inputs, 0..128 hidden units, 3 outputs")
        expected = 5 + 12 + (hidden * 6 + hidden if hidden else 0) + 3 * (hidden or 6) + 3
        if len(tokens) != expected:
            raise ModelError(f"Model numerical length mismatch: expected {expected}, got {len(tokens)}")
        numerical = [float(token) for token in tokens[5:]]
        if not all(math.isfinite(value) and abs(value) <= 1e15 for value in numerical):
            raise ModelError("Model contains nonfinite or excessively large values")
        position = 0
        def take(count):
            nonlocal position
            result = numerical[position:position + count]
            position += count
            return result
        model = {"schema_version": 1, "hidden_size": hidden, "feature_names": list(FEATURE_NAMES),
                 "labels": list(LABELS), "mean": take(6), "scale": take(6)}
        if any(value < 1e-15 for value in model["scale"]):
            raise ModelError("Model scaler standard deviations must be at least 1e-15")
        model["w1"] = [take(6) for _ in range(hidden)]
        model["b1"] = take(hidden)
        model["w2"] = [take(hidden or 6) for _ in range(3)]
        model["b2"] = take(3)
        model["metadata"] = metadata
        model["path"] = str(source)
        return model
    except (UnicodeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ModelError):
            raise
        raise ModelError(f"Malformed native model: {exc}") from exc


def validate_model(path: str | Path) -> dict:
    model = load_model(path)
    return {"schema_version": 1, "valid": True, "path": model["path"],
            "sha256": file_hash(model["path"]), "hidden_size": model["hidden_size"],
            "feature_names": list(FEATURE_NAMES), "labels": list(LABELS)}


def evaluate_model(model: dict | str | Path, dataset: str | Path, **kwargs) -> dict:
    if not isinstance(model, dict):
        model = load_model(model)
    training = (model.get("metadata") or {}).get("training", {})
    samples = build_samples(dataset, kwargs.get("horizon", training.get("horizon", 4)),
                            kwargs.get("threshold", training.get("threshold", .00015)),
                            kwargs.get("max_samples", 20000))
    return _metrics([row["label"] for row in samples["rows"]],
                    [predict_proba(model, row["features"]) for row in samples["rows"]])


def list_models(root: str | Path = "models") -> list[dict]:
    directory = Path(root)
    records = []
    for path in sorted(directory.rglob("*.hftmodel")) if directory.exists() else []:
        try:
            model = load_model(path)
            metadata = model.get("metadata") or {}
            records.append({"path": str(path.resolve()), "name": path.parent.name,
                            "kind": metadata.get("kind", "mlp" if model["hidden_size"] else "logistic"),
                            "timestamp": metadata.get("timestamp"), "test": metadata.get("test", {}),
                            "sha256": file_hash(path), "status": "AVAILABLE"})
        except (ModelError, OSError, json.JSONDecodeError) as exc:
            records.append({"path": str(path.resolve()), "status": "INCOMPATIBLE", "error": str(exc)})
    return records
