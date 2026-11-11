import json
import math

import pytest

from hftlab.ai import (ModelError, build_samples, extract_features, load_model,
                       export_holdout, predict, predict_proba, train_model, validate_model)
from hftlab.ai.learning import FeatureState, _initial_model, _python_train, _split
from hftlab.datasets import generate_dataset, iter_events, validate_dataset


def test_features_are_causal_and_instrument_isolated():
    def row(price, instrument=0, quantity=100, side="BUY", kind="QUOTE"):
        return {"price_ticks": price, "instrument_id": instrument,
                "quantity": quantity, "side": side, "event_type": kind}
    events = [row(100), row(110), row(120), row(130), row(140), row(900,1)]
    features = extract_features(events)
    assert features[0] == pytest.approx([0,0,0,0,.5,.02])
    assert features[1][0] == pytest.approx(.1)
    assert features[4][1] == pytest.approx(.4)
    assert features[5][0:4] == pytest.approx([0,0,0,0])
    prefix = extract_features(events[:4])
    assert prefix == features[:4]
    state = FeatureState()
    state.update(row(200))
    cancelled, price = state.update(row(0, quantity=0, kind="CANCEL"))
    assert price == 200 and cancelled[0] == 0 and cancelled[4] == 0


def test_chronological_purging_prevents_future_label_overlap(tmp_path):
    path = tmp_path / "market.csv"
    generate_dataset(path, events=1000, instruments=3, scenario="correlated")
    samples = build_samples(path, horizon=8)
    train, validation, test, metadata = _split(samples["rows"], .7, .15, 16)
    assert max(row["target_index"] for row in train) < min(row["index"] for row in validation)
    assert max(row["target_index"] for row in validation) < min(row["index"] for row in test)
    assert all(row["target_timestamp_ns"] >= row["timestamp_ns"] for row in samples["rows"])
    assert metadata["scaler_fit"] == "training-only"


@pytest.mark.parametrize("kind", ["logistic", "mlp"])
def test_training_exports_actual_weights_with_prediction_roundtrip(tmp_path, kind):
    path = tmp_path / "market.csv"
    generate_dataset(path, events=1000, instruments=2, scenario="predictable", seed=10)
    metadata = train_model(path, tmp_path / kind, kind=kind, epochs=4, backend="python", seed=2)
    assert metadata["backend"] == "python"
    assert metadata["test"]["samples"] > 0
    assert metadata["epochs_completed"] == 4
    model = load_model(metadata["native_model"])
    fixture = json.loads(open(metadata["native_model"] + ".fixture.json", encoding="utf-8").read())
    probabilities = predict_proba(model, fixture["features"])
    for actual, expected in zip(probabilities, fixture["probabilities"]):
        assert actual == pytest.approx(expected, abs=1e-13)
        assert sum(actual) == pytest.approx(1.0)
    assert validate_model(metadata["native_model"])["valid"]
    assert metadata["split"]["train_last_target_index"] < metadata["split"]["validation_first_index"]
    assert len(predict(model, fixture["features"])) == len(fixture["features"])


def test_train_only_scaler_and_seed_reproducibility(tmp_path):
    path = tmp_path / "market.csv"
    generate_dataset(path, events=600, instruments=2, scenario="normal", seed=10)
    samples = build_samples(path)
    train, _, _, _ = _split(samples["rows"], .7, .15, 16)
    first = train_model(path, tmp_path / "first", epochs=2, backend="python", seed=7)
    second = train_model(path, tmp_path / "second", epochs=2, backend="python", seed=7)
    expected = [sum(row["features"][index] for row in train) / len(train) for index in range(6)]
    assert first["scaler"]["mean"] == pytest.approx(expected)
    assert first["sha256"] == second["sha256"]


def test_actual_optimizer_learns_separable_three_class_problem():
    import random
    model = _initial_model(6, random.Random(4))
    train = [([float(label - 1),float(label == 1),0,0,0,0], label)
             for label in range(3) for _ in range(30)]
    trained, history, _ = _python_train(model, train, train, 20, .03, 15, 42, 20, None)
    assert history[-1]["validation_loss"] < history[0]["validation_loss"] * .3
    assert history[-1]["validation_accuracy"] > .95


def test_model_checksum_and_length_failures(tmp_path):
    path = tmp_path / "market.csv"
    generate_dataset(path, events=500, instruments=2)
    train_model(path, tmp_path / "weights.hftmodel", epochs=1, backend="python")
    model_path = tmp_path / "weights.hftmodel"
    model_path.write_text(model_path.read_text() + " 1")
    with pytest.raises(ModelError, match="checksum"):
        load_model(model_path)
    with pytest.raises(ModelError, match="length mismatch"):
        load_model(model_path, verify_checksum=False)
    model_path.write_text("HFTMODEL 1\n1000000000 1000000000 3")
    with pytest.raises(ModelError, match="6 inputs"):
        load_model(model_path, verify_checksum=False)


def test_invalid_training_parameters_and_missing_models(tmp_path):
    with pytest.raises(ModelError, match="not found"):
        load_model(tmp_path / "missing.hftmodel")
    with pytest.raises(ModelError, match="epochs"):
        train_model("missing.csv", tmp_path, epochs=0)
    with pytest.raises(ModelError, match="CUDA"):
        train_model("missing.csv", tmp_path, backend="python", device="cuda")
    with pytest.raises(ModelError, match="six finite"):
        predict_proba(_initial_model(0, __import__("random").Random(1)), [math.nan] * 6)


def test_export_holdout_contains_only_test_events_with_standalone_references(tmp_path):
    source = tmp_path / "market.csv"
    generate_dataset(source, events=1000, instruments=2, seed=8)
    metadata = train_model(source, tmp_path / "weights", epochs=2, backend="python")
    destination = tmp_path / "held-out.csv"
    manifest = export_holdout(source, metadata["native_model"], destination)
    events = list(iter_events(destination))
    assert validate_dataset(destination)["events"] == len(events) > 50
    assert min(row["event_id"] for row in events) >= metadata["split"]["test_first_index"] + 1
    assert max(row["event_id"] for row in events) <= metadata["training"]["events_read"]
    assert manifest["training_events_replayed"] == 0
    assert manifest["embargo_events_replayed"] == 0
    assert manifest["selected_source_events"] == manifest["events"] + manifest["dropped_prior_order_cancels"]
    assert manifest["initial_feature_state"] == "cold-start"
    with pytest.raises(ModelError, match="must differ"):
        export_holdout(source, metadata["native_model"], source)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ModelError, match="SHA256"):
        export_holdout(source, metadata["native_model"], destination)
