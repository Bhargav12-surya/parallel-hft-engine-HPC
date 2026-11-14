"""Real cross-language numerical parity and deterministic AI replay.

These are hardware gates: absence of a compiled backend skips its execution,
while an available backend is required to execute correctly and match the oracle.
"""
from dataclasses import replace
import importlib.util
import json
from itertools import combinations
from pathlib import Path
import subprocess

import pytest

from hftlab.ai import export_holdout, load_model, predict_proba, train_model
from hftlab.configuration import SimulationConfig
from hftlab.datasets import generate_dataset
from hftlab.engines import capabilities, engine_available, find_runner, mpi_launcher
from hftlab.experiments import ExperimentRegistry
from hftlab.orchestration import JobManager

pytestmark = pytest.mark.native


@pytest.fixture(scope="module")
def native_runner():
    try:
        return find_runner()
    except RuntimeError as exc:
        pytest.skip(f"Native compilation gate not satisfied: {exc}")


@pytest.fixture(scope="module")
def trained_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("native-model-parity")
    dataset = root / "market.csv"
    generate_dataset(dataset, events=1200, instruments=3, scenario="predictable", seed=28)
    artifacts = {}
    for kind in ("logistic", "mlp"):
        metadata = train_model(dataset, root / kind, kind=kind, epochs=5, seed=9, backend="python")
        artifacts[kind] = Path(metadata["native_model"])
    if importlib.util.find_spec("sklearn"):
        metadata = train_model(dataset, root / "sklearn", kind="logistic", epochs=5, seed=9, backend="sklearn")
        artifacts["sklearn"] = Path(metadata["native_model"])
    if importlib.util.find_spec("torch"):
        metadata = train_model(dataset, root / "torch", kind="mlp", epochs=5, seed=9, backend="torch", device="cpu")
        artifacts["torch"] = Path(metadata["native_model"])
    return dataset, artifacts


@pytest.mark.parametrize("kind", ["logistic", "mlp", "sklearn", "torch"])
def test_python_and_native_inference_weights_and_scaler_match(native_runner, trained_models, tmp_path, kind):
    _, artifacts = trained_models
    if kind not in artifacts:
        pytest.skip(f"Optional {kind} training backend is unavailable")
    model_path = artifacts[kind]
    fixture = json.loads(Path(str(model_path) + ".fixture.json").read_text(encoding="utf-8"))
    # Include held-out real features and explicit edge cases to exercise scaler,
    # negative ReLU activation, softmax stability, and output-class ordering.
    features = fixture["features"] + [[0.0] * 6, [.001, -.002, .003, .0005, -.8, .0002],
                                      [-.002, .001, -.003, .001, .9, .0001]]
    feature_path = tmp_path / "features.txt"
    feature_path.write_text("\n".join(" ".join(format(value, ".17g") for value in row) for row in features) + "\n", encoding="ascii")
    completed = subprocess.run([str(native_runner), "--infer", "--model", str(model_path),
                                "--features", str(feature_path)], capture_output=True, text=True,
                               timeout=30, check=True)
    actual = json.loads(completed.stdout)["probabilities"]
    expected = predict_proba(load_model(model_path), features)
    assert len(actual) == len(expected)
    for native, python in zip(actual, expected):
        assert native == pytest.approx(python, abs=2e-11, rel=2e-11)
        assert sum(native) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("engine,cpu_mode,components", [
    ("sequential", "single-single", ["cpu", "openmp"]),
    ("threaded", "single-multi", ["cpu", "openmp"]),
    ("sequential", "multi-single", ["cpu", "openmp"]),
    ("threaded", "multi-multi", ["cpu", "openmp"]),
    ("openmp", "multi-multi", ["cpu", "openmp"]),
    ("mpi", "multi-multi", ["cpu", "mpi"]),
    ("cuda", "multi-multi", ["cpu", "cuda"]),
] + [("hybrid", "multi-multi", list(parts)) for count in (2, 3, 4)
     for parts in combinations(("cpu", "openmp", "mpi", "cuda"), count)])
@pytest.mark.parametrize("kind", ["logistic", "mlp"])
def test_ai_replay_matches_sequential_oracle(native_runner, trained_models, tmp_path, engine, cpu_mode, components, kind):
    capability = capabilities()
    if not engine_available(capability, engine):
        pytest.skip(f"{engine} hardware/build execution gate is unavailable")
    if (engine == "mpi" or engine == "hybrid" and "mpi" in components) and not mpi_launcher():
        pytest.skip("MPI launcher is unavailable")
    if engine == "hybrid":
        for part in components:
            if not engine_available(capability, "sequential" if part == "cpu" else part):
                pytest.skip(f"Hybrid component {part} is unavailable")
    dataset, artifacts = trained_models
    base = SimulationConfig(dataset=dataset, strategy="ai", model=artifacts[kind], threads=2,
                            output_dir=tmp_path / "results")
    registry = ExperimentRegistry(tmp_path / "registry")
    expected = JobManager(registry).run(base)
    actual = JobManager(registry).run(replace(base, engine=engine, cpu_mode=cpu_mode, components=components))
    assert actual["events_processed"] == expected["events_processed"] == 1200
    assert actual["checksum"] == expected["checksum"]
    assert actual["trades"] == expected["trades"]
    assert actual["orders"] == expected["orders"]
    assert actual["pnl"] == pytest.approx(expected["pnl"], abs=1e-9)
    assert actual["equity"] == pytest.approx(expected["equity"], abs=1e-9)
    assert actual["fees"] == pytest.approx(expected["fees"], abs=1e-10)


def test_heldout_trading_replays_only_test_market_interval(native_runner, trained_models, tmp_path):
    dataset, artifacts = trained_models
    model = artifacts.get("torch", artifacts["mlp"])
    heldout = tmp_path / "heldout.csv"
    provenance = export_holdout(dataset, model, heldout)
    assert provenance["training_events_replayed"] == provenance["embargo_events_replayed"] == 0
    assert provenance["source_start_index_inclusive"] == load_model(model)["metadata"]["split"]["test_first_index"]
    registry = ExperimentRegistry(tmp_path / "registry")
    base = SimulationConfig(dataset=heldout, output_dir=tmp_path / "results")
    results = [JobManager(registry).run(replace(base, strategy=strategy, model=weights))
               for strategy, weights in (("ai", model), ("ai", artifacts["logistic"]),
                                         ("momentum", None), ("mean-reversion", None), ("hold", None))]
    assert all(row["events_processed"] == provenance["events"] for row in results)
    assert all(row["events_processed"] < 1200 for row in results)
    assert results[-1]["trades"] == results[-1]["fees"] == results[-1]["pnl"] == 0
