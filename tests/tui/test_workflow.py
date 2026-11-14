"""Exercise the application widgets and real data/training/native workers."""
import asyncio
import json
import sys

import pytest
from textual.widgets import Checkbox, ContentSwitcher, DataTable, Input, Select, SelectionList, TabbedContent

from hftlab.engines import capabilities, engine_available
from hftlab.experiments import ExperimentRegistry
from hftlab.tui.app import HFTLabApp, PAGES, STRATEGIES


@pytest.mark.parametrize("preferences", [[], {"strategy": [], "engine": {}}, {"compact": True}])
def test_preferences_restore_without_crashing_or_losing_compact_mode(monkeypatch, tmp_path, preferences):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(HFTLabApp, "run_diagnostics", lambda self: None)
    (tmp_path / ".hftlab-preferences.json").write_text(json.dumps(preferences), encoding="utf-8")

    async def exercise():
        app = HFTLabApp()
        async with app.run_test(size=(130, 45)) as pilot:
            await pilot.pause()
            compact = isinstance(preferences, dict) and preferences.get("compact") is True
            assert app.has_class("compact") == compact
            assert app.query_one("#compact-mode", Checkbox).value == compact
            app.save_preferences()
            assert json.loads((tmp_path / ".hftlab-preferences.json").read_text())["compact"] == compact

    asyncio.run(exercise())


def test_failed_experiment_opens_through_history_selection(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(HFTLabApp, "run_diagnostics", lambda self: None)
    ExperimentRegistry().save({"id": "failed_run", "timestamp": "2026-11-14",
                               "config": {"engine": "sequential", "strategy": "momentum"},
                               "status": "FAILED", "result": None, "error": "Native execution failed"})

    async def exercise():
        app = HFTLabApp()
        async with app.run_test(size=(130, 45)) as pilot:
            await pilot.pause()
            await pilot.click("#nav-history")
            app.query_one("#history-table", DataTable).focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#pages", ContentSwitcher).current == "results"
            assert app.result["status"] == "FAILED"
            assert app.result["error"] == "Native execution failed"

    asyncio.run(exercise())


def test_corrupt_history_database_does_not_crash_application(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(HFTLabApp, "run_diagnostics", lambda self: None)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments/registry.sqlite3").write_bytes(b"corrupt database")

    async def exercise():
        app = HFTLabApp()
        async with app.run_test(size=(130, 45)) as pilot:
            await pilot.pause()
            await pilot.click("#nav-history")
            assert app.query_one("#history-table", DataTable).row_count == 0
            assert app.query_one("#pages", ContentSwitcher).current == "history"

    asyncio.run(exercise())


def test_application_cancel_confirmation_reaps_native_job(monkeypatch, tmp_path):
    from hftlab import orchestration
    from hftlab.datasets import generate_dataset
    from hftlab.tui.app import ConfirmScreen

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(HFTLabApp, "run_diagnostics", lambda self: None)
    dataset = tmp_path / "data/generated/cancel.csv"
    generate_dataset(dataset, events=100, instruments=2)
    stub = tmp_path / "slow_backend.py"
    stub.write_text("import time\nprint('{\"schema_version\":1,\"type\":\"progress\",\"events_processed\":1}',flush=True)\ntime.sleep(30)\n")
    monkeypatch.setattr(orchestration, "validate_engine", lambda config: {})
    monkeypatch.setattr(orchestration, "find_runner", lambda: stub)
    monkeypatch.setattr(orchestration, "command", lambda config, output: [sys.executable, str(stub)])

    async def exercise():
        app = HFTLabApp()
        async with app.run_test(size=(130, 45)) as pilot:
            app.query_one("#dataset-path", Input).value = str(dataset)
            await pilot.press("f5")
            for _ in range(100):
                await pilot.pause(.05)
                if app.manager is not None and app.manager.state == "RUNNING":
                    break
            assert app.busy and app.manager.state == "RUNNING"
            await pilot.press("ctrl+c")
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("escape")
            assert app.busy
            await pilot.press("ctrl+c")
            await pilot.click("#confirm-yes")
            await pilot.pause()
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=15)
            await pilot.pause()
            assert app.manager.state == "CANCELLED"
            assert app.manager._process.poll() is not None
            assert not app.busy
            assert ExperimentRegistry().list()[0]["status"] == "CANCELLED"

    asyncio.run(exercise())


@pytest.mark.native
def test_complete_application_workflow(monkeypatch, tmp_path):
    capability = capabilities()
    if not engine_available(capability, "sequential"):
        pytest.skip(f"Native runner unavailable: {capability.get('error')}")
    monkeypatch.chdir(tmp_path)
    captures = tmp_path / "screens"
    captures.mkdir()

    async def exercise():
        app = HFTLabApp()
        async with app.run_test(size=(160, 55)) as pilot:
            async def finish_worker():
                # Let button messages start their worker before taking the
                # worker-manager snapshot (Pilot.click posts the message).
                await pilot.pause()
                await asyncio.wait_for(app.workers.wait_for_complete(), timeout=240)
                await pilot.pause()
                assert not app.busy

            async def page(name):
                await pilot.click(f"#nav-{name}")
                await pilot.pause()
                assert app.query_one("#pages", ContentSwitcher).current == name

            def field(name, value):
                app.query_one(f"#{name}", Input).value = str(value)

            await finish_worker()
            assert app.diagnostics["engines"]["sequential"] == "AVAILABLE"
            await page("data")
            field("dataset-path", "data/generated/ui.csv")
            field("data-events", 1600)
            app.query_one("#data-scenario", Select).value = "predictable"
            await pilot.click("#generate-data")
            await finish_worker()
            assert (tmp_path / "data/generated/ui.csv").is_file()
            await pilot.click("#validate-data")
            await finish_worker()
            field("import-source", "data/generated/ui.csv")
            field("dataset-path", "data/generated/ui.hftdata")
            await pilot.click("#import-data")
            await finish_worker()
            assert (tmp_path / "data/generated/ui.hftdata") in app.dataset_paths

            await page("training")
            field("train-epochs", 2)
            field("train-output", "models/ui-model")
            await pilot.click("#train-model")
            await finish_worker()
            model = tmp_path / "models/ui-model/model.hftmodel"
            assert model.is_file()
            assert app.query_one("#model-path", Input).value == str(model)
            await pilot.click("#use-model")
            await pilot.pause()
            assert app.query_one("#strategy-select", Select).value == "ai"
            assert app.query_one("#pages", ContentSwitcher).current == "engine"

            results = []
            for _, strategy in STRATEGIES:
                app.query_one("#strategy-select", Select).value = strategy
                app.query_one("#engine-select", Select).value = "cpu"
                app.query_one("#cpu-mode", Select).value = "single-single"
                await pilot.press("f5")
                await finish_worker()
                assert app.manager.state == "COMPLETED"
                assert app.result["strategy"] == strategy
                assert app.result["events_processed"] == 1600
                results.append({"strategy": strategy, "engine": app.result["engine"]})

            app.query_one("#strategy-select", Select).value = "momentum"
            modes = [("cpu", mode) for mode in ("single-single", "single-multi", "multi-single", "multi-multi")]
            modes += [(engine, "multi-multi") for engine in ("openmp", "mpi", "cuda", "hybrid")
                      if engine_available(capability, engine)]
            for engine, mode in modes:
                app.query_one("#engine-select", Select).value = engine
                app.query_one("#cpu-mode", Select).value = mode
                if engine == "hybrid" and all(engine_available(capability, item) for item in ("openmp", "mpi", "cuda")):
                    app.query_one("#hybrid-components", SelectionList).select_all()
                await pilot.pause()
                await pilot.press("f5")
                await finish_worker()
                assert app.manager.state == "COMPLETED"
                assert app.result["events_processed"] == 1600
                results.append({"category": engine, "mode": mode, "engine": app.result["engine"]})

            await pilot.press("f6")
            tabs = app.query_one("#result-tabs", TabbedContent)
            for key in ("overview", "performance", "scaling", "profiling", "openmp", "mpi",
                        "cuda", "hybrid", "trading", "ai", "correctness", "export"):
                tabs.active = f"tab-{key}"
                await pilot.pause()
            for extension in ("html", "csv", "json"):
                field("report-path", f"reports/ui-result.{extension}")
                await pilot.click("#export-report")
                assert (tmp_path / f"reports/ui-result.{extension}").is_file()

            await page("history")
            app.query_one("#history-table", DataTable).focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#pages", ContentSwitcher).current == "results"

            app.query_one("#engine-select", Select).value = "cpu"
            await page("config")
            await pilot.click("#run-benchmark")
            await finish_worker()
            assert app.result["status"] == "COMPLETED"
            assert all(row["status"] in ("COMPLETED", "UNAVAILABLE") for row in app.result["engines"])
            assert app.query_one("#benchmark-table", DataTable).row_count >= 2

            for name, _ in PAGES:
                await page(name)
                app.save_screenshot(filename=f"{name}.svg", path=str(captures))
            (tmp_path / "ui-workflow.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    asyncio.run(exercise())
