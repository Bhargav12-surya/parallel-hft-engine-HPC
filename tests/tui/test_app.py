"""Headless navigation and validation tests against the actual Textual app."""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("textual") is None, reason="Install hftlab[tui] for headless UI tests")


def test_launch_navigation_and_engine_categories(monkeypatch, tmp_path):
    from hftlab.tui.app import HFTLabApp, ENGINES, CPU_MODES
    from textual.widgets import ContentSwitcher, Select

    # Discovery is covered by integration tests; UI tests avoid a system probe.
    monkeypatch.setattr(HFTLabApp, "run_diagnostics", lambda self: None)
    monkeypatch.chdir(tmp_path)

    async def exercise():
        app = HFTLabApp()
        async with app.run_test(size=(130, 45)) as pilot:
            await pilot.pause()
            assert app.query_one("#pages", ContentSwitcher).current == "home"
            commands = [command.title for command in app.get_system_commands(app.screen)]
            assert "Run simulation" in commands
            assert "Cancel active job" in commands
            await pilot.click("#nav-engine")
            await pilot.pause()
            assert app.query_one("#pages", ContentSwitcher).current == "engine"
            assert len(ENGINES) == 5
            assert len(CPU_MODES) == 4
            app.query_one("#engine-select", Select).value = "hybrid"
            await pilot.pause()
            assert app.query_one("#hybrid-options").display
            assert not app.query_one("#cpu-options").display
            app.query_one("#engine-select", Select).value = "openmp"
            await pilot.pause()
            assert not app.query_one("#hybrid-options").display
            assert not app.query_one("#cpu-options").display
            await pilot.press("f1")
            assert app.query_one("#pages", ContentSwitcher).current == "help"
            await pilot.press("f2")
            assert app.query_one("#pages", ContentSwitcher).current == "data"

    asyncio.run(exercise())


def test_invalid_dataset_shows_friendly_error(monkeypatch, tmp_path):
    from hftlab.tui.app import HFTLabApp
    from textual.widgets import ContentSwitcher, Input, Static

    monkeypatch.setattr(HFTLabApp, "run_diagnostics", lambda self: None)
    monkeypatch.chdir(tmp_path)

    async def exercise():
        app = HFTLabApp()
        async with app.run_test(size=(110, 40)) as pilot:
            await pilot.pause()
            app.query_one("#dataset-path", Input).value = str(tmp_path / "does-not-exist.csv")
            await pilot.press("f5")
            await pilot.pause()
            assert not app.busy
            assert app.query_one("#pages", ContentSwitcher).current == "config"
            assert "does not exist" in str(app.query_one("#config-error", Static).render())

    asyncio.run(exercise())


def test_results_and_responsive_layout(monkeypatch, tmp_path):
    from hftlab.tui.app import HFTLabApp
    from textual.widgets import ContentSwitcher, Static

    monkeypatch.setattr(HFTLabApp, "run_diagnostics", lambda self: None)
    monkeypatch.chdir(tmp_path)

    async def exercise():
        app = HFTLabApp()
        async with app.run_test(size=(60, 30)) as pilot:
            await pilot.pause()
            assert app.has_class("tiny")
            app._simulation_finished({"engine": "sequential", "strategy": "momentum", "events_processed": 100, "wall_seconds": 0.05, "pnl": 12.5, "equity": 100012.5, "trades": 4, "checksum": "abc"})
            await pilot.press("f6")
            assert app.query_one("#pages", ContentSwitcher).current == "results"
            assert "sequential" in str(app.query_one("#result-overview", Static).render())
            assert "12.5" in str(app.query_one("#result-trading", Static).render())
            app.experiment_rows = [
                {"id": "slower", "result": {"wall_seconds": 2.0}},
                {"id": "faster", "result": {"wall_seconds": 0.1}},
            ]
            app.sort_history()
            assert [row["id"] for row in app.experiment_rows] == ["faster", "slower"]

    asyncio.run(exercise())


def test_actual_native_runner_through_tui_worker(monkeypatch, tmp_path):
    """Exercise generation → Textual worker → native subprocess → registry → UI."""
    from hftlab.datasets import generate_dataset
    from hftlab.engines import find_runner
    from hftlab.experiments import ExperimentRegistry
    from hftlab.tui.app import HFTLabApp
    from textual.widgets import Input, Static

    try:
        find_runner()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    monkeypatch.setattr(HFTLabApp, "run_diagnostics", lambda self: None)
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "native-market.csv"
    generate_dataset(dataset, events=256, instruments=2, seed=19, scenario="normal")

    async def exercise():
        app = HFTLabApp()
        async with app.run_test(size=(120, 42)) as pilot:
            await pilot.pause()
            app.query_one("#dataset-path", Input).value = str(dataset)
            await pilot.press("f5")
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=30)
            await pilot.pause()
            assert not app.busy
            assert app.result is not None
            assert app.result["engine"] == "sequential"
            assert app.result["events_processed"] == 256
            assert app.result["wall_seconds"] > 0
            assert app.result["checksum"]
            assert app.manager.state == "COMPLETED"
            assert "256 / 256" in str(app.query_one("#monitor-metrics", Static).render())
            persisted = ExperimentRegistry().get(app.manager.last_experiment_id)
            assert persisted["status"] == "COMPLETED"
            assert persisted["result"]["checksum"] == app.result["checksum"]
            assert "sequential" in str(app.query_one("#result-overview", Static).render())

    asyncio.run(exercise())


def test_simulation_worker_marshals_real_event_contract(monkeypatch, tmp_path):
    from hftlab.tui.app import HFTLabApp
    from textual.widgets import Input, Static

    class Runner:
        state = "CREATED"

        def run(self, config, on_event):
            self.state = "RUNNING"
            on_event({"type": "progress", "events_processed": 10, "total_events": 10, "elapsed_seconds": 0.01, "trades": 2, "pnl": 1.5, "equity": 100001.5})
            self.state = "COMPLETED"
            return {"engine": config.engine, "strategy": config.strategy, "wall_seconds": 0.01, "events_processed": 10, "trades": 2, "pnl": 1.5, "equity": 100001.5}

    monkeypatch.setattr(HFTLabApp, "run_diagnostics", lambda self: None)
    monkeypatch.setattr("hftlab.orchestration.JobManager", Runner)
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "market.csv"
    dataset.write_text("event_id,timestamp_ns\n1,1\n", encoding="utf-8")

    async def exercise():
        app = HFTLabApp()
        async with app.run_test(size=(120, 42)) as pilot:
            await pilot.pause()
            app.query_one("#dataset-path", Input).value = str(dataset)
            await pilot.press("f5")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not app.busy
            assert app.result["trades"] == 2
            assert "COMPLETED" in str(app.query_one("#monitor-state", Static).render())
            assert "10 / 10" in str(app.query_one("#monitor-metrics", Static).render())

    asyncio.run(exercise())
