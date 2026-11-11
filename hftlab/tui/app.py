"""Responsive terminal workspace connected to the native simulation runner.

All long-running operations live in Textual worker threads. Worker callbacks
cross into the event loop with ``call_from_thread``; subprocess stdout never
writes directly to the terminal.
"""

from __future__ import annotations

import json
import os
import threading
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable

from rich.markup import escape
from textual import on, work
from textual.events import Resize
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    ContentSwitcher,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    SelectionList,
    Static,
    TabbedContent,
    TabPane,
)


PAGES = (
    ("home", "⌂  Home"),
    ("data", "01  Market Data"),
    ("strategy", "02  Strategy"),
    ("engine", "03  HPC Engine"),
    ("config", "04  Simulation"),
    ("monitor", "05  Live Monitor"),
    ("results", "06  Results"),
    ("training", "07  AI Training"),
    ("history", "08  Experiments"),
    ("doctor", "09  Diagnostics"),
    ("help", "?   Help & Settings"),
)

STRATEGIES = (
    ("Market making", "market-making"),
    ("Momentum trading", "momentum"),
    ("Statistical arbitrage", "stat-arb"),
    ("Mean reversion", "mean-reversion"),
    ("AI-powered trading", "ai"),
)

ENGINES = (
    ("1. CPU Computing", "cpu"),
    ("2. OpenMP Shared-Memory Computing", "openmp"),
    ("3. MPI Distributed-Memory Computing", "mpi"),
    ("4. GPU-Accelerated CUDA Computing", "cuda"),
    ("5. Hybrid Computing", "hybrid"),
)

CPU_MODES = (
    ("Single-core, single-threaded", "single-single"),
    ("Single-core, multithreaded", "single-multi"),
    ("Multi-core, single-threaded", "multi-single"),
    ("Multi-core, multithreaded", "multi-multi"),
)


def render_json(value: Any) -> str:
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


class ConfirmScreen(ModalScreen[bool]):
    """Small keyboard-accessible confirmation for destructive/cancel actions."""

    BINDINGS = [Binding("escape", "dismiss(False)", "Back")]

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(self.message, id="confirm-message")
            with Horizontal(classes="button-row"):
                yield Button("Keep / Go back", id="confirm-no")
                yield Button("Confirm", variant="error", id="confirm-yes")

    @on(Button.Pressed, "#confirm-no")
    def no(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#confirm-yes")
    def yes(self) -> None:
        self.dismiss(True)


class HFTLabApp(App[None]):
    """One terminal workspace for data, training, native execution and reports."""

    TITLE = "HFT-LAB"
    SUB_TITLE = "PARALLEL & GPU-ACCELERATED TRADING SIMULATION"
    CSS_PATH = "styles.tcss"
    ENABLE_COMMAND_PALETTE = True
    BINDINGS = [
        Binding("ctrl+q", "request_quit", "Quit"),
        Binding("f1", "show_help", "Help"),
        Binding("f2", "show_data", "Data"),
        Binding("f5", "start_simulation", "Run"),
        Binding("f6", "show_results", "Results"),
        Binding("escape", "home", "Home"),
        Binding("ctrl+c", "cancel_job", "Cancel", priority=True),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.manager: Any = None
        self.result: dict[str, Any] | None = None
        self.diagnostics: dict[str, Any] = {}
        self.busy = False
        self.active_operation: str | None = None
        self._training_cancel = threading.Event()
        self._benchmark_cancel = threading.Event()
        self.dataset_paths: list[Path] = []
        self.experiment_rows: list[dict[str, Any]] = []
        self.history_sort_reverse = True
        self._prefs_path = Path(".hftlab-preferences.json")
        self._preferences: dict[str, Any] = {}
        try:
            self._preferences = json.loads(self._prefs_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="sidebar"):
                yield Static("[bold cyan]HFT-LAB[/]\n[dim]RESEARCH WORKSPACE[/]", id="brand")
                for key, label in PAGES:
                    yield Button(label, id=f"nav-{key}", classes="nav")
                yield Static("OFFLINE SIMULATION\n[dim]No live exchange access[/]", id="offline")
            with Vertical(id="main"):
                yield Static("READY  •  Native computation · reproducible experiments", id="status-line")
                with ContentSwitcher(initial="home", id="pages"):
                    yield from self._compose_home()
                    yield from self._compose_data()
                    yield from self._compose_strategy()
                    yield from self._compose_engine()
                    yield from self._compose_config()
                    yield from self._compose_monitor()
                    yield from self._compose_results()
                    yield from self._compose_training()
                    yield from self._compose_history()
                    yield from self._compose_doctor()
                    yield from self._compose_help()
        yield Footer()

    def _compose_home(self) -> Iterable[Any]:
        with VerticalScroll(id="home", classes="page"):
            from hftlab import __version__

            yield Static(f"HFT-LAB  [dim]v{__version__}[/]", classes="hero")
            yield Static("PARALLEL & GPU-ACCELERATED TRADING SIMULATION", classes="subtitle")
            yield Static(
                "A local research laboratory for deterministic market simulation, native C engines, "
                "machine learning and measured parallel performance.", classes="intro"
            )
            yield Static("Detecting computing resources…", id="home-resources", classes="card")
            with Horizontal(classes="button-row"):
                yield Button("New simulation →", variant="primary", id="quick-sim")
                yield Button("Generate data", id="quick-data")
                yield Button("Train model", id="quick-train")
            with Horizontal(classes="button-row"):
                yield Button("HPC benchmark", id="quick-benchmark")
                yield Button("Compare engines", id="quick-compare")
                yield Button("System diagnostics", id="quick-doctor")
            yield Label("RECENT EXPERIMENTS", classes="section-label")
            yield DataTable(id="home-history", cursor_type="row", zebra_stripes=True)
            yield Label("RECENT MODELS", classes="section-label")
            yield Static("No trained models yet. Open AI Training to create a native inference artifact.", id="home-models")
            yield Static("F2 Data  •  F5 Run  •  F6 Results  •  F1 Help  •  Ctrl+P Command palette", classes="muted")

    def _compose_data(self) -> Iterable[Any]:
        with VerticalScroll(id="data", classes="page"):
            yield Static("Market Data", classes="page-title")
            yield Static("Generate seeded scenarios or import a canonical event dataset. Validation runs before execution.", classes="muted")
            with Horizontal(classes="form-row"):
                with Vertical(classes="field"):
                    yield Label("Dataset output / selected dataset")
                    yield Input("data/generated/market.csv", id="dataset-path")
                with Vertical(classes="field small-field"):
                    yield Label("Events")
                    yield Input("10000", id="data-events", type="integer")
            with Horizontal(classes="form-row"):
                with Vertical(classes="field"):
                    yield Label("Scenario")
                    yield Select(
                        [(v.replace("-", " ").title(), v) for v in (
                            "normal", "high-volatility", "high-liquidity", "low-liquidity",
                            "shock", "correlated", "stress", "predictable")],
                        value="normal", allow_blank=False, id="data-scenario",
                    )
                with Vertical(classes="field small-field"):
                    yield Label("Instruments")
                    yield Input("4", id="data-instruments", type="integer")
                with Vertical(classes="field small-field"):
                    yield Label("Random seed")
                    yield Input("42", id="data-seed", type="integer")
            with Horizontal(classes="button-row"):
                yield Button("Generate dataset", variant="primary", id="generate-data")
                yield Button("Validate / inspect", id="validate-data")
                yield Button("Browse files", id="refresh-data")
                yield Button("Delete selected", variant="error", id="delete-data")
            yield Label("Import source path")
            yield Input(placeholder="/path/to/market.csv or events.bin", id="import-source")
            yield Button("Import into selected output path", id="import-data")
            yield DataTable(id="datasets-table", cursor_type="row", zebra_stripes=True)
            yield RichLog(id="data-log", highlight=True, markup=False, wrap=True, min_width=20)

    def _compose_strategy(self) -> Iterable[Any]:
        with VerticalScroll(id="strategy", classes="page"):
            yield Static("Trading Strategy", classes="page-title")
            yield Label("One strategy contract across compatible computation engines")
            yield Select(STRATEGIES, value="momentum", allow_blank=False, id="strategy-select")
            yield Static("", id="strategy-description", classes="card")
            yield Label("AI native model path (required only for AI trading)")
            yield Input("models/reference.hftmodel", id="model-path")
            yield Static("Position, loss, size and execution-cost controls are configured on the Simulation page.", classes="muted")
            yield Button("Choose computation engine →", variant="primary", id="strategy-next")

    def _compose_engine(self) -> Iterable[Any]:
        with VerticalScroll(id="engine", classes="page"):
            yield Static("HPC Computation Engine", classes="page-title")
            yield Select(ENGINES, value="cpu", allow_blank=False, id="engine-select")
            with Vertical(id="cpu-options", classes="card"):
                yield Label("CPU Computing — execution topology")
                yield Select(CPU_MODES, value="single-single", allow_blank=False, id="cpu-mode")
                yield Static("A single software worker may migrate between cores; only multiple workers run simultaneously.", classes="muted")
            with Vertical(id="hybrid-options", classes="card"):
                yield Label("Choose two or more components. Unsupported plans are rejected.")
                yield SelectionList(
                    ("CPU Computing", "cpu", True),
                    ("OpenMP", "openmp", True),
                    ("MPI", "mpi", False),
                    ("CUDA", "cuda", False),
                    id="hybrid-components",
                )
            yield Static("OpenMP and MPI configure workers automatically. CUDA requires a compatible native build and GPU.", id="engine-note", classes="card")
            yield Static("", id="engine-capabilities", classes="muted")
            yield Button("Configure simulation →", variant="primary", id="engine-next")

    def _compose_config(self) -> Iterable[Any]:
        with VerticalScroll(id="config", classes="page"):
            yield Static("Simulation Configuration", classes="page-title")
            yield Static("Dataset, strategy and engine use the selections on their workflow pages.", id="config-summary", classes="card")
            fields = (
                ("Initial capital", "capital", "100000"),
                ("Fee (basis points)", "fee-bps", "1"),
                ("Slippage (ticks)", "slippage-ticks", "0"),
                ("Simulated latency (ns)", "latency-ns", "0"),
                ("Position limit", "position-limit", "100"),
                ("Maximum order quantity", "max-order", "10"),
                ("Maximum loss", "max-loss", "10000"),
                ("Timeout (seconds)", "timeout", "300"),
            )
            for index in range(0, len(fields), 2):
                with Horizontal(classes="form-row"):
                    for title, key, value in fields[index:index + 2]:
                        with Vertical(classes="field"):
                            yield Label(title)
                            yield Input(value, id=key, type="number")
            yield Label("Output directory")
            yield Input("results", id="output-directory")
            with Horizontal(classes="button-row"):
                yield Button("Run simulation  F5", variant="success", id="run-simulation")
                yield Button("Benchmark available CPU engines", id="run-benchmark")
                yield Button("Save preferences", id="save-preferences")
            yield Static("", id="config-error", classes="error")

    def _compose_monitor(self) -> Iterable[Any]:
        with Vertical(id="monitor", classes="page"):
            yield Static("Live Simulation Monitor", classes="page-title")
            yield Static("IDLE  •  Configure a simulation and press F5", id="monitor-state", classes="card")
            yield ProgressBar(total=100, show_eta=True, id="simulation-progress")
            yield Static("Events 0  •  Trades 0  •  Elapsed 0 s", id="monitor-metrics", classes="card")
            yield Static("P&L —  •  Equity —  •  Throughput —", id="monitor-trading", classes="card")
            yield Static("Resource measurements appear when reported by the native runner.", id="monitor-resources", classes="muted")
            with Horizontal(classes="button-row"):
                yield Button("Cancel job", variant="error", id="cancel-job")
                yield Button("Open final results", id="monitor-results")
            yield RichLog(id="monitor-log", highlight=False, markup=False, wrap=True, min_width=20)

    def _compose_results(self) -> Iterable[Any]:
        with Vertical(id="results", classes="page"):
            yield Static("Results Center", classes="page-title")
            with TabbedContent(id="result-tabs"):
                for label, key in (
                    ("HPC Overview", "overview"), ("Engine", "performance"),
                    ("Scalability", "scaling"), ("CPU / OS", "profiling"),
                    ("OpenMP", "openmp"), ("MPI", "mpi"), ("CUDA", "cuda"),
                    ("Hybrid", "hybrid"), ("Trading", "trading"),
                    ("AI", "ai"), ("Correctness", "correctness"),
                ):
                    with TabPane(label, id=f"tab-{key}"):
                        with VerticalScroll():
                            if key == "overview":
                                yield DataTable(id="benchmark-table", cursor_type="row", zebra_stripes=True)
                            yield Static("Run a simulation or reopen an experiment to view measured results.", id=f"result-{key}", markup=False)
                with TabPane("Export", id="tab-export"):
                    yield Label("Report path (.html, .json or .csv)")
                    yield Input("reports/experiment.html", id="report-path")
                    yield Button("Export current result", variant="primary", id="export-report")
                    yield Static("", id="export-status")

    def _compose_training(self) -> Iterable[Any]:
        with VerticalScroll(id="training", classes="page"):
            yield Static("AI Training Center", classes="page-title")
            yield Static("Causal features · chronological purged splits · training-only preprocessing · native weights", classes="muted")
            yield Label("Training dataset")
            yield Input("data/generated/market.csv", id="train-dataset")
            yield Label("Model type")
            yield Select([("Logistic regression", "logistic"), ("Multilayer perceptron", "mlp")], value="mlp", allow_blank=False, id="train-kind")
            with Horizontal(classes="form-row"):
                with Vertical(classes="field"):
                    yield Label("Epochs")
                    yield Input("30", id="train-epochs", type="integer")
                with Vertical(classes="field"):
                    yield Label("Random seed")
                    yield Input("42", id="train-seed", type="integer")
            yield Label("Native artifact output")
            yield Input("models/reference.hftmodel", id="train-output")
            with Horizontal(classes="button-row"):
                yield Button("Train & export native model", variant="primary", id="train-model")
                yield Button("Browse models", id="browse-models")
                yield Button("Use selected model for AI trading", id="use-model")
            yield Static("DIRECTION CLASSES: BUY / HOLD / SELL\nPredictable scenarios are synthetic teaching data. Test accuracy does not establish trading profitability.", classes="card")
            yield RichLog(id="training-log", highlight=False, markup=False, wrap=True, min_width=20)

    def _compose_history(self) -> Iterable[Any]:
        with Vertical(id="history", classes="page"):
            yield Static("Experiment History", classes="page-title")
            yield Input(placeholder="Filter by experiment, dataset, engine, strategy…", id="history-filter")
            with Horizontal(classes="button-row"):
                yield Button("Refresh", id="refresh-history")
                yield Button("Open selected", variant="primary", id="open-history")
                yield Button("Sort by runtime", id="sort-history")
            yield DataTable(id="history-table", cursor_type="row", zebra_stripes=True)
            yield Static("Select an experiment row and press Enter to reopen its results.", classes="muted")

    def _compose_doctor(self) -> Iterable[Any]:
        with VerticalScroll(id="doctor", classes="page"):
            yield Static("System Diagnostics", classes="page-title")
            yield Static("AVAILABLE / UNAVAILABLE / INCOMPATIBLE / NOT TESTED reflect actual discovery.", classes="muted")
            yield Button("Refresh diagnostics", variant="primary", id="refresh-doctor")
            yield RichLog(id="doctor-log", highlight=False, markup=False, wrap=True, min_width=20)

    def _compose_help(self) -> Iterable[Any]:
        with VerticalScroll(id="help", classes="page"):
            yield Static("Help & Settings", classes="page-title")
            yield Static(
                "WORKFLOW\n\n1  Generate or import market data.\n2  Choose a native strategy.\n"
                "3  Select a supported engine.\n4  Set capital and risk controls.\n"
                "5  Run, monitor, compare and export.\n\n"
                "KEYBOARD\n\nTab / Shift+Tab: move focus\nEnter / Space: activate controls\n"
                "F1: Help   F2: Data   F5: Run   F6: Results\n"
                "Ctrl+P: command palette   Ctrl+C: cancel job   Ctrl+Q: quit\n\n"
                "BENCHMARKS\n\nAvailable engines are measured against a sequential reference. "
                "Missing capabilities cause clear errors; an unavailable GPU is never substituted with CPU execution.\n\n"
                "HEADLESS COMMANDS\n\npython -m hftlab doctor\npython -m hftlab generate --help\n"
                "python -m hftlab simulate --help\npython -m hftlab train --help\n"
                "python -m hftlab benchmark --help\n\n"
                "Documentation: README.md and docs/USER_GUIDE.md", classes="card", markup=False,
            )
            yield Checkbox("Use compact interface", value=bool(self._preferences.get("compact", False)), id="compact-mode")
            yield Button("Save preferences", id="help-save-preferences")
            yield Static("Saved preferences: .hftlab-preferences.json", classes="muted")

    def on_mount(self) -> None:
        self.query_one("#datasets-table", DataTable).add_columns("Dataset", "Size", "Modified")
        self.query_one("#history-table", DataTable).add_columns("Experiment", "Engine", "Strategy", "Seconds", "P&L", "Status")
        self.query_one("#home-history", DataTable).add_columns("Experiment", "Engine", "Strategy", "Status")
        self.query_one("#benchmark-table", DataTable).add_columns("Engine", "Median (s)", "Events/s", "Speedup", "Efficiency", "Status")
        self.query_one("#benchmark-table", DataTable).display = False
        self._update_strategy_description()
        self._update_engine_controls()
        self._restore_preferences()
        self.refresh_datasets()
        self.refresh_history()
        self._browse_models()
        self.run_diagnostics()

    def on_resize(self, event: Resize) -> None:
        self.set_class(event.size.width < 90, "narrow")
        self.set_class(event.size.width < 65, "tiny")

    def _input(self, key: str) -> str:
        return self.query_one(f"#{key}", Input).value.strip()

    def _selected(self, key: str) -> str:
        value = self.query_one(f"#{key}", Select).value
        if value is Select.BLANK:
            raise ValueError(f"Choose {key.replace('-', ' ')}")
        return str(value)

    def show_page(self, page: str) -> None:
        self.query_one("#pages", ContentSwitcher).current = page
        for button in self.query(".nav"):
            button.set_class(button.id == f"nav-{page}", "active")
        if page == "config":
            self._update_config_summary()
        if page == "history":
            self.refresh_history()

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        for command in super().get_system_commands(screen):
            if command.title != "Quit the application":
                yield command
        yield SystemCommand("Quit HFT-LAB", "Confirm before cancelling active work", self.action_request_quit)
        for page, label in PAGES:
            yield SystemCommand(f"Open {label.strip()}", f"Navigate to the {page} workspace", partial(self.show_page, page))
        yield SystemCommand("Run simulation", "Validate configuration and launch the native engine", self.action_start_simulation)
        yield SystemCommand("Cancel active job", "Request native, benchmark or training cancellation", self.action_cancel_job)

    def action_home(self) -> None:
        self.show_page("home")

    def action_show_help(self) -> None:
        self.show_page("help")

    def action_show_data(self) -> None:
        self.show_page("data")

    def action_show_results(self) -> None:
        self.show_page("results")

    def action_request_quit(self) -> None:
        if self.busy:
            self.push_screen(ConfirmScreen("Cancel the running job and quit HFT-LAB?"), self._quit_confirmed)
        else:
            self.exit()

    def _quit_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            self._training_cancel.set()
            self._benchmark_cancel.set()
            if self.manager is not None:
                self.manager.cancel()
            self.exit()

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        key = event.button.id or ""
        if key.startswith("nav-"):
            self.show_page(key[4:])
            return
        pages = {
            "quick-sim": "data", "quick-data": "data", "quick-train": "training",
            "quick-doctor": "doctor", "strategy-next": "engine", "engine-next": "config",
            "monitor-results": "results", "quick-benchmark": "config", "quick-compare": "config",
        }
        if key in pages:
            self.show_page(pages[key])
            return
        actions: dict[str, Callable[[], Any]] = {
            "generate-data": self.generate_data,
            "validate-data": self.validate_data,
            "import-data": self.import_data,
            "refresh-data": self.refresh_datasets,
            "delete-data": self.confirm_delete_dataset,
            "run-simulation": self.action_start_simulation,
            "run-benchmark": self.start_benchmark,
            "cancel-job": self.action_cancel_job,
            "train-model": self.start_training,
            "browse-models": self._browse_models,
            "use-model": self.use_trained_model,
            "refresh-history": self.refresh_history,
            "open-history": self.open_selected_history,
            "sort-history": self.sort_history,
            "refresh-doctor": self.run_diagnostics,
            "export-report": self.export_current_report,
            "save-preferences": self.save_preferences,
            "help-save-preferences": self.save_preferences,
        }
        if key in actions:
            try:
                actions[key]()
            except (ValueError, OSError, TypeError) as exc:
                self._show_error(exc)

    @on(Select.Changed, "#strategy-select")
    def strategy_changed(self) -> None:
        self._update_strategy_description()

    @on(Select.Changed, "#engine-select")
    def engine_changed(self) -> None:
        self._update_engine_controls()

    @on(Input.Changed, "#history-filter")
    def history_filter_changed(self) -> None:
        self.refresh_history()

    @on(Checkbox.Changed, "#compact-mode")
    def compact_changed(self, event: Checkbox.Changed) -> None:
        self.set_class(event.value, "compact")

    @on(DataTable.RowSelected, "#datasets-table")
    def dataset_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key.value:
            self.query_one("#dataset-path", Input).value = event.row_key.value
            self.query_one("#train-dataset", Input).value = event.row_key.value

    @on(DataTable.RowSelected, "#history-table")
    def history_selected(self) -> None:
        self.open_selected_history()

    @on(DataTable.RowSelected, "#home-history")
    def home_history_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key.value:
            self._open_experiment(event.row_key.value)

    def _update_strategy_description(self) -> None:
        if not self.is_mounted:
            return
        descriptions = {
            "market-making": "MARKET MAKING\nInventory-aware bid/ask quotes with refresh, matching, fees and risk limits.",
            "momentum": "MOMENTUM\nCausal short/long price indicators generate orders within inventory and execution-cost limits.",
            "stat-arb": "STATISTICAL ARBITRAGE\nPaired instrument spread signals with execution costs. At least two instruments are required.",
            "mean-reversion": "MEAN REVERSION\nRolling fair value and volatility produce deviation signals and position targets.",
            "ai": "AI-POWERED TRADING\nA trained native model supplies BUY/HOLD/SELL probabilities. Risk and execution controls still apply.",
        }
        strategy = self._selected("strategy-select")
        self.query_one("#strategy-description", Static).update(descriptions[strategy])
        self.query_one("#model-path", Input).disabled = strategy != "ai"

    def _update_engine_controls(self) -> None:
        if not self.is_mounted:
            return
        engine = self._selected("engine-select")
        self.query_one("#cpu-options").display = engine == "cpu"
        self.query_one("#hybrid-options").display = engine == "hybrid"

    def _update_config_summary(self) -> None:
        try:
            summary = (
                f"Dataset: {self._input('dataset-path')}\n"
                f"Strategy: {self._selected('strategy-select')}  •  Engine: {self._selected('engine-select')}"
            )
            self.query_one("#config-summary", Static).update(summary)
        except (ValueError, LookupError):
            pass

    def _show_error(self, exc: BaseException) -> None:
        message = str(exc) or type(exc).__name__
        self.notify(message, severity="error", timeout=8)
        self.query_one("#config-error", Static).update(message)
        self.query_one("#status-line", Static).update(f"ERROR  •  {escape(message)}")

    def _set_busy(self, busy: bool, label: str = "READY") -> None:
        self.busy = busy
        if not busy:
            self.active_operation = None
        self.query_one("#status-line", Static).update(escape(label))
        for key in ("run-simulation", "run-benchmark", "generate-data", "import-data", "train-model"):
            self.query_one(f"#{key}", Button).disabled = busy

    def _begin_job(self, label: str) -> bool:
        if self.busy:
            self.notify("A job is already running. Finish or cancel it before starting another.", severity="warning")
            return False
        self.manager = None
        self._set_busy(True, label)
        return True

    def _build_config(self) -> Any:
        from hftlab.configuration import SimulationConfig

        category = self._selected("engine-select")
        mode = self._selected("cpu-mode")
        engine = ("threaded" if mode in ("single-multi", "multi-multi") else "sequential") if category == "cpu" else category
        components = list(self.query_one("#hybrid-components", SelectionList).selected)
        if engine == "hybrid" and len(components) < 2:
            raise ValueError("Hybrid computing requires at least two selected components.")
        dataset = Path(self._input("dataset-path")).expanduser()
        if not dataset.is_file():
            raise ValueError(f"Dataset does not exist: {dataset}. Generate or import one on the Market Data page.")
        strategy = self._selected("strategy-select")
        model = Path(self._input("model-path")).expanduser() if strategy == "ai" else None
        if model is not None and not model.is_file():
            raise ValueError(f"AI model does not exist: {model}. Train a model in the AI Training Center.")
        config = SimulationConfig(
            dataset=dataset,
            engine=engine,
            cpu_mode=mode,
            strategy=strategy,
            threads=max(1, min(os.cpu_count() or 1, 4)),
            components=components,
            capital=float(self._input("capital")),
            fee_bps=float(self._input("fee-bps")),
            slippage_ticks=int(self._input("slippage-ticks")),
            latency_ns=int(self._input("latency-ns")),
            position_limit=int(self._input("position-limit")),
            max_order=int(self._input("max-order")),
            max_loss=float(self._input("max-loss")),
            model=model,
            output_dir=Path(self._input("output-directory")).expanduser(),
            timeout=float(self._input("timeout")),
        )
        validate = getattr(config, "validate", None)
        if validate is not None:
            validate()
        return config

    def action_start_simulation(self) -> None:
        try:
            config = self._build_config()
        except (ValueError, OSError, TypeError) as exc:
            self.show_page("config")
            self._show_error(exc)
            return
        if not self._begin_job("STARTING  •  Native simulation"):
            return
        self.active_operation = "native"
        self.show_page("monitor")
        self.query_one("#monitor-log", RichLog).clear()
        self.query_one("#simulation-progress", ProgressBar).update(total=100, progress=0)
        self.query_one("#monitor-state", Static).update(f"STARTING  •  {config.engine}  •  {config.strategy}")
        self._run_simulation(config)

    @work(thread=True, group="native", exclusive=True, exit_on_error=False)
    def _run_simulation(self, config: Any) -> None:
        try:
            from hftlab.orchestration import JobManager

            self.manager = JobManager()
            result = self.manager.run(config, on_event=lambda event: self.call_from_thread(self._progress, event))
            self.call_from_thread(self._simulation_finished, result)
        except Exception as exc:
            self.call_from_thread(self._job_failed, str(exc), "monitor-log")

    def _progress(self, event: dict[str, Any]) -> None:
        kind = event.get("type", "event")
        if kind == "progress":
            processed = int(event.get("events_processed", event.get("processed", 0)))
            total = max(processed, int(event.get("total_events", event.get("total", 0))), 1)
            self.query_one("#simulation-progress", ProgressBar).update(total=total, progress=processed)
            seconds = float(event.get("elapsed_seconds", event.get("wall_seconds", 0)))
            trades = event.get("trades", event.get("executed_trades", "—"))
            throughput = event.get("throughput", processed / seconds if seconds > 0 else 0)
            self.query_one("#monitor-metrics", Static).update(
                f"Events {processed:,} / {total:,}  •  Trades {trades}  •  Elapsed {seconds:.3f} s\n"
                f"Orders {event.get('orders', '—')}  •  Market time {event.get('timestamp_ns', '—')} ns"
            )
            self.query_one("#monitor-trading", Static).update(
                f"P&L {event.get('pnl', '—')}  •  Equity {event.get('equity', '—')}  •  Throughput {float(throughput):,.0f} events/s"
            )
            self.query_one("#monitor-state", Static).update(f"RUNNING  •  {event.get('phase', 'processing events')}")
            if any(k in event for k in ("cpu_percent", "memory_mb", "gpu_percent")):
                self.query_one("#monitor-resources", Static).update(
                    f"CPU {event.get('cpu_percent', '—')}%  •  Memory {event.get('memory_mb', '—')} MB  •  GPU {event.get('gpu_percent', '—')}%"
                )
        elif kind == "state":
            self.query_one("#monitor-state", Static).update(f"{event.get('state', 'RUNNING')}  •  Native job lifecycle")
            self.query_one("#monitor-log", RichLog).write(str(event.get("state", "")))
        else:
            self.query_one("#monitor-log", RichLog).write(render_json(event))
        if "memory_bytes" in event:
            self.query_one("#monitor-resources", Static).update(f"Native process resident memory: {int(event['memory_bytes']) / (1024 * 1024):,.2f} MB  •  CPU/GPU utilization requires profiler telemetry")

    def _simulation_finished(self, result: dict[str, Any]) -> None:
        self.result = result
        self._set_busy(False, "COMPLETED  •  Results persisted")
        self.query_one("#monitor-state", Static).update("COMPLETED  •  Open Results to inspect the experiment")
        progress = self.query_one("#simulation-progress", ProgressBar)
        progress.update(progress=progress.total or 100)
        self.query_one("#monitor-log", RichLog).write("Native runner completed; final result validated and saved.")
        self._render_results(result)
        self.refresh_history()
        self.notify("Simulation completed. Press F6 for results.", severity="information")

    def _job_failed(self, message: str, log_id: str) -> None:
        state = getattr(self.manager, "state", "FAILED")
        state_text = str(getattr(state, "value", state))
        if "cancel" in message.lower():
            state_text = "CANCELLED"
        self._set_busy(False, f"{state_text}  •  {message}")
        self.query_one(f"#{log_id}", RichLog).write(message)
        self.query_one("#monitor-state", Static).update(f"{state_text}  •  {message}")
        self.notify(message, severity="warning" if "CANCEL" in state_text else "error", timeout=10)
        self.refresh_history()

    def action_cancel_job(self) -> None:
        if not self.busy:
            self.notify("No job is running.")
        elif self.active_operation == "training":
            self.push_screen(ConfirmScreen("Cancel training after the current epoch?"), self._cancel_training_confirmed)
        elif self.active_operation == "benchmark":
            self.push_screen(ConfirmScreen("Cancel the running benchmark and clean up its native processes?"), self._cancel_benchmark_confirmed)
        elif self.active_operation != "native" or self.manager is None:
            self.notify("This operation finishes at its current work unit; native simulation cancellation is available while a simulation is active.", severity="warning")
        else:
            self.push_screen(ConfirmScreen("Cancel the running native simulation?"), self._cancel_confirmed)

    def _cancel_confirmed(self, confirmed: bool | None) -> None:
        if confirmed and self.manager is not None:
            self.manager.cancel()
            self.query_one("#monitor-state", Static).update("CANCELLING  •  Waiting for native process cleanup")

    def _cancel_training_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            self._training_cancel.set()
            self.query_one("#training-log", RichLog).write("Cancellation requested; waiting for the current training epoch.")

    def _cancel_benchmark_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            self._benchmark_cancel.set()
            self.query_one("#monitor-state", Static).update("CANCELLING  •  Waiting for benchmark process cleanup")

    def _render_results(self, result: dict[str, Any]) -> None:
        r = result.get("result", result)
        self.query_one("#benchmark-table", DataTable).display = False
        overview_keys = ("schema_version", "experiment_id", "engine", "strategy", "status", "events_processed", "wall_seconds", "throughput", "workers", "threads", "checksum")
        self._result_update("overview", {k: r[k] for k in overview_keys if k in r})
        self._result_update("performance", {k: r[k] for k in ("wall_seconds", "throughput", "latency_us", "timings", "workers", "threads", "ranks") if k in r})
        self._result_update("scaling", r.get("scaling", r.get("benchmarks", "Run an engine benchmark to measure speedup, efficiency and scaling. No scaling measurements are inferred from a single run.")))
        profiling = r.get("profiling", {})
        self._result_update("profiling", profiling or "Native profiling measurements were not included in this result.")
        for name in ("openmp", "mpi", "cuda", "hybrid"):
            value = r.get(name, profiling.get(name) if isinstance(profiling, dict) else None)
            if value is None:
                matching = {k: v for k, v in profiling.items() if name in k} if isinstance(profiling, dict) else {}
                value = matching or f"No {name.upper()} measurements in this experiment. Select the corresponding available engine to collect them."
            self._result_update(name, value)
        self._result_update("trading", {k: r[k] for k in ("pnl", "equity", "cash", "fees", "trades", "orders", "rejected_orders", "drawdown", "positions", "instruments", "slippage", "equity_curve") if k in r})
        self._result_update("ai", r.get("ai", r.get("model", "No AI model was used in this experiment.")))
        self._result_update("correctness", {k: r[k] for k in ("checksum", "correctness", "reference_checksum", "deterministic", "dataset_hash", "config") if k in r})

    def _result_update(self, key: str, value: Any) -> None:
        self.query_one(f"#result-{key}", Static).update(value if isinstance(value, str) else render_json(value))

    def generate_data(self) -> None:
        try:
            path = Path(self._input("dataset-path"))
            events = int(self._input("data-events"))
            instruments = int(self._input("data-instruments"))
            seed = int(self._input("data-seed"))
            if events < 1 or instruments < 1:
                raise ValueError("Event and instrument counts must be positive.")
            scenario = self._selected("data-scenario")
        except ValueError as exc:
            self._show_error(exc)
            return
        if self._begin_job("GENERATING  •  Writing reproducible market events"):
            self._data_operation("generate", path, events=events, instruments=instruments, seed=seed, scenario=scenario)

    def validate_data(self) -> None:
        if self._begin_job("VALIDATING  •  Dataset schema and event integrity"):
            self._data_operation("validate", Path(self._input("dataset-path")))

    def import_data(self) -> None:
        source = Path(self._input("import-source"))
        if not source.is_file():
            self._show_error(ValueError(f"Import source does not exist: {source}"))
            return
        destination = Path(self._input("dataset-path"))
        if source.resolve() == destination.resolve():
            self._show_error(ValueError("Choose a different output path for import."))
            return
        if self._begin_job("IMPORTING  •  Validating market events"):
            self._data_operation("import", destination, source=source)

    @work(thread=True, group="data", exclusive=True, exit_on_error=False)
    def _data_operation(self, operation: str, path: Path, **kwargs: Any) -> None:
        try:
            from hftlab.datasets import generate_dataset, import_dataset, validate_dataset

            if operation == "generate":
                value = generate_dataset(path, **kwargs)
            elif operation == "import":
                value = import_dataset(kwargs["source"], path)
            else:
                value = validate_dataset(path)
            self.call_from_thread(self._data_finished, operation, path, value)
        except Exception as exc:
            self.call_from_thread(self._job_failed, str(exc), "data-log")

    def _data_finished(self, operation: str, path: Path, value: Any) -> None:
        self._set_busy(False, f"COMPLETED  •  Dataset {operation}")
        self.query_one("#data-log", RichLog).write(render_json(value))
        self.query_one("#train-dataset", Input).value = str(path)
        self.refresh_datasets()
        self.notify(f"Dataset {operation} completed: {path}")

    def refresh_datasets(self) -> None:
        import datetime

        table = self.query_one("#datasets-table", DataTable)
        table.clear()
        roots = [Path("data/generated"), Path("data/samples"), Path("data/processed")]
        paths = {p.resolve() for root in roots if root.exists() for p in root.iterdir() if p.is_file() and p.suffix in (".csv", ".bin", ".parquet")}
        selected = Path(self._input("dataset-path"))
        if selected.is_file():
            paths.add(selected.resolve())
        self.dataset_paths = sorted(paths)
        for path in self.dataset_paths:
            stat = path.stat()
            table.add_row(str(path), f"{stat.st_size / 1024:,.1f} KB", datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%m-%d %H:%M"), key=str(path))

    def confirm_delete_dataset(self) -> None:
        path = Path(self._input("dataset-path")).resolve()
        managed = Path("data").resolve()
        if not path.is_relative_to(managed):
            raise ValueError("The UI deletes only files inside the project data directory.")
        if not path.is_file():
            raise ValueError(f"Dataset does not exist: {path}")
        self.push_screen(ConfirmScreen(f"Delete dataset {path.name}? This cannot be undone."), lambda yes: self._delete_dataset(path) if yes else None)

    def _delete_dataset(self, path: Path) -> None:
        try:
            path.unlink()
            self.refresh_datasets()
            self.notify(f"Deleted {path.name}")
        except OSError as exc:
            self._show_error(exc)

    def start_training(self) -> None:
        try:
            epochs = int(self._input("train-epochs"))
            seed = int(self._input("train-seed"))
            if epochs < 1:
                raise ValueError("Epochs must be positive.")
            dataset = Path(self._input("train-dataset"))
            if not dataset.is_file():
                raise ValueError(f"Training dataset does not exist: {dataset}")
            output = Path(self._input("train-output"))
            kind = self._selected("train-kind")
        except ValueError as exc:
            self._show_error(exc)
            return
        if self._begin_job("TRAINING  •  Fitting model on chronological training data"):
            self.active_operation = "training"
            self._training_cancel.clear()
            self.query_one("#training-log", RichLog).write(f"Training {kind}; epochs={epochs}; seed={seed}; dataset={dataset}")
            self._train_model(dataset, output, kind, epochs, seed)

    @work(thread=True, group="training", exclusive=True, exit_on_error=False)
    def _train_model(self, dataset: Path, output: Path, kind: str, epochs: int, seed: int) -> None:
        try:
            from hftlab.ai import train_model

            def progress(record: dict[str, Any]) -> None:
                if self._training_cancel.is_set():
                    raise RuntimeError("Training cancelled at an epoch boundary.")
                self.call_from_thread(self._training_progress, record)

            value = train_model(dataset, output, kind=kind, epochs=epochs, seed=seed, callback=progress)
            self.call_from_thread(self._training_finished, output, value)
        except Exception as exc:
            self.call_from_thread(self._job_failed, str(exc), "training-log")

    def _training_progress(self, record: dict[str, Any]) -> None:
        self.query_one("#training-log", RichLog).write(render_json(record))
        self.query_one("#status-line", Static).update(f"TRAINING  •  Epoch {record.get('epoch', '—')}  •  Validation loss {record.get('validation_loss', record.get('val_loss', '—'))}")

    def _training_finished(self, output: Path, value: dict[str, Any]) -> None:
        self._set_busy(False, "COMPLETED  •  Model trained and exported")
        self.query_one("#training-log", RichLog).write(render_json(value))
        self.query_one("#model-path", Input).value = str(output)
        self._browse_models()
        self.notify("Model training completed. Validation and test metrics are in the training log.")

    def _browse_models(self) -> None:
        root = Path("models")
        paths = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix in (".hftmodel", ".json", ".txt", ".model")), key=lambda p: p.stat().st_mtime, reverse=True)[:10] if root.exists() else []
        summary = "\n".join(str(path) for path in paths) or "No trained models yet. Train a logistic regression or MLP model in the AI Training Center."
        self.query_one("#home-models", Static).update(summary)
        if self.is_mounted and self.query_one("#pages", ContentSwitcher).current == "training":
            self.query_one("#training-log", RichLog).write(summary)

    def use_trained_model(self) -> None:
        output = Path(self._input("train-output"))
        if not output.is_file():
            raise ValueError(f"Model artifact does not exist: {output}")
        self.query_one("#model-path", Input).value = str(output)
        self.query_one("#strategy-select", Select).value = "ai"
        self.show_page("engine")

    def start_benchmark(self) -> None:
        try:
            config = self._build_config()
        except (ValueError, OSError, TypeError) as exc:
            self._show_error(exc)
            return
        if self._begin_job("BENCHMARKING  •  Available native CPU engines"):
            self.active_operation = "benchmark"
            self._benchmark_cancel.clear()
            self.show_page("monitor")
            self.query_one("#monitor-state", Static).update("BENCHMARKING  •  Timing sequential reference and available parallel engines")
            self.query_one("#monitor-log", RichLog).write("Benchmark measurements exclude UI refresh work. Progress is reported at completed run boundaries.")
            self._benchmark(config)

    @work(thread=True, group="benchmark", exclusive=True, exit_on_error=False)
    def _benchmark(self, config: Any) -> None:
        try:
            from hftlab.benchmarks import benchmark

            value = benchmark(config, repeats=3, engines=["sequential", "threaded", "openmp"], output=Path("reports/benchmark.json"), cancel_event=self._benchmark_cancel)
            self.call_from_thread(self._benchmark_finished, value)
        except Exception as exc:
            self.call_from_thread(self._job_failed, str(exc), "monitor-log")

    def _benchmark_finished(self, value: dict[str, Any]) -> None:
        self.result = value
        self._set_busy(False, "COMPLETED  •  Benchmark report saved")
        self.query_one("#monitor-state", Static).update("COMPLETED  •  Benchmark measurements available in Results")
        self.query_one("#monitor-log", RichLog).write(render_json(value))
        self._result_update("overview", value)
        self._result_update("scaling", value)
        self._result_update("correctness", value.get("correctness", value))
        table = self.query_one("#benchmark-table", DataTable)
        table.clear()
        table.display = True
        for row in value.get("engines", []):
            timing = row.get("timing", {})
            def number(metric: Any, precision: int = 3) -> str:
                return f"{metric:,.{precision}f}" if isinstance(metric, (int, float)) else "—"
            table.add_row(
                str(row.get("engine", "—")), number(timing.get("median_seconds"), 6),
                number(row.get("throughput"), 0), number(row.get("speedup")),
                number(row.get("efficiency")), str(row.get("status", "—")),
            )
        self.show_page("results")
        self.refresh_history()

    def refresh_history(self) -> None:
        try:
            from hftlab.experiments import ExperimentRegistry

            search = self._input("history-filter")
            self.experiment_rows = ExperimentRegistry().list(limit=100, search=search)
        except (ImportError, OSError, ValueError):
            self.experiment_rows = []
        self._render_history()

    def _render_history(self) -> None:
        table = self.query_one("#history-table", DataTable)
        recent = self.query_one("#home-history", DataTable)
        table.clear()
        recent.clear()
        for index, row in enumerate(self.experiment_rows):
            key = str(row.get("id", row.get("experiment_id", index)))
            result = row.get("result", {}) or {}
            if not isinstance(result, dict):
                result = {}
            engine = str(row.get("engine", result.get("engine", "—")))
            strategy = str(row.get("strategy", result.get("strategy", "—")))
            status = str(row.get("status", "—"))
            runtime = row.get("wall_seconds", row.get("runtime", result.get("wall_seconds")))
            pnl = row.get("pnl", result.get("pnl"))
            table.add_row(key[:24], engine, strategy, f"{runtime:.5f}" if isinstance(runtime, (int, float)) else "—", f"{pnl:.2f}" if isinstance(pnl, (int, float)) else "—", status, key=key)
            if index < 5:
                recent.add_row(key[:24], engine, strategy, status, key=key)

    def sort_history(self) -> None:
        self.history_sort_reverse = not self.history_sort_reverse
        def runtime(row: dict[str, Any]) -> float:
            result = row.get("result") or {}
            nested = result.get("wall_seconds", 0) if isinstance(result, dict) else 0
            return float(row.get("wall_seconds", row.get("runtime", nested)) or 0)
        self.experiment_rows.sort(key=runtime, reverse=self.history_sort_reverse)
        self._render_history()

    def open_selected_history(self) -> None:
        table = self.query_one("#history-table", DataTable)
        if table.row_count == 0:
            self.notify("No experiment is selected.")
            return
        coordinate = table.cursor_coordinate
        row_key = table.coordinate_to_cell_key(coordinate).row_key
        if row_key.value:
            self._open_experiment(row_key.value)

    def _open_experiment(self, experiment_id: str) -> None:
        try:
            from hftlab.experiments import ExperimentRegistry

            experiment = ExperimentRegistry().get(experiment_id)
            result = experiment.get("result", experiment)
            if isinstance(result, str):
                result = json.loads(result)
            self.result = result
            self._render_results(result)
            self.show_page("results")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._show_error(exc)

    @work(thread=True, group="diagnostics", exclusive=True, exit_on_error=False)
    def run_diagnostics(self) -> None:
        try:
            from hftlab.monitoring import diagnose

            value = diagnose()
            self.call_from_thread(self._diagnostics_finished, value)
        except Exception as exc:
            self.call_from_thread(self._diagnostics_failed, str(exc))

    def _diagnostics_finished(self, value: dict[str, Any]) -> None:
        self.diagnostics = value
        log = self.query_one("#doctor-log", RichLog)
        log.clear()
        log.write(render_json(value))
        self.query_one("#home-resources", Static).update(
            f"Python {os.sys.version.split()[0]}  •  Logical CPUs {os.cpu_count() or 'unknown'}\n"
            f"Host {os.name}  •  Native capability details in System Diagnostics\n"
            + render_json(value.get("engines", value.get("native", {})))
        )
        self.query_one("#engine-capabilities", Static).update(render_json(value.get("engines", value.get("native", {}))))

    def _diagnostics_failed(self, message: str) -> None:
        self.query_one("#doctor-log", RichLog).write(message)
        self.query_one("#home-resources", Static).update(f"Resource discovery failed: {message}\nOpen Diagnostics for details.")

    def export_current_report(self) -> None:
        if self.result is None:
            raise ValueError("Run or reopen an experiment before exporting a report.")
        from hftlab.reports import export_report

        path = export_report(self.result, Path(self._input("report-path")))
        self.query_one("#export-status", Static).update(f"Report saved: {path}")
        self.notify(f"Exported {path}")

    def save_preferences(self) -> None:
        preferences = {
            "compact": self.query_one("#compact-mode", Checkbox).value,
            "dataset": self._input("dataset-path"),
            "strategy": self._selected("strategy-select"),
            "engine": self._selected("engine-select"),
            "cpu_mode": self._selected("cpu-mode"),
            "output_directory": self._input("output-directory"),
        }
        temporary = self._prefs_path.with_suffix(".tmp")
        temporary.write_text(render_json(preferences) + "\n", encoding="utf-8")
        temporary.replace(self._prefs_path)
        self.notify("Preferences saved.")

    def _restore_preferences(self) -> None:
        for key, widget in (("dataset", "dataset-path"), ("output_directory", "output-directory")):
            value = self._preferences.get(key)
            if isinstance(value, str):
                self.query_one(f"#{widget}", Input).value = value
        for key, widget, allowed in (
            ("strategy", "strategy-select", {v for _, v in STRATEGIES}),
            ("engine", "engine-select", {v for _, v in ENGINES}),
            ("cpu_mode", "cpu-mode", {v for _, v in CPU_MODES}),
        ):
            value = self._preferences.get(key)
            if value in allowed:
                self.query_one(f"#{widget}", Select).value = value
        self.set_class(bool(self._preferences.get("compact", False)), "compact")
