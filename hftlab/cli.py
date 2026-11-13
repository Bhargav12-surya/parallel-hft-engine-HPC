"""The headless CLI and default Textual entry point."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from hftlab import APP_NAME, __version__
from hftlab.configuration import CPU_MODES, ENGINES, STRATEGIES, SimulationConfig
from hftlab.utilities import atomic_json, read_json


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, default=str, allow_nan=False))


def _simulation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="JSON/YAML simulation configuration; explicit CLI values override")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--engine", choices=ENGINES)
    parser.add_argument("--cpu-mode", choices=CPU_MODES)
    parser.add_argument("--strategy", choices=STRATEGIES)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--components", help="Comma-separated hybrid components, e.g. cpu,openmp")
    parser.add_argument("--capital", type=float)
    parser.add_argument("--fee-bps", type=float)
    parser.add_argument("--slippage-ticks", type=int)
    parser.add_argument("--latency-ns", type=int)
    parser.add_argument("--position-limit", type=int)
    parser.add_argument("--max-order", type=int)
    parser.add_argument("--max-loss", type=float)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float)


def _config(args: argparse.Namespace) -> SimulationConfig:
    values = {}
    if args.config:
        if args.config.suffix.lower() in (".yaml", ".yml"):
            import yaml
            try:
                values = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            except yaml.YAMLError as error:
                raise ValueError(f"Invalid YAML configuration: {error}") from error
            if not isinstance(values, dict):
                raise ValueError("Configuration must be a mapping")
        else:
            values = read_json(args.config)
    for name in SimulationConfig.__dataclass_fields__:
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value.split(",") if name == "components" else value
    if "dataset" not in values:
        raise ValueError("Provide --dataset or a dataset field in --config")
    return SimulationConfig.from_dict(values).validate()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="hftlab", description=f"{APP_NAME}: offline native simulation and HPC research")
    root.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    commands = root.add_subparsers(dest="command")
    commands.add_parser("doctor", help="Probe actual binaries, hardware, and optional runtimes").add_argument("--json", action="store_true")
    generate = commands.add_parser("generate", help="Stream a reproducible synthetic market dataset")
    generate.add_argument("--output", type=Path, default=Path("data/generated/demo.csv"))
    generate.add_argument("--events", type=int, default=10000)
    generate.add_argument("--instruments", type=int, default=4)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--scenario", choices=("normal", "high-volatility", "high-liquidity", "low-liquidity", "shock", "correlated", "stress", "predictable"), default="normal")
    data = commands.add_parser("data", help="Import, inspect, validate, or list datasets")
    data.add_argument("action", choices=("list", "validate", "inspect", "import"))
    data.add_argument("path", type=Path, nargs="?")
    data.add_argument("--output", type=Path)
    train = commands.add_parser("train", help="Train a chronological logistic or MLP model and export native weights")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("models/reference"))
    train.add_argument("--kind", choices=("logistic", "mlp"), default="mlp")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--backend", choices=("auto", "python", "sklearn", "torch"), default="auto")
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--horizon", type=int, default=4, help="Future events per instrument used only to form labels")
    train.add_argument("--threshold", type=float, default=.00015, help="Future return threshold for BUY/SELL labels")
    train.add_argument("--hidden-size", type=int, default=12, help="MLP ReLU hidden units (1..128)")
    simulate = commands.add_parser("simulate", help="Run a real native backend and persist its experiment")
    _simulation_arguments(simulate)
    simulate.add_argument("--progress", action="store_true", help="Emit native JSONL progress on stderr")
    for name in ("benchmark", "compare"):
        bench = commands.add_parser(name, help="Repeated, correctness-checked native engine comparison")
        _simulation_arguments(bench)
        bench.add_argument("--engines", default="sequential,threaded,openmp")
        bench.add_argument("--all", action="store_true", help="Request every backend and report unavailable engines explicitly")
        bench.add_argument("--repeats", type=int, default=3)
        bench.add_argument("--warmups", type=int, default=1)
        bench.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    scale = commands.add_parser("scaling", help="Strong or weak scaling experiments")
    _simulation_arguments(scale)
    scale.add_argument("--workers", default="1,2,4")
    scale.add_argument("--weak", action="store_true")
    scale.add_argument("--events-per-worker", type=int, default=10000)
    scale.add_argument("--repeats", type=int, default=3)
    scale.add_argument("--output", type=Path, default=Path("reports/scaling.json"))
    report = commands.add_parser("report", help="Export JSON, HTML, CSV, or Markdown from real results")
    report.add_argument("input", type=Path, nargs="?")
    report.add_argument("--experiment")
    report.add_argument("--output", type=Path, default=Path("reports/experiment.html"))
    experiments = commands.add_parser("experiments", help="Persisted experiment registry")
    experiments.add_argument("action", choices=("list", "show", "repeat", "delete", "export"), default="list", nargs="?")
    experiments.add_argument("id", nargs="?")
    experiments.add_argument("--search", default="")
    experiments.add_argument("--sort", choices=("date", "runtime", "pnl"), default="date")
    experiments.add_argument("--output", type=Path, default=Path("reports/experiment.json"))
    experiments.add_argument("--yes", action="store_true", help="Confirm registry deletion (result files retained)")
    models = commands.add_parser("models", help="List trained model metadata and validate artifacts")
    models.add_argument("--root", type=Path, default=Path("models"))
    models.add_argument("--validate", type=Path)
    demo = commands.add_parser("demo", help="Generate, train, compare, simulate, and export one reproducible demonstration")
    demo.add_argument("--events", type=int, default=10000)
    demo.add_argument("--epochs", type=int, default=15)
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    commands.add_parser("build", help="Build the available native targets using scripts/build.py")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command is None:
            from hftlab.tui import run_app
            run_app()
        elif args.command == "doctor":
            from hftlab.monitoring import diagnose
            _print(diagnose())
        elif args.command == "generate":
            from hftlab.datasets import generate_dataset
            _print(generate_dataset(args.output, events=args.events, instruments=args.instruments, seed=args.seed, scenario=args.scenario))
        elif args.command == "data":
            from hftlab.datasets import import_dataset, inspect_dataset, validate_dataset
            if args.action == "list":
                _print([str(path) for path in sorted((args.path or Path("data")).rglob("*.csv"))])
            elif args.path is None:
                raise ValueError("This data action requires a path")
            elif args.action == "import":
                if args.output is None:
                    raise ValueError("data import requires --output")
                _print(import_dataset(args.path, args.output))
            else:
                _print((validate_dataset if args.action == "validate" else inspect_dataset)(args.path))
        elif args.command == "train":
            from hftlab.ai import train_model
            _print(train_model(args.dataset, args.output, kind=args.kind, epochs=args.epochs, seed=args.seed,
                               backend=args.backend, device=args.device, horizon=args.horizon,
                               threshold=args.threshold, hidden_size=args.hidden_size))
        elif args.command == "simulate":
            from hftlab.orchestration import JobManager
            callback = (lambda event: print(json.dumps(event), file=sys.stderr, flush=True)) if args.progress else None
            _print(JobManager().run(_config(args), on_event=callback))
        elif args.command in ("benchmark", "compare"):
            from hftlab.benchmarks import benchmark
            from hftlab.reports import export_report
            result = benchmark(_config(args), repeats=args.repeats, engines=list(ENGINES) if args.all else args.engines.split(","),
                               output=args.output, warmups=args.warmups)
            export_report(result, args.output.with_suffix(".html"))
            _print(result)
            if any(row["status"] in ("FAILED", "INCORRECT") for row in result["engines"]):
                return 1
        elif args.command == "scaling":
            from hftlab.benchmarks import scaling
            result = scaling(_config(args), [int(value) for value in args.workers.split(",")], weak=args.weak,
                             repeats=args.repeats, events_per_worker=args.events_per_worker, output=args.output)
            _print(result)
            if result["status"] != "COMPLETED":
                return 1
        elif args.command == "report":
            from hftlab.experiments import ExperimentRegistry
            from hftlab.reports import export_report
            value = ExperimentRegistry().get(args.experiment) if args.experiment else read_json(args.input) if args.input else None
            if value is None:
                raise ValueError("Provide an input JSON file or --experiment ID")
            _print({"report": str(export_report(value, args.output))})
        elif args.command == "experiments":
            from hftlab.experiments import ExperimentRegistry
            from hftlab.orchestration import JobManager
            from hftlab.reports import export_report
            registry = ExperimentRegistry()
            if args.action == "list":
                _print(registry.list(search=args.search, sort=args.sort))
            else:
                if not args.id:
                    raise ValueError("Provide an experiment ID")
                value = registry.get(args.id)
                if args.action == "show":
                    _print(value)
                elif args.action == "repeat":
                    _print(JobManager(registry).run(SimulationConfig.from_dict(value["config"])))
                elif args.action == "export":
                    _print({"report": str(export_report(value, args.output))})
                elif args.yes:
                    registry.delete(args.id)
                    _print({"deleted_from_registry": args.id})
                else:
                    raise ValueError("Registry deletion requires --yes")
        elif args.command == "models":
            from hftlab.ai import list_models, validate_model
            if args.validate:
                _print(validate_model(args.validate))
            else:
                _print(list_models(args.root))
        elif args.command == "demo":
            _print(run_demo(args.output, args.events, args.epochs))
        elif args.command == "build":
            import subprocess
            from hftlab.engines import PROJECT_ROOT
            return subprocess.call([sys.executable, str(PROJECT_ROOT / "scripts/build.py")])
        return 0
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, TimeoutError, ImportError) as error:
        print(f"{APP_NAME}: {error}", file=sys.stderr)
        return 1


def run_demo(output: Path, events: int = 10000, epochs: int = 15) -> dict:
    from hftlab.ai import export_holdout, train_model
    from hftlab.benchmarks import benchmark
    from hftlab.datasets import generate_dataset
    from hftlab.monitoring import diagnose
    from hftlab.orchestration import JobManager
    from hftlab.reports import export_report
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = output / "market.csv"
    data = generate_dataset(dataset, events=events, instruments=4, seed=42, scenario="predictable")
    config = SimulationConfig(dataset=dataset, output_dir=output / "results")
    comparison = benchmark(config, repeats=3, engines=list(ENGINES), output=output / "benchmark.json")
    if comparison["status"] != "COMPLETED":
        raise RuntimeError("Demo benchmark failed its execution/correctness checks; inspect benchmark.json")
    logistic = train_model(dataset, output / "logistic", kind="logistic", epochs=epochs, seed=42)
    model = train_model(dataset, output / "mlp", kind="mlp", epochs=epochs, seed=42)
    holdout = export_holdout(dataset, model["native_model"], output / "heldout.csv")
    test_config = replace(config, dataset=output / "heldout.csv")
    ai = JobManager().run(replace(test_config, strategy="ai", model=Path(model["native_model"])))
    trading_comparison = {"mlp": ai,
                          "logistic": JobManager().run(replace(test_config, strategy="ai", model=Path(logistic["native_model"])))}
    for strategy in ("momentum", "mean-reversion", "hold"):
        trading_comparison[strategy] = JobManager().run(replace(test_config, strategy=strategy))
    summary = {"schema_version": 1, "warning": "Predictable synthetic educational data; not evidence of real-world profitability.",
               "diagnostics": diagnose(), "dataset": data, "benchmark": comparison,
               "logistic": logistic, "mlp": model, "ai_simulation": ai,
               "heldout": holdout, "trading_comparison": trading_comparison}
    atomic_json(output / "demo.json", summary)
    export_report(summary, output / "demo.html")
    export_report(comparison, output / "benchmark.html")
    return summary
