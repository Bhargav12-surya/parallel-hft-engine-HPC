"""Versioned, validated simulation configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from hftlab.utilities import finite_number

ENGINES = ("sequential", "threaded", "openmp", "mpi", "cuda", "hybrid")
STRATEGIES = ("market-making", "momentum", "stat-arb", "mean-reversion", "ai", "hold")
CPU_MODES = ("single-single", "single-multi", "multi-single", "multi-multi")
COMPONENTS = ("cpu", "openmp", "mpi", "cuda")


@dataclass
class SimulationConfig:
    dataset: Path
    engine: str = "sequential"
    cpu_mode: str = "single-single"
    strategy: str = "momentum"
    threads: int = 4
    components: list[str] = field(default_factory=lambda: ["cpu", "openmp"])
    capital: float = 100000.0
    fee_bps: float = 1.0
    slippage_ticks: int = 0
    latency_ns: int = 0
    position_limit: int = 100
    max_order: int = 10
    max_loss: float = 10000.0
    model: Path | None = None
    output_dir: Path = Path("results")
    timeout: float = 300.0
    schema_version: int = 1

    def validate(self) -> "SimulationConfig":
        self.dataset = Path(self.dataset).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        if not self.dataset.is_file():
            raise ValueError(f"Dataset does not exist: {self.dataset}")
        if self.engine not in ENGINES:
            raise ValueError(f"Unknown engine {self.engine!r}; choose {', '.join(ENGINES)}")
        if self.cpu_mode not in CPU_MODES:
            raise ValueError(f"Unknown CPU mode: {self.cpu_mode}")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        if not isinstance(self.threads, int) or not 1 <= self.threads <= 1024:
            raise ValueError("threads must be an integer between 1 and 1024")
        if self.schema_version != 1:
            raise ValueError("Unsupported simulation schema version")
        finite_number(self.capital, "capital", minimum=0.01)
        finite_number(self.fee_bps, "fee_bps", maximum=10000)
        finite_number(self.max_loss, "max_loss")
        finite_number(self.timeout, "timeout", minimum=0.01, maximum=86400)
        for name, lower, upper in (("slippage_ticks", 0, 1000000), ("latency_ns", 0, 10**15),
                                   ("position_limit", 1, 10**9), ("max_order", 1, 10**9)):
            value = getattr(self, name)
            if not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        if self.max_order > self.position_limit:
            raise ValueError("max_order cannot exceed position_limit")
        if self.engine == "hybrid":
            if len(set(self.components)) != len(self.components) or len(self.components) < 2:
                raise ValueError("Hybrid requires at least two distinct components")
            if any(part not in COMPONENTS for part in self.components):
                raise ValueError("Unknown hybrid component")
        if self.strategy == "ai":
            if self.model is None or not Path(self.model).is_file():
                raise ValueError("AI strategy requires an existing exported native model (--model)")
            self.model = Path(self.model).resolve()
        return self

    def to_dict(self) -> dict:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, value: dict) -> "SimulationConfig":
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
        return cls(**value)
