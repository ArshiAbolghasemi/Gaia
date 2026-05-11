from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    path: Path
    date_column: str
    drop_columns: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FeatureSelectionConfig:
    include_groups: list[str]
    include_columns: list[str] = field(default_factory=list)
    exclude_columns: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DependenceConfig:
    method: str = "parcorr"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PCMCIConfig:
    tau_min: int = 1
    tau_max: int = 16
    pc_alpha: float = 0.2
    alpha_level: float = 0.05
    fdr_method: str = "none"
    max_conds_dim: int | None = None
    max_combinations: int = 1
    max_conds_py: int | None = None
    max_conds_px: int | None = None


@dataclass(slots=True)
class OutputConfig:
    directory: Path
    run_name: str | None = None
    max_links: int = 10
    save_tigramite_plots: bool = True
    save_networkx_plot: bool = True


@dataclass(slots=True)
class RunConfig:
    name: str
    data: DataConfig
    features: FeatureSelectionConfig
    dependence: DependenceConfig
    pcmci: PCMCIConfig
    output: OutputConfig


@dataclass(slots=True)
class LinkResult:
    source: str
    target: str
    lag: int
    p_value: float
    effect_size: float


@dataclass(slots=True)
class RunResult:
    config_name: str
    selected_columns: list[str]
    row_count: int
    links: list[LinkResult]
