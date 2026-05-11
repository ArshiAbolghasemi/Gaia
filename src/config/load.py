from __future__ import annotations

import json
from pathlib import Path

from src.config.model import (
    DataConfig,
    DependenceConfig,
    FeatureSelectionConfig,
    OutputConfig,
    PCMCIConfig,
    RunConfig,
)


def load_config(path: Path) -> RunConfig:
    payload = json.loads(path.read_text())

    return RunConfig(
        name=payload["name"],
        data=DataConfig(
            path=Path(payload["data"]["path"]),
            date_column=payload["data"]["date_column"],
            drop_columns=payload["data"].get("drop_columns", []),
        ),
        features=FeatureSelectionConfig(
            include_groups=payload["features"]["include_groups"],
            include_columns=payload["features"].get("include_columns", []),
            exclude_columns=payload["features"].get("exclude_columns", []),
        ),
        dependence=DependenceConfig(
            method=payload["dependence"]["method"],
            params=payload["dependence"].get("params", {}),
        ),
        pcmci=PCMCIConfig(
            **payload.get("pcmci", {}),
        ),
        output=OutputConfig(
            directory=Path(payload["output"]["directory"]),
            run_name=payload["output"].get("run_name"),
            max_links=payload["output"].get("max_links", 10),
            save_tigramite_plots=payload["output"].get("save_tigramite_plots", True),
            save_networkx_plot=payload["output"].get("save_networkx_plot", True),
        ),
    )


def load_config_dir(config_dir: Path) -> list[RunConfig]:
    config_paths = sorted(config_dir.glob("*.json"))
    if not config_paths:
        msg = f"No config files found in {config_dir}"
        raise FileNotFoundError(msg)
    return [load_config(path) for path in config_paths]
