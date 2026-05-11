from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from src.config.model import DataConfig


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        msg = f"Dataset not found: {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path)


def prepare_numeric_frame(df: pd.DataFrame, data_config: DataConfig) -> pd.DataFrame:
    reduced = df.drop(columns=data_config.drop_columns, errors="ignore")
    numeric = reduced.drop(columns=[data_config.date_column], errors="ignore").apply(
        pd.to_numeric,
        errors="coerce",
    )
    cleaned = numeric.dropna(axis=0, how="any")
    if cleaned.empty:
        msg = f"No rows left after cleaning dataset: {data_config.path}"
        raise ValueError(msg)
    return cast("pd.DataFrame", cleaned)
