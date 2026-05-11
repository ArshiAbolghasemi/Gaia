from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from config.model import DataConfig, FeatureSelectionConfig

FEATURE_GROUPS: dict[str, list[str]] = {
    "rv": [
        "__weekly_rv__",
    ],
    "macro": [
        "DJIA_Index",
        "WTI_Index",
        "Broad_Dollar_index",
        "Stock_Uncertainty",
        "epu_index",
    ],
    "news": [
        "Text_Climate_Anomaly",
        "frbsf_sentiment",
    ],
    "climate": [
        "ssta_elino",
        "ssta_lanina",
        "SOI_index",
        "NAO_index",
        "tmax_hot_in_planting",
        "tmax_hot_in_harvesting",
        "tmax_very_hot_in_planting",
        "tmax_very_hot_in_harvesting",
        "tmin_cold_in_planting",
        "tmin_cold_in_harvesting",
        "tmin_very_cold_in_planting",
        "tmin_very_cold_in_harvesting",
        "awnd_moderate_high_wind_in_planting",
        "awnd_moderate_high_wind_in_harvesting",
        "awnd_extreme_high_wind_in_planting",
        "awnd_extreme_high_wind_in_harvesting",
        "spi_7d_very_wet_in_planting",
        "spi_7d_very_wet_in_harvesting",
        "spi_7d_extreme_wet_in_planting",
        "spi_7d_extreme_wet_in_harvesting",
        "spi_7d_very_dry_in_planting",
        "spi_7d_very_dry_in_harvesting",
        "spi_7d_extreme_dry_in_planting",
        "spi_7d_extreme_dry_in_harvesting",
        "spi_1m_very_wet_in_planting",
        "spi_1m_very_wet_in_harvesting",
        "spi_1m_extreme_wet_in_planting",
        "spi_1m_extreme_wet_in_harvesting",
        "spi_1m_very_dry_in_planting",
        "spi_1m_very_dry_in_harvesting",
        "spi_1m_extreme_dry_in_planting",
        "spi_1m_extreme_dry_in_harvesting",
        "spi_3m_very_wet_in_planting",
        "spi_3m_very_wet_in_harvesting",
        "spi_3m_extreme_wet_in_planting",
        "spi_3m_extreme_wet_in_harvesting",
        "spi_3m_very_dry_in_planting",
        "spi_3m_very_dry_in_harvesting",
        "spi_3m_extreme_dry_in_planting",
        "spi_3m_extreme_dry_in_harvesting",
        "pdsi_very_wet_in_planting",
        "pdsi_very_wet_in_harvesting",
        "pdsi_extreme_wet_in_planting",
        "pdsi_extreme_wet_in_harvesting",
        "pdsi_extreme_drought_in_planting",
        "pdsi_extreme_drought_in_harvesting",
        "pdsi_severe_drought_in_planting",
        "pdsi_severe_drought_in_harvesting",
        "co2_extreme_in_planting",
        "co2_extreme_in_harvesting",
    ],
    "climate_raw": [
        "prcp",
        "awnd",
        "tmin",
        "tmax",
        "co2",
        "pdsi",
    ],
}


def get_available_groups() -> list[str]:
    return list(FEATURE_GROUPS.keys())


RV_ALIAS = "__weekly_rv__"


def _resolve_group_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    rv_columns = [column for column in df.columns if column.endswith("_weekly_rv")]
    if not rv_columns and any(RV_ALIAS in columns for columns in FEATURE_GROUPS.values()):
        msg = "No '*_weekly_rv' column found in dataset for the 'rv' group"
        raise ValueError(msg)

    resolved: dict[str, list[str]] = {}
    for group_name, columns in FEATURE_GROUPS.items():
        group_columns: list[str] = []
        for column in columns:
            if column == RV_ALIAS:
                group_columns.extend(rv_columns)
            else:
                group_columns.append(column)
        resolved[group_name] = group_columns
    return resolved


def select_feature(
    df: pd.DataFrame,
    data_config: DataConfig,
    feature_config: FeatureSelectionConfig,
) -> pd.DataFrame:
    resolved_groups = _resolve_group_columns(df)
    requested_columns: list[str] = []

    for group_name in feature_config.include_groups:
        if group_name not in resolved_groups:
            known = ", ".join(sorted(resolved_groups))
            msg = f"Unknown feature group '{group_name}'. Known groups: {known}"
            raise ValueError(msg)
        requested_columns.extend(resolved_groups[group_name])

    requested_columns.extend(feature_config.include_columns)

    excluded = set(feature_config.exclude_columns)
    requested_columns = [column for column in requested_columns if column not in excluded]

    selected_columns = list(
        OrderedDict.fromkeys(column for column in requested_columns if column)
    )
    missing = [column for column in selected_columns if column not in df.columns]
    if missing:
        msg = f"Requested columns are missing from dataset: {missing}"
        raise ValueError(msg)

    keep_columns = [data_config.date_column, *selected_columns]
    return df.loc[:, keep_columns]
