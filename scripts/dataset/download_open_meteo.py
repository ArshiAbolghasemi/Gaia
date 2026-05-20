"""US Climate Data Downloader using the Open-Meteo Historical API.

Variables : Soil Moisture (0-7 cm), Cloud Cover, Humidity
Period    : 2009-01-01 -> 2026-05-15
Locations : 45 major US cities (all states/regions covered)
Output    : us_climate_weekly.csv  +  us_climate_monthly.csv

Requirements:
    pip install requests pandas tqdm

Run:
    python download_us_climate.py
"""

import logging
import sys
import time
from typing import cast

import pandas as pd
import requests
from tqdm import tqdm

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("download_us_climate.log"),
    ],
)
logger = logging.getLogger(__name__)

# 45 representative US cities
US_LOCATIONS = [
    ("New York", "NY", 40.71, -74.01),
    ("Los Angeles", "CA", 34.05, -118.25),
    ("Chicago", "IL", 41.88, -87.63),
    ("Houston", "TX", 29.76, -95.37),
    ("Phoenix", "AZ", 33.45, -112.07),
    ("Philadelphia", "PA", 39.95, -75.16),
    ("San Antonio", "TX", 29.42, -98.49),
    ("San Diego", "CA", 32.72, -117.16),
    ("Dallas", "TX", 32.78, -96.80),
    ("San Jose", "CA", 37.34, -121.89),
    ("Austin", "TX", 30.27, -97.74),
    ("Jacksonville", "FL", 30.33, -81.66),
    ("Fort Worth", "TX", 32.75, -97.33),
    ("Columbus", "OH", 39.96, -82.99),
    ("Charlotte", "NC", 35.23, -80.84),
    ("Indianapolis", "IN", 39.77, -86.16),
    ("San Francisco", "CA", 37.77, -122.42),
    ("Seattle", "WA", 47.61, -122.33),
    ("Denver", "CO", 39.74, -104.98),
    ("Nashville", "TN", 36.17, -86.78),
    ("Oklahoma City", "OK", 35.47, -97.52),
    ("El Paso", "TX", 31.76, -106.49),
    ("Washington DC", "DC", 38.91, -77.04),
    ("Las Vegas", "NV", 36.17, -115.14),
    ("Louisville", "KY", 38.25, -85.76),
    ("Memphis", "TN", 35.15, -90.05),
    ("Portland", "OR", 45.52, -122.68),
    ("Baltimore", "MD", 39.29, -76.61),
    ("Milwaukee", "WI", 43.04, -87.91),
    ("Albuquerque", "NM", 35.08, -106.65),
    ("Tucson", "AZ", 32.22, -110.97),
    ("Fresno", "CA", 36.74, -119.77),
    ("Sacramento", "CA", 38.58, -121.49),
    ("Kansas City", "MO", 39.10, -94.58),
    ("Mesa", "AZ", 33.42, -111.83),
    ("Atlanta", "GA", 33.75, -84.39),
    ("Omaha", "NE", 41.26, -95.94),
    ("Colorado Springs", "CO", 38.83, -104.82),
    ("Raleigh", "NC", 35.78, -78.64),
    ("Miami", "FL", 25.77, -80.19),
    ("Minneapolis", "MN", 44.98, -93.27),
    ("Wichita", "KS", 37.69, -97.34),
    ("New Orleans", "LA", 29.95, -90.07),
    ("Anchorage", "AK", 61.22, -149.90),
    ("Honolulu", "HI", 21.31, -157.86),
]

# API settings
API_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = "soil_moisture_0_to_7cm,cloud_cover,relative_humidity_2m"
START_DATE = "2009-01-01"
END_DATE = "2026-05-15"

# Aggregation spec: mean, min, max for each variable.
AGGS = {
    "soil_moisture_0_7cm_m3m3": ["mean", "min", "max"],
    "cloud_cover_pct": ["mean", "min", "max"],
    "relative_humidity_2m_pct": ["mean", "min", "max"],
}

_WEEKLY_META = [
    "location",
    "state",
    "latitude",
    "longitude",
    "year",
    "week",
    "week_start_monday",
]
_MONTHLY_META = [
    "location",
    "state",
    "latitude",
    "longitude",
    "year",
    "month",
    "month_name",
    "month_start",
]


def fetch_location(name: str, state: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetch daily climate data for one location from the Open-Meteo Historical API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": VARIABLES,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": "America/Chicago",
    }
    resp = requests.get(API_URL, params=params, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "soil_moisture_0_7cm_m3m3": daily.get("soil_moisture_0_to_7cm"),
            "cloud_cover_pct": daily.get("cloud_cover"),
            "relative_humidity_2m_pct": daily.get("relative_humidity_2m"),
        }
    )
    frame.insert(0, "location", name)
    frame.insert(1, "state", state)
    frame.insert(2, "latitude", lat)
    frame.insert(3, "longitude", lon)
    return frame


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a multi-level column index produced by groupby + agg."""
    df.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in df.columns]
    return df


def _fetch_all_locations() -> list[pd.DataFrame]:
    """Download daily data for every US location, returning one DataFrame per city."""
    frames: list[pd.DataFrame] = []
    total = len(US_LOCATIONS)
    iterator = tqdm(US_LOCATIONS, unit="city")

    for item in iterator:
        name, state, lat, lon = item
        label = f"{name}, {state}"
        idx = US_LOCATIONS.index(item) + 1

        tqdm.write(f"  Fetching [{idx:02d}/{total}] {label}...")

        try:
            location_df = fetch_location(name, state, lat, lon)
            frames.append(location_df)
            logger.info(
                "[%02d/%d] OK  %s  -  %d days fetched", idx, total, label, len(location_df)
            )
        except Exception:
            logger.exception("[%02d/%d] FAIL  %s", idx, total, label)

        time.sleep(0.35)

    return frames


def _build_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily rows into Monday-anchored ISO weeks."""
    logger.info("Building weekly aggregates...")
    weekly_df = (
        daily_df.groupby(
            [
                "location",
                "state",
                "latitude",
                "longitude",
                pd.Grouper(key="date", freq="W-MON"),
            ],
            observed=True,
        )
        .agg(AGGS)
        .reset_index()
    )
    weekly_df = flatten_columns(weekly_df)
    weekly_df = weekly_df.rename(columns={"date": "week_start_monday"})
    weekly_df["year"] = weekly_df["week_start_monday"].dt.year
    weekly_df["week"] = weekly_df["week_start_monday"].dt.isocalendar().week.astype(int)
    extra = [c for c in weekly_df.columns if c not in _WEEKLY_META]
    weekly_df = weekly_df[_WEEKLY_META + extra]
    logger.info("Weekly aggregation complete  -  %d rows", len(weekly_df))
    return cast("pd.DataFrame", weekly_df)


def _build_monthly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily rows into calendar months."""
    logger.info("Building monthly aggregates...")
    monthly_df = (
        daily_df.groupby(
            [
                "location",
                "state",
                "latitude",
                "longitude",
                pd.Grouper(key="date", freq="MS"),
            ],
            observed=True,
        )
        .agg(AGGS)
        .reset_index()
    )
    monthly_df = flatten_columns(monthly_df)
    monthly_df = monthly_df.rename(columns={"date": "month_start"})
    monthly_df["year"] = monthly_df["month_start"].dt.year
    monthly_df["month"] = monthly_df["month_start"].dt.month
    monthly_df["month_name"] = monthly_df["month_start"].dt.strftime("%B")
    extra = [c for c in monthly_df.columns if c not in _MONTHLY_META]
    monthly_df = monthly_df[_MONTHLY_META + extra]
    logger.info("Monthly aggregation complete  -  %d rows", len(monthly_df))
    return cast("pd.DataFrame", monthly_df)


def _save_outputs(weekly_df: pd.DataFrame, monthly_df: pd.DataFrame) -> None:
    """Write weekly and monthly CSVs to the working directory."""
    weekly_path = "us_climate_weekly.csv"
    monthly_path = "us_climate_monthly.csv"
    weekly_df.to_csv(weekly_path, index=False)
    logger.info("Saved weekly CSV  ->  %s  (%d rows)", weekly_path, len(weekly_df))
    monthly_df.to_csv(monthly_path, index=False)
    logger.info("Saved monthly CSV ->  %s  (%d rows)", monthly_path, len(monthly_df))


def main() -> None:
    """Entry point: fetch, aggregate, and save US climate data."""
    logger.info("=" * 58)
    logger.info("  US Climate Downloader  -  Open-Meteo Historical API")
    logger.info("=" * 58)
    logger.info("Period    : %s  ->  %s", START_DATE, END_DATE)
    logger.info("Variables : Soil Moisture | Cloud Cover | Humidity")
    logger.info("Locations : %d US cities", len(US_LOCATIONS))
    logger.info("=" * 58)

    frames = _fetch_all_locations()

    if not frames:
        logger.critical("No data was fetched. Check your internet connection.")
        sys.exit(1)

    logger.info("Combining data from %d locations...", len(frames))
    daily_df = pd.concat(frames, ignore_index=True)
    daily_df = daily_df.sort_values(["location", "date"])
    logger.info("Total daily rows: %d", len(daily_df))

    weekly_df = _build_weekly(daily_df)
    monthly_df = _build_monthly(daily_df)

    _save_outputs(weekly_df, monthly_df)

    logger.info("=" * 58)
    logger.info("ALL DONE")
    logger.info("=" * 58)
    logger.info(
        "Column reference:\n"
        "  soil_moisture_0_7cm_m3m3_mean/min/max  [m3/m3]  0=dry ~0.6=saturated\n"
        "  cloud_cover_pct_mean/min/max            [%%]      0=clear 100=overcast\n"
        "  relative_humidity_2m_pct_mean/min/max   [%%]      0=dry 100=saturated"
    )


if __name__ == "__main__":
    main()
