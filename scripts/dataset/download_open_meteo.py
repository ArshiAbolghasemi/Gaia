"""US Climate Data Downloader using the Open-Meteo Historical API.

Variables : Soil Moisture (0-7 cm), Cloud Cover, Humidity
Period    : 2009-01-01 -> 2026-05-15
Locations : 50 US states (one representative city each)
Output    : us_climate_weekly.csv  +  us_climate_monthly.csv

Requirements:
    pip install requests pandas tqdm tenacity

Run:
    python download_us_climate.py
"""

import logging
import sys
import time
from http import HTTPStatus

import pandas as pd
import requests
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
)

try:
    from tqdm import tqdm

    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
US_LOCATIONS = [
    ("Birmingham", "AL", 33.52, -86.81, "America/Chicago"),
    ("Anchorage", "AK", 61.22, -149.90, "America/Anchorage"),
    ("Phoenix", "AZ", 33.45, -112.07, "America/Phoenix"),
    ("Little Rock", "AR", 34.75, -92.29, "America/Chicago"),
    ("Los Angeles", "CA", 34.05, -118.25, "America/Los_Angeles"),
    ("Denver", "CO", 39.74, -104.98, "America/Denver"),
    ("Hartford", "CT", 41.76, -72.68, "America/New_York"),
    ("Wilmington", "DE", 39.74, -75.54, "America/New_York"),
    ("Miami", "FL", 25.77, -80.19, "America/New_York"),
    ("Atlanta", "GA", 33.75, -84.39, "America/New_York"),
    ("Honolulu", "HI", 21.31, -157.86, "Pacific/Honolulu"),
    ("Boise", "ID", 43.61, -116.20, "America/Boise"),
    ("Chicago", "IL", 41.88, -87.63, "America/Chicago"),
    ("Indianapolis", "IN", 39.77, -86.16, "America/Indiana/Indianapolis"),
    ("Des Moines", "IA", 41.59, -93.62, "America/Chicago"),
    ("Wichita", "KS", 37.69, -97.34, "America/Chicago"),
    ("Louisville", "KY", 38.25, -85.76, "America/Kentucky/Louisville"),
    ("New Orleans", "LA", 29.95, -90.07, "America/Chicago"),
    ("Portland", "ME", 43.66, -70.26, "America/New_York"),
    ("Baltimore", "MD", 39.29, -76.61, "America/New_York"),
    ("Boston", "MA", 42.36, -71.06, "America/New_York"),
    ("Detroit", "MI", 42.33, -83.05, "America/Detroit"),
    ("Minneapolis", "MN", 44.98, -93.27, "America/Chicago"),
    ("Jackson", "MS", 32.30, -90.18, "America/Chicago"),
    ("Kansas City", "MO", 39.10, -94.58, "America/Chicago"),
    ("Billings", "MT", 45.78, -108.50, "America/Denver"),
    ("Omaha", "NE", 41.26, -95.94, "America/Chicago"),
    ("Las Vegas", "NV", 36.17, -115.14, "America/Los_Angeles"),
    ("Manchester", "NH", 42.99, -71.46, "America/New_York"),
    ("Newark", "NJ", 40.73, -74.17, "America/New_York"),
    ("Albuquerque", "NM", 35.08, -106.65, "America/Denver"),
    ("New York", "NY", 40.71, -74.01, "America/New_York"),
    ("Charlotte", "NC", 35.23, -80.84, "America/New_York"),
    ("Fargo", "ND", 46.88, -96.79, "America/Chicago"),
    ("Columbus", "OH", 39.96, -82.99, "America/New_York"),
    ("Oklahoma City", "OK", 35.47, -97.52, "America/Chicago"),
    ("Portland", "OR", 45.52, -122.68, "America/Los_Angeles"),
    ("Philadelphia", "PA", 39.95, -75.16, "America/New_York"),
    ("Providence", "RI", 41.82, -71.42, "America/New_York"),
    ("Columbia", "SC", 34.00, -81.03, "America/New_York"),
    ("Sioux Falls", "SD", 43.55, -96.73, "America/Chicago"),
    ("Nashville", "TN", 36.17, -86.78, "America/Chicago"),
    ("Houston", "TX", 29.76, -95.37, "America/Chicago"),
    ("Salt Lake City", "UT", 40.76, -111.89, "America/Denver"),
    ("Burlington", "VT", 44.48, -73.21, "America/New_York"),
    ("Richmond", "VA", 37.54, -77.43, "America/New_York"),
    ("Seattle", "WA", 47.61, -122.33, "America/Los_Angeles"),
    ("Charleston", "WV", 38.35, -81.63, "America/New_York"),
    ("Milwaukee", "WI", 43.04, -87.91, "America/Chicago"),
    ("Cheyenne", "WY", 41.14, -104.82, "America/Denver"),
    ("Washington DC", "DC", 38.91, -77.04, "America/New_York"),
]

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
API_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = (
    "soil_moisture_0_to_7cm_mean,"
    "cloud_cover_mean,"
    "relative_humidity_2m_mean,"
    "relative_humidity_2m_max,"
    "relative_humidity_2m_min,"
    "cloud_cover_max,"
    "cloud_cover_min"
)
START_DATE = "2009-01-01"
END_DATE = "2026-05-15"

# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------
RETRY_AFTER_DEFAULT = 60  # seconds to wait when Retry-After header is absent
MAX_ATTEMPTS = 6  # total attempts (1 original + 5 retries)
MAX_BACKOFF_SECONDS = 120
TRANSIENT_STATUS_CODES = {
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}


class RateLimitError(Exception):
    """Raised when the API returns HTTP 429, carrying the requested wait time."""

    def __init__(self, wait: int) -> None:
        super().__init__(f"Rate limited — retry after {wait}s")
        self.wait = wait


class TransientError(Exception):
    """Raised on 5xx or network-level failures that are worth retrying."""


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (RateLimitError, TransientError))


def _before_sleep(retry_state: RetryCallState) -> None:
    """Log the upcoming retry attempt and wait duration."""
    exc = retry_state.outcome.exception()
    wait = retry_state.next_action.sleep  # seconds tenacity will sleep
    logger.warning(
        "Retry %d/%d in %.0fs — %s",
        retry_state.attempt_number,
        MAX_ATTEMPTS - 1,
        wait,
        exc,
    )


def _wait_strategy(retry_state: RetryCallState) -> float:
    """Return the next retry delay.

    Use `Retry-After` from a `RateLimitError` when available; otherwise fall
    back to exponential back-off capped at 120 seconds.
    """
    exc = retry_state.outcome.exception()
    if isinstance(exc, RateLimitError):
        return min(exc.wait, MAX_BACKOFF_SECONDS)
    # Exponential: 2^attempt, capped at 120 s
    return min(2**retry_state.attempt_number, MAX_BACKOFF_SECONDS)


# ---------------------------------------------------------------------------
# Core fetch with tenacity retry
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception(_is_retryable),
    wait=_wait_strategy,
    stop=stop_after_attempt(MAX_ATTEMPTS),
    before_sleep=_before_sleep,
    reraise=True,
)
def _fetch_raw(params: dict) -> dict:
    """Fetch daily payload data from Open-Meteo.

    Raises:
        RateLimitError: If the API responds with HTTP 429.
        TransientError: If a network error or retryable 5xx occurs.
        requests.HTTPError: If a non-retryable 4xx response occurs.

    """
    try:
        resp = requests.get(API_URL, params=params, timeout=90)
    except requests.exceptions.RequestException as exc:
        message = f"Network error: {exc}"
        raise TransientError(message) from exc

    if resp.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        wait = int(resp.headers.get("Retry-After", RETRY_AFTER_DEFAULT))
        raise RateLimitError(wait)

    if resp.status_code in TRANSIENT_STATUS_CODES:
        message = f"HTTP {resp.status_code}"
        raise TransientError(message)

    resp.raise_for_status()  # propagate any other 4xx immediately
    return resp.json().get("daily", {})


def fetch_location(
    name: str, state: str, lat: float, lon: float, timezone: str
) -> pd.DataFrame:
    """Fetch daily climate data for one location, with automatic retries."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": DAILY_VARS,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": timezone,
    }
    daily = _fetch_raw(params)

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "soil_moisture_0_7cm_m3m3": daily.get("soil_moisture_0_to_7cm_mean"),
            "cloud_cover_mean_pct": daily.get("cloud_cover_mean"),
            "cloud_cover_max_pct": daily.get("cloud_cover_max"),
            "cloud_cover_min_pct": daily.get("cloud_cover_min"),
            "relative_humidity_mean_pct": daily.get("relative_humidity_2m_mean"),
            "relative_humidity_max_pct": daily.get("relative_humidity_2m_max"),
            "relative_humidity_min_pct": daily.get("relative_humidity_2m_min"),
        }
    )
    frame.insert(0, "location", name)
    frame.insert(1, "state", state)
    frame.insert(2, "latitude", lat)
    frame.insert(3, "longitude", lon)
    return frame


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in df.columns]
    return df


def build_aggregates(
    daily_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggs = {
        "soil_moisture_0_7cm_m3m3": ["mean", "min", "max"],
        "cloud_cover_mean_pct": ["mean"],
        "cloud_cover_max_pct": ["max"],
        "cloud_cover_min_pct": ["min"],
        "relative_humidity_mean_pct": ["mean"],
        "relative_humidity_max_pct": ["max"],
        "relative_humidity_min_pct": ["min"],
    }
    column_renames = {
        "soil_moisture_0_7cm_m3m3_mean": "soil_moisture_mean_m3m3",
        "soil_moisture_0_7cm_m3m3_min": "soil_moisture_min_m3m3",
        "soil_moisture_0_7cm_m3m3_max": "soil_moisture_max_m3m3",
        "cloud_cover_mean_pct_mean": "cloud_cover_mean_pct",
        "cloud_cover_max_pct_max": "cloud_cover_max_pct",
        "cloud_cover_min_pct_min": "cloud_cover_min_pct",
        "relative_humidity_mean_pct_mean": "humidity_mean_pct",
        "relative_humidity_max_pct_max": "humidity_max_pct",
        "relative_humidity_min_pct_min": "humidity_min_pct",
    }
    group_keys = ["location", "state", "latitude", "longitude"]

    logger.info("Building weekly aggregates...")
    weekly = (
        daily_df.groupby([*group_keys, pd.Grouper(key="date", freq="W-MON")], observed=True)
        .agg(aggs)
        .reset_index()
    )
    weekly = flatten_columns(weekly)
    weekly = weekly.rename(columns={"date": "week_start_monday", **column_renames})
    weekly["year"] = weekly["week_start_monday"].dt.year
    weekly["week"] = weekly["week_start_monday"].dt.isocalendar().week.astype(int)
    front_w = [
        "location",
        "state",
        "latitude",
        "longitude",
        "year",
        "week",
        "week_start_monday",
    ]
    weekly = weekly[[*front_w, *[c for c in weekly.columns if c not in front_w]]]
    logger.info("Weekly rows: %d", len(weekly))

    # Monthly
    logger.info("Building monthly aggregates...")
    monthly = (
        daily_df.groupby([*group_keys, pd.Grouper(key="date", freq="MS")], observed=True)
        .agg(aggs)
        .reset_index()
    )
    monthly = flatten_columns(monthly)
    monthly = monthly.rename(columns={"date": "month_start", **column_renames})
    monthly["year"] = monthly["month_start"].dt.year
    monthly["month"] = monthly["month_start"].dt.month
    monthly["month_name"] = monthly["month_start"].dt.strftime("%B")
    front_m = [
        "location",
        "state",
        "latitude",
        "longitude",
        "year",
        "month",
        "month_name",
        "month_start",
    ]
    monthly = monthly[[*front_m, *[c for c in monthly.columns if c not in front_m]]]
    logger.info("Monthly rows: %d", len(monthly))

    return weekly, monthly


# ---------------------------------------------------------------------------
# Fetch loop
# ---------------------------------------------------------------------------
def _fetch_all_locations() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    total = len(US_LOCATIONS)
    iterator = tqdm(US_LOCATIONS, unit="city") if USE_TQDM else US_LOCATIONS

    for item in iterator:
        name, state, lat, lon, tz = item
        label = f"{name}, {state}"
        idx = US_LOCATIONS.index(item) + 1

        if USE_TQDM:
            tqdm.write(f"  Fetching [{idx:02d}/{total}] {label}...")
        else:
            logger.info("[%02d/%d] Fetching %s ...", idx, total, label)

        try:
            location_frame = fetch_location(name, state, lat, lon, tz)
            frames.append(location_frame)
            logger.info(
                "[%02d/%d] OK   %s  —  %d days",
                idx,
                total,
                label,
                len(location_frame),
            )
        except Exception:
            logger.exception(
                "[%02d/%d] FAIL  %s  (all retries exhausted)", idx, total, label
            )

        time.sleep(0.35)  # polite inter-request pause

    return frames


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("  US Climate Downloader  —  Open-Meteo Historical API")
    logger.info("=" * 60)
    logger.info("Period    : %s  ->  %s", START_DATE, END_DATE)
    logger.info("Variables : Soil Moisture | Cloud Cover | Humidity (daily)")
    logger.info("Locations : %d  (50 states + DC)", len(US_LOCATIONS))
    logger.info("Retries   : up to %d attempts per city (tenacity)", MAX_ATTEMPTS)
    logger.info("=" * 60)

    frames = _fetch_all_locations()

    if not frames:
        logger.critical("No data was fetched. Check your internet connection.")
        sys.exit(1)

    logger.info("Combining %d location frames...", len(frames))
    daily_df = pd.concat(frames, ignore_index=True)
    daily_df = daily_df.sort_values(["location", "date"])
    logger.info("Total daily rows: %d", len(daily_df))

    weekly_df, monthly_df = build_aggregates(daily_df)

    weekly_path = "us_climate_weekly.csv"
    monthly_path = "us_climate_monthly.csv"

    weekly_df.to_csv(weekly_path, index=False)
    logger.info("Saved  ->  %s  (%d rows)", weekly_path, len(weekly_df))

    monthly_df.to_csv(monthly_path, index=False)
    logger.info("Saved  ->  %s  (%d rows)", monthly_path, len(monthly_df))

    logger.info("=" * 60)
    logger.info("ALL DONE")
    logger.info("=" * 60)
    logger.info(
        "Output columns:\n"
        "  soil_moisture_mean/min/max_m3m3   [m3/m3]  0=dry ~0.6=saturated\n"
        "  cloud_cover_mean/max/min_pct      [%%]      0=clear 100=overcast\n"
        "  humidity_mean/max/min_pct         [%%]      0=dry 100=saturated\n"
    )


if __name__ == "__main__":
    main()
