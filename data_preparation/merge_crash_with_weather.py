"""Merge crash data with weather data based on nearest hour timestamp."""

import pandas as pd
from pathlib import Path

from data_preparation.helpers.csv_loaders import (
    get_traffic_crashes,
    get_weather_stations,
)


def merge_crash_with_weather(
    output_path: Path,
    crash_timestamp_col: str = "CRASH_DATE",
    weather_timestamp_col: str = "Measurement Timestamp",
):
    """
    Merge crash data with weather data by rounding crash time to nearest hour.

    Example:
        >>> df = merge_crash_with_weather(
        ...     crash_path=Path("data/traffic_crashes.csv"),
        ...     weather_path=Path("data/weather.csv"),
        ...     output_path=Path("data/crashes_with_weather.csv")
        ... )
    """

    # Load data
    crash_df = get_traffic_crashes()
    weather_df = get_weather_stations()

    # Parse timestamps
    crash_df[crash_timestamp_col] = pd.to_datetime(crash_df[crash_timestamp_col])
    weather_df[weather_timestamp_col] = pd.to_datetime(
        weather_df[weather_timestamp_col]
    )

    # Round crash time to nearest hour (add 1 second to force 30 min to round up)
    crash_df["_merge_hour"] = (
        crash_df[crash_timestamp_col] + pd.Timedelta(seconds=1)
    ).dt.round("h")

    # Create merge key for weather (just the hour, no rounding needed)
    weather_df["_merge_hour"] = weather_df[weather_timestamp_col].dt.floor("h")

    # Drop duplicate weather entries for same hour (keep first)
    weather_df = weather_df.drop_duplicates(subset=["_merge_hour"], keep="first")

    # Merge on the rounded hour
    merged_df = crash_df.merge(
        weather_df, on="_merge_hour", how="left", suffixes=("", "_weather")
    )

    # Clean up merge column
    merged_df = merged_df.drop(columns=["_merge_hour"])

    # Report merge stats
    total_crashes = len(crash_df)
    matched_crashes = merged_df[weather_timestamp_col].notna().sum()
    unmatched_crashes = total_crashes - matched_crashes

    print(f"Total crashes: {total_crashes}")
    print(f"Matched with weather: {matched_crashes}")
    print(f"Unmatched (no weather data): {unmatched_crashes}")
    print(f"Output columns: {len(merged_df.columns)}")

    # Save merged data
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged data to: {output_path}")

    return merged_df
