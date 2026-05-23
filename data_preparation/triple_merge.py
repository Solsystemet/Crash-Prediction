"""Multi-source data integration: merge crashes with vehicles, people, and weather.

This module implements the "Triple Merge" approach from research to capture
a holistic view of accidents by fusing:
- Human Factors: driver demographics, BAC, safety equipment
- Crash-Specific Factors: environmental conditions from weather data
- Vehicle-Related Factors: vehicle age, mechanical condition, defects

All models should use `triple_merge()` or `get_model_data()` as the single
unified data loading interface. Configure which data sources to include
via parameters or DataSourceConfig.
"""

import logging

import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from data_preparation.helpers.csv_loaders import (
    get_traffic_crashes,
    get_crash_people,
    get_crash_vehicles,
    get_weather_stations,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Source Configuration
# =============================================================================

@dataclass
class DataSourceConfig:
    """Configuration for which data sources to include in the merge.
    
    Crash data is always included as the base. Other sources are optional.
    
    Attributes:
        use_vehicles: Include vehicle data (count, age, types, speed violations).
        use_people: Include people data (demographics, BAC, safety equipment).
        use_weather: Include weather data (temperature, humidity, rain, wind).
    """
    use_vehicles: bool = True
    use_people: bool = True
    use_weather: bool = True
    
    def __repr__(self) -> str:
        sources = ["crash"]
        if self.use_vehicles:
            sources.append("vehicles")
        if self.use_people:
            sources.append("people")
        if self.use_weather:
            sources.append("weather")
        return f"DataSourceConfig({' + '.join(sources)})"
    
    def get_name_suffix(self) -> str:
        """Generate naming suffix based on data sources.
        
        Returns suffix like 'crash', 'crash_vehicle', 'crash_vehicle_people', etc.
        Used for model directory naming to indicate which datasets were used.
        
        Returns:
            String suffix with underscore-separated data source names.
        """
        parts = ["crash"]
        if self.use_vehicles:
            parts.append("vehicle")
        if self.use_people:
            parts.append("people")
        if self.use_weather:
            parts.append("weather")
        return "_".join(parts)


# Preset configurations for common use cases
CRASH_ONLY = DataSourceConfig(use_vehicles=False, use_people=False, use_weather=False)
"""Base crash data only - no additional data sources."""

CRASH_WITH_WEATHER = DataSourceConfig(use_vehicles=False, use_people=False, use_weather=True)
"""Crash data merged with weather conditions."""

CRASH_WITH_VEHICLES = DataSourceConfig(use_vehicles=True, use_people=False, use_weather=False)
"""Crash data with vehicle information (count, age, types)."""

CRASH_WITH_PEOPLE = DataSourceConfig(use_vehicles=False, use_people=True, use_weather=False)
"""Crash data with people information (demographics, BAC, safety)."""

CRASH_VEHICLES_PEOPLE = DataSourceConfig(use_vehicles=True, use_people=True, use_weather=False)
"""Crash data with both vehicle and people data, no weather."""

FULL_MERGE = DataSourceConfig(use_vehicles=True, use_people=True, use_weather=True)
"""All data sources: crash + vehicles + people + weather (default)."""


def aggregate_vehicle_data(vehicles_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate vehicle-level data to crash-level.

    For each crash, computes:
    - VEHICLE_COUNT: Number of vehicles involved
    - OLDEST_VEHICLE_YEAR: Minimum vehicle year (oldest vehicle)
    - NEWEST_VEHICLE_YEAR: Maximum vehicle year (newest vehicle)
    - AVG_VEHICLE_YEAR: Average vehicle year
    - ANY_SPEED_VIOLATION: True if any vehicle exceeded speed limit
    - VEHICLE_TYPES: Concatenated unique vehicle types

    Args:
        vehicles_df: DataFrame with vehicle data containing CRASH_RECORD_ID.

    Returns:
        DataFrame aggregated to crash level with one row per crash.
    """
    agg_dict = {
        "VEHICLE_ID": "count",  # Count of vehicles
    }

    # Vehicle year statistics
    if "VEHICLE_YEAR" in vehicles_df.columns:
        agg_dict["VEHICLE_YEAR"] = ["min", "max", "mean"]

    # Speed violation flag
    if "EXCEED_SPEED_LIMIT_I" in vehicles_df.columns:
        # Convert to boolean if needed
        vehicles_df["_speed_flag"] = vehicles_df["EXCEED_SPEED_LIMIT_I"].fillna("").astype(str).str.upper() == "Y"
        agg_dict["_speed_flag"] = "any"

    # Vehicle type
    if "VEHICLE_TYPE" in vehicles_df.columns:
        pass  # Handle separately due to string aggregation

    # Perform aggregation
    vehicle_agg = vehicles_df.groupby("CRASH_RECORD_ID").agg(agg_dict)

    # Flatten column names
    vehicle_agg.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in vehicle_agg.columns
    ]

    # Rename columns for clarity
    rename_map = {
        "VEHICLE_ID_count": "VEHICLE_COUNT",
        "VEHICLE_YEAR_min": "OLDEST_VEHICLE_YEAR",
        "VEHICLE_YEAR_max": "NEWEST_VEHICLE_YEAR",
        "VEHICLE_YEAR_mean": "AVG_VEHICLE_YEAR",
        "_speed_flag_any": "ANY_SPEED_VIOLATION",
    }
    vehicle_agg = vehicle_agg.rename(columns=rename_map)

    # Aggregate vehicle types (unique concatenation)
    if "VEHICLE_TYPE" in vehicles_df.columns:
        vehicle_types = (
            vehicles_df.groupby("CRASH_RECORD_ID")["VEHICLE_TYPE"]
            .apply(lambda x: "|".join(x.dropna().unique()))
            .rename("VEHICLE_TYPES")
        )
        vehicle_agg = vehicle_agg.join(vehicle_types)

    vehicle_agg = vehicle_agg.reset_index()

    return vehicle_agg


def aggregate_people_data(people_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate person-level data to crash-level.

    For each crash, computes:
    - PERSON_COUNT: Total people involved
    - DRIVER_COUNT: Number of drivers
    - MIN_AGE: Youngest person age
    - MAX_AGE: Oldest person age
    - AVG_AGE: Average age
    - MAX_BAC: Maximum blood alcohol content
    - ANY_BAC_POSITIVE: True if any person had positive BAC
    - SEATBELT_USAGE_RATE: Proportion using seatbelts
    - ANY_CELL_PHONE_USE: True if any person was using cell phone
    - ANY_EJECTION: True if any person was ejected

    Args:
        people_df: DataFrame with people data containing CRASH_RECORD_ID.

    Returns:
        DataFrame aggregated to crash level with one row per crash.
    """
    agg_dict = {}

    # Person count
    if "PERSON_TYPE" in people_df.columns:
        agg_dict["PERSON_TYPE"] = "count"
        # Count drivers specifically
        people_df["_is_driver"] = people_df["PERSON_TYPE"].fillna("").str.upper() == "DRIVER"

    # Age statistics
    if "AGE" in people_df.columns:
        # Clean age data
        people_df["_age_clean"] = pd.to_numeric(people_df["AGE"], errors="coerce")
        agg_dict["_age_clean"] = ["min", "max", "mean"]

    # BAC (Blood Alcohol Content)
    # Note: Column may be "BAC_RESULT VALUE" (space) or "BAC_RESULT_VALUE" (underscore)
    bac_col = "BAC_RESULT VALUE" if "BAC_RESULT VALUE" in people_df.columns else "BAC_RESULT_VALUE"
    if bac_col in people_df.columns:
        people_df["_bac_clean"] = pd.to_numeric(people_df[bac_col], errors="coerce")
        agg_dict["_bac_clean"] = "max"

    if "BAC_RESULT" in people_df.columns:
        people_df["_bac_positive"] = people_df["BAC_RESULT"].fillna("").str.upper() == "POSITIVE"
        agg_dict["_bac_positive"] = "any"

    # Safety equipment (seatbelt usage)
    if "SAFETY_EQUIPMENT" in people_df.columns:
        seatbelt_keywords = ["SHOULDER", "LAP", "BELT", "CHILD RESTRAINT"]
        people_df["_using_seatbelt"] = people_df["SAFETY_EQUIPMENT"].fillna("").str.upper().apply(
            lambda x: any(kw in x for kw in seatbelt_keywords)
        )
        agg_dict["_using_seatbelt"] = "mean"  # Proportion

    # Cell phone use
    if "CELL_PHONE_USE" in people_df.columns:
        people_df["_cell_phone"] = people_df["CELL_PHONE_USE"].fillna("").str.upper() == "YES"
        agg_dict["_cell_phone"] = "any"

    # Ejection
    if "EJECTION" in people_df.columns:
        people_df["_ejected"] = people_df["EJECTION"].fillna("").str.upper().apply(
            lambda x: "EJECTED" in x and "NOT EJECTED" not in x
        )
        agg_dict["_ejected"] = "any"

    # Driver count
    if "_is_driver" in people_df.columns:
        agg_dict["_is_driver"] = "sum"

    # Perform aggregation
    people_agg = people_df.groupby("CRASH_RECORD_ID").agg(agg_dict)

    # Flatten column names
    people_agg.columns = [
        "_".join(str(c) for c in col).strip("_") if isinstance(col, tuple) else col
        for col in people_agg.columns
    ]

    # Rename columns for clarity
    rename_map = {
        "PERSON_TYPE_count": "PERSON_COUNT",
        "_is_driver_sum": "DRIVER_COUNT",
        "_age_clean_min": "MIN_AGE",
        "_age_clean_max": "MAX_AGE",
        "_age_clean_mean": "AVG_AGE",
        "_bac_clean_max": "MAX_BAC",
        "_bac_positive_any": "ANY_BAC_POSITIVE",
        "_using_seatbelt_mean": "SEATBELT_USAGE_RATE",
        "_cell_phone_any": "ANY_CELL_PHONE_USE",
        "_ejected_any": "ANY_EJECTION",
    }
    people_agg = people_agg.rename(columns=rename_map)

    # Convert boolean columns to int for cleaner output
    bool_cols = ["ANY_BAC_POSITIVE", "ANY_CELL_PHONE_USE", "ANY_EJECTION"]
    for col in bool_cols:
        if col in people_agg.columns:
            people_agg[col] = people_agg[col].astype(float)

    people_agg = people_agg.reset_index()

    return people_agg


def merge_with_weather(
    crash_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    crash_timestamp_col: str = "CRASH_DATE",
    weather_timestamp_col: str = "Measurement Timestamp",
) -> pd.DataFrame:
    """Merge crash data with weather data by nearest hour.

    Args:
        crash_df: Crash DataFrame with timestamp column.
        weather_df: Weather DataFrame with timestamp column.
        crash_timestamp_col: Name of crash timestamp column.
        weather_timestamp_col: Name of weather timestamp column.

    Returns:
        Merged DataFrame with weather columns added.
    """
    # Parse timestamps
    crash_df = crash_df.copy()
    weather_df = weather_df.copy()

    crash_df[crash_timestamp_col] = pd.to_datetime(crash_df[crash_timestamp_col])
    weather_df[weather_timestamp_col] = pd.to_datetime(weather_df[weather_timestamp_col])

    # Round crash time to nearest hour
    crash_df["_merge_hour"] = (
        crash_df[crash_timestamp_col] + pd.Timedelta(seconds=1)
    ).dt.round("h")

    # Floor weather time to hour
    weather_df["_merge_hour"] = weather_df[weather_timestamp_col].dt.floor("h")

    # Drop duplicate weather entries for same hour
    weather_df = weather_df.drop_duplicates(subset=["_merge_hour"], keep="first")

    # Select weather columns to merge
    weather_cols_to_keep = [
        "_merge_hour",
        "Air Temperature",
        "Humidity",
        "Rain Intensity",
        "Interval Rain",
        "Total Rain",
        "Precipitation Type",
        "Wind Speed",
        "Barometric Pressure",
    ]
    weather_cols_available = [c for c in weather_cols_to_keep if c in weather_df.columns]
    weather_subset = weather_df[weather_cols_available]

    # Merge
    merged_df = crash_df.merge(
        weather_subset, on="_merge_hour", how="left", suffixes=("", "_weather")
    )

    # Clean up
    merged_df = merged_df.drop(columns=["_merge_hour"])

    return merged_df


def triple_merge(
    use_vehicles: bool = True,
    use_people: bool = True,
    use_weather: bool = True,
    config: DataSourceConfig | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Unified data loading: merge crashes with optional vehicles, people, and weather.

    This is the single entry point for loading model training data. All training
    scripts should use this function to ensure consistent data preparation.

    The function creates a crash-level dataset with configurable data sources:
    - Base: Crash data (road conditions, severity, location, time) - always included
    - Optional: Vehicle data (vehicle count, age, types, speed violations)
    - Optional: People data (demographics, BAC, safety equipment)
    - Optional: Weather data (temperature, humidity, rain, wind)

    Args:
        use_vehicles: Include aggregated vehicle data. Ignored if config is provided.
        use_people: Include aggregated people data. Ignored if config is provided.
        use_weather: Include weather data. Ignored if config is provided.
        config: DataSourceConfig object. If provided, overrides individual bool params.
        verbose: Whether to print progress information.

    Returns:
        Merged DataFrame at crash level.
    
    Examples:
        >>> # Full merge (default) - all data sources
        >>> df = triple_merge()
        
        >>> # Crash data only
        >>> df = triple_merge(use_vehicles=False, use_people=False, use_weather=False)
        
        >>> # Using preset config
        >>> from data_preparation.triple_merge import CRASH_WITH_WEATHER
        >>> df = triple_merge(config=CRASH_WITH_WEATHER)
    """
    # Use config if provided, otherwise use individual bool params
    if config is not None:
        use_vehicles = config.use_vehicles
        use_people = config.use_people
        use_weather = config.use_weather
    
    # Track which sources are being used for verbose output
    sources_used = ["crash"]
    vehicle_agg = None
    people_agg = None
    
    if verbose:
        logger.info("Loading crash data...")
    crash_df = get_traffic_crashes()
    initial_count = len(crash_df)
    merged_df = crash_df

    if verbose:
        logger.info(f"  Loaded {initial_count:,} crashes")

    # Load and aggregate vehicle data (optional)
    if use_vehicles:
        if verbose:
            logger.info("Loading and aggregating vehicle data...")
        vehicles_df = get_crash_vehicles()
        vehicle_agg = aggregate_vehicle_data(vehicles_df)

        if verbose:
            logger.info(f"  Aggregated {len(vehicles_df):,} vehicle records to {len(vehicle_agg):,} crashes")

        # Merge vehicles
        merged_df = merged_df.merge(vehicle_agg, on="CRASH_RECORD_ID", how="left")
        sources_used.append("vehicles")

    # Load and aggregate people data (optional)
    if use_people:
        if verbose:
            logger.info("Loading and aggregating people data...")
        people_df = get_crash_people()
        people_agg = aggregate_people_data(people_df)

        if verbose:
            logger.info(f"  Aggregated {len(people_df):,} person records to {len(people_agg):,} crashes")

        # Merge people
        merged_df = merged_df.merge(people_agg, on="CRASH_RECORD_ID", how="left")
        sources_used.append("people")

    # Load and merge weather (optional)
    if use_weather:
        if verbose:
            logger.info("Loading and merging weather data...")
        weather_df = get_weather_stations()
        merged_df = merge_with_weather(merged_df, weather_df)

        weather_matched = merged_df["Air Temperature"].notna().sum()
        if verbose:
            logger.info(f"  Matched {weather_matched:,} crashes with weather data")
        sources_used.append("weather")

    if verbose:
        logger.info(f"Data merge complete! Sources: {' + '.join(sources_used)}")
        logger.info(f"  Final dataset: {len(merged_df):,} rows x {len(merged_df.columns)} columns")
        if vehicle_agg is not None:
            logger.info(f"  Columns from vehicles: {list(vehicle_agg.columns[1:])}")
        if people_agg is not None:
            logger.info(f"  Columns from people: {list(people_agg.columns[1:])}")

    return merged_df


def get_model_data(
    config: DataSourceConfig = FULL_MERGE,
    verbose: bool = True,
) -> pd.DataFrame:
    """Unified entry point for loading model training data.
    
    This is an alias for triple_merge() that uses DataSourceConfig.
    All models should use this function or triple_merge() directly.
    
    Args:
        config: DataSourceConfig specifying which data sources to include.
        verbose: Whether to print progress information.
    
    Returns:
        Merged DataFrame at crash level.
    
    Examples:
        >>> from data_preparation.triple_merge import get_model_data, CRASH_ONLY
        >>> df = get_model_data()  # Full merge (default)
        >>> df = get_model_data(CRASH_ONLY)  # Crash data only
    """
    return triple_merge(config=config, verbose=verbose)


def save_triple_merged_data(output_path: Path | str, **kwargs) -> pd.DataFrame:
    """Perform triple merge and save to CSV.

    Args:
        output_path: Path to save the merged CSV.
        **kwargs: Arguments passed to triple_merge().

    Returns:
        Merged DataFrame.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_df = triple_merge(**kwargs)
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged data to: {output_path}")

    return merged_df


if __name__ == "__main__":
    # Demo: Different data source configurations
    print("=" * 60)
    print("Demo: Unified data loading with configurable sources")
    print("=" * 60)
    
    # Full merge (default) - all data sources
    print("\n1. Full merge (crash + vehicles + people + weather):")
    df_full = triple_merge(verbose=True)
    print(f"   Columns: {len(df_full.columns)}")
    
    # Crash only
    print("\n2. Crash data only:")
    df_crash = triple_merge(use_vehicles=False, use_people=False, use_weather=False, verbose=True)
    print(f"   Columns: {len(df_crash.columns)}")
    
    # Using preset config
    print("\n3. Using preset config (CRASH_WITH_WEATHER):")
    df_weather = triple_merge(config=CRASH_WITH_WEATHER, verbose=True)
    print(f"   Columns: {len(df_weather.columns)}")
    
    print("\n" + "=" * 60)
    print("Column comparison:")
    print(f"  Crash only:    {len(df_crash.columns)} columns")
    print(f"  With weather:  {len(df_weather.columns)} columns")
    print(f"  Full merge:    {len(df_full.columns)} columns")
