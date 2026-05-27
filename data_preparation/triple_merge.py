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


