"""Data preparation package for crash prediction models.

Main entry points:
- triple_merge(): Unified data loading with configurable data sources
- get_model_data(): Alias for triple_merge using DataSourceConfig

Data source configurations:
- CRASH_ONLY: Base crash data only
- CRASH_WITH_WEATHER: Crash + weather data
- CRASH_WITH_VEHICLES: Crash + vehicle data
- CRASH_WITH_PEOPLE: Crash + people data  
- CRASH_VEHICLES_PEOPLE: Crash + vehicles + people (no weather)
- FULL_MERGE: All data sources (default)
"""

from data_preparation.triple_merge import (
    triple_merge,
    get_model_data,
    DataSourceConfig,
    CRASH_ONLY,
    CRASH_WITH_WEATHER,
    CRASH_WITH_VEHICLES,
    CRASH_WITH_PEOPLE,
    CRASH_VEHICLES_PEOPLE,
    FULL_MERGE,
)

__all__ = [
    "triple_merge",
    "get_model_data",
    "DataSourceConfig",
    "CRASH_ONLY",
    "CRASH_WITH_WEATHER",
    "CRASH_WITH_VEHICLES",
    "CRASH_WITH_PEOPLE",
    "CRASH_VEHICLES_PEOPLE",
    "FULL_MERGE",
]