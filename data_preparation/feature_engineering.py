"""Feature engineering for crash severity prediction.

This module implements aggressive feature engineering as recommended by research:
- Temporal features: peak hour, weekend, season, night indicators
- Vehicle features: vehicle age derived from vehicle year
- Spatial features: location cluster assignments
- Interaction features: combined weather and road conditions
"""

import pandas as pd
import numpy as np
from typing import Literal


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived temporal features from existing time columns.

    Creates:
    - IS_PEAK_HOUR: True if crash occurred during rush hour (7-9 AM or 4-7 PM)
    - IS_WEEKEND: True if crash occurred on Saturday or Sunday
    - IS_NIGHT: True if crash occurred at night (8 PM - 6 AM)
    - SEASON: Categorical season (Winter, Spring, Summer, Fall)
    - TIME_OF_DAY: Categorical time period (Morning, Afternoon, Evening, Night)

    Args:
        df: DataFrame with CRASH_HOUR, CRASH_DAY_OF_WEEK, CRASH_MONTH columns.

    Returns:
        DataFrame with new temporal feature columns added.
    """
    df = df.copy()

    # Peak hour (rush hour): 7-9 AM and 4-7 PM
    if "CRASH_HOUR" in df.columns:
        hour = df["CRASH_HOUR"]
        df["IS_PEAK_HOUR"] = ((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))
        df["IS_PEAK_HOUR"] = df["IS_PEAK_HOUR"].astype(int)

        # Night: 8 PM to 6 AM
        df["IS_NIGHT"] = (hour >= 20) | (hour <= 5)
        df["IS_NIGHT"] = df["IS_NIGHT"].astype(int)

        # Time of day categories
        conditions = [
            (hour >= 6) & (hour < 12),   # Morning
            (hour >= 12) & (hour < 17),  # Afternoon
            (hour >= 17) & (hour < 21),  # Evening
        ]
        choices = ["MORNING", "AFTERNOON", "EVENING"]
        df["TIME_OF_DAY"] = np.select(conditions, choices, default="NIGHT")

    # Weekend: Saturday (7) and Sunday (1) in Chicago data format
    # Note: Day of week encoding varies by dataset - adjust if needed
    if "CRASH_DAY_OF_WEEK" in df.columns:
        dow = df["CRASH_DAY_OF_WEEK"]
        # Assuming 1=Sunday, 7=Saturday (common format)
        df["IS_WEEKEND"] = dow.isin([1, 7]).astype(int)

    # Season from month
    if "CRASH_MONTH" in df.columns:
        month = df["CRASH_MONTH"]
        conditions = [
            month.isin([12, 1, 2]),   # Winter
            month.isin([3, 4, 5]),    # Spring
            month.isin([6, 7, 8]),    # Summer
            month.isin([9, 10, 11]),  # Fall
        ]
        choices = ["WINTER", "SPRING", "SUMMER", "FALL"]
        df["SEASON"] = np.select(conditions, choices, default="UNKNOWN")

    return df


def add_vehicle_age_features(
    df: pd.DataFrame,
    crash_year_col: str | None = None,
) -> pd.DataFrame:
    """Add vehicle age derived features.

    Creates:
    - VEHICLE_AGE: Age of oldest vehicle involved (crash year - oldest vehicle year)
    - OLD_VEHICLE_FLAG: True if oldest vehicle is > 10 years old
    - VEHICLE_AGE_CATEGORY: Categorical age group (New, Mid, Old, Very Old)

    Args:
        df: DataFrame with OLDEST_VEHICLE_YEAR column (from triple merge)
            and optionally a crash year column or CRASH_DATE.
        crash_year_col: Column name for crash year. If None, extracts from CRASH_DATE.

    Returns:
        DataFrame with vehicle age features added.
    """
    df = df.copy()

    # Determine crash year
    if crash_year_col and crash_year_col in df.columns:
        crash_year = df[crash_year_col]
    elif "CRASH_DATE" in df.columns:
        crash_year = pd.to_datetime(df["CRASH_DATE"]).dt.year
    else:
        # Default to current year if no date info
        crash_year = pd.Series([2024] * len(df))

    # Calculate vehicle age from oldest vehicle
    if "OLDEST_VEHICLE_YEAR" in df.columns:
        vehicle_year = pd.to_numeric(df["OLDEST_VEHICLE_YEAR"], errors="coerce")
        df["VEHICLE_AGE"] = crash_year - vehicle_year

        # Clean up unrealistic ages (negative or > 50 years)
        df.loc[df["VEHICLE_AGE"] < 0, "VEHICLE_AGE"] = np.nan
        df.loc[df["VEHICLE_AGE"] > 50, "VEHICLE_AGE"] = np.nan

        # Old vehicle flag (> 10 years)
        df["OLD_VEHICLE_FLAG"] = (df["VEHICLE_AGE"] > 10).astype(int)

        # Age categories
        conditions = [
            df["VEHICLE_AGE"] <= 3,
            (df["VEHICLE_AGE"] > 3) & (df["VEHICLE_AGE"] <= 7),
            (df["VEHICLE_AGE"] > 7) & (df["VEHICLE_AGE"] <= 15),
            df["VEHICLE_AGE"] > 15,
        ]
        choices = ["NEW", "MID", "OLD", "VERY_OLD"]
        df["VEHICLE_AGE_CATEGORY"] = np.select(conditions, choices, default="UNKNOWN")

    return df


def add_spatial_cluster_feature(
    df: pd.DataFrame,
    cluster_labels: np.ndarray | None = None,
    n_clusters: int = 10,
) -> pd.DataFrame:
    """Add spatial cluster feature based on crash location.

    If cluster_labels are provided, uses those directly.
    Otherwise, this is a placeholder that should be called after
    fitting a clustering model.

    Args:
        df: DataFrame with LATITUDE and LONGITUDE columns.
        cluster_labels: Pre-computed cluster labels. If None, creates placeholder.
        n_clusters: Number of clusters (for reference).

    Returns:
        DataFrame with LOCATION_CLUSTER column added.
    """
    df = df.copy()

    if cluster_labels is not None:
        df["LOCATION_CLUSTER"] = cluster_labels
    else:
        # Placeholder - actual clustering should be done during training
        df["LOCATION_CLUSTER"] = 0

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features combining multiple conditions.

    Creates:
    - ADVERSE_CONDITIONS: Count of adverse conditions present
    - NIGHT_POOR_LIGHTING: Night + poor lighting condition
    - WET_ROAD: Rain/wet road surface condition
    - IMPAIRED_DRIVER: Any BAC positive flag

    Args:
        df: DataFrame with weather, lighting, road condition columns.

    Returns:
        DataFrame with interaction features added.
    """
    df = df.copy()

    # Count adverse conditions
    adverse_count = pd.Series(0, index=df.index)

    # Weather-related adverse conditions
    if "WEATHER_CONDITION" in df.columns:
        bad_weather = ["RAIN", "SNOW", "SLEET", "FOG", "FREEZING"]
        adverse_count += df["WEATHER_CONDITION"].fillna("").str.upper().apply(
            lambda x: any(w in x for w in bad_weather)
        ).astype(int)

    # Road surface adverse conditions
    if "ROADWAY_SURFACE_COND" in df.columns:
        bad_surface = ["WET", "SNOW", "ICE", "SAND", "MUD"]
        adverse_count += df["ROADWAY_SURFACE_COND"].fillna("").str.upper().apply(
            lambda x: any(s in x for s in bad_surface)
        ).astype(int)

    # Lighting adverse conditions
    if "LIGHTING_CONDITION" in df.columns:
        bad_lighting = ["DARK", "DUSK", "DAWN"]
        adverse_count += df["LIGHTING_CONDITION"].fillna("").str.upper().apply(
            lambda x: any(l in x for l in bad_lighting)
        ).astype(int)

    df["ADVERSE_CONDITIONS_COUNT"] = adverse_count

    # Night + poor lighting interaction
    if "IS_NIGHT" in df.columns and "LIGHTING_CONDITION" in df.columns:
        poor_lighting = df["LIGHTING_CONDITION"].fillna("").str.upper().str.contains("DARK")
        df["NIGHT_POOR_LIGHTING"] = (df["IS_NIGHT"] == 1) & poor_lighting
        df["NIGHT_POOR_LIGHTING"] = df["NIGHT_POOR_LIGHTING"].astype(int)

    # Wet road condition
    if "ROADWAY_SURFACE_COND" in df.columns:
        df["WET_ROAD"] = df["ROADWAY_SURFACE_COND"].fillna("").str.upper().str.contains("WET")
        df["WET_ROAD"] = df["WET_ROAD"].astype(int)

    # Impaired driver flag (from people aggregation)
    if "ANY_BAC_POSITIVE" in df.columns:
        df["IMPAIRED_DRIVER"] = df["ANY_BAC_POSITIVE"].fillna(0).astype(int)

    return df


def add_severity_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features specifically predictive of severity.

    Creates:
    - HIGH_SPEED_AREA: Posted speed limit >= 40 mph
    - MULTI_VEHICLE: More than 2 vehicles involved
    - PEDESTRIAN_INVOLVED: Any pedestrian in crash (from person types if available)

    Args:
        df: DataFrame with crash data.

    Returns:
        DataFrame with severity risk features added.
    """
    df = df.copy()

    # High speed area
    if "POSTED_SPEED_LIMIT" in df.columns:
        df["HIGH_SPEED_AREA"] = (df["POSTED_SPEED_LIMIT"] >= 40).astype(int)

    # Multi-vehicle crash
    if "VEHICLE_COUNT" in df.columns:
        df["MULTI_VEHICLE"] = (df["VEHICLE_COUNT"] > 2).astype(int)

    return df


def add_binary_targets(
    df: pd.DataFrame,
    severity_column: str = "MOST_SEVERE_INJURY",
) -> pd.DataFrame:
    """Add binary target columns for hierarchical classification.

    Creates four binary targets for four-stage classification:
    - IS_INJURY: 1 if any injury occurred, 0 if "NO INDICATION OF INJURY"
    - IS_SEVERE: 1 if FATAL or INCAPACITATING, 0 otherwise (among injuries)
    - IS_FATAL: 1 if FATAL, 0 if INCAPACITATING (among severe injuries)
    - IS_REPORTED: 1 if REPORTED, NOT EVIDENT, 0 if NONINCAPACITATING (among minors)

    This enables hierarchical classification:
    - Level 1: Predict INJURY vs NO_INJURY (14% vs 86% - more balanced)
    - Level 2: For injury cases, predict SEVERE vs MINOR
    - Level 2.5: For severe cases, predict FATAL vs INCAPACITATING
    - Level 3: For minor cases, predict REPORTED vs NONINCAPACITATING

    Args:
        df: DataFrame with severity column.
        severity_column: Name of the column containing severity labels.

    Returns:
        DataFrame with IS_INJURY, IS_SEVERE, IS_FATAL, and IS_REPORTED columns added.
    """
    df = df.copy()

    if severity_column not in df.columns:
        raise ValueError(f"Severity column '{severity_column}' not found in DataFrame")

    severity = df[severity_column].fillna("UNKNOWN").str.upper()

    # IS_INJURY: 1 if any injury, 0 if no injury indicated
    no_injury_patterns = ["NO INDICATION OF INJURY", "NO_INDICATION_OF_INJURY"]
    is_no_injury = severity.isin([p.upper() for p in no_injury_patterns])
    df["IS_INJURY"] = (~is_no_injury).astype(int)

    # IS_SEVERE: 1 if FATAL or INCAPACITATING, 0 otherwise
    severe_patterns = ["FATAL", "INCAPACITATING INJURY", "INCAPACITATING_INJURY"]
    is_severe = severity.isin([p.upper() for p in severe_patterns])
    df["IS_SEVERE"] = is_severe.astype(int)

    # IS_FATAL: 1 if FATAL, 0 if INCAPACITATING (used for L2.5 classifier)
    fatal_patterns = ["FATAL"]
    is_fatal = severity.isin([p.upper() for p in fatal_patterns])
    df["IS_FATAL"] = is_fatal.astype(int)

    # IS_REPORTED: 1 if REPORTED, NOT EVIDENT, 0 if NONINCAPACITATING
    # This distinguishes subtle injuries from visible ones
    reported_patterns = ["REPORTED, NOT EVIDENT", "REPORTED_NOT_EVIDENT"]
    is_reported = severity.isin([p.upper() for p in reported_patterns])
    df["IS_REPORTED"] = is_reported.astype(int)

    return df


def get_severity_level(severity_value: str) -> int:
    """Map severity string to ordinal level for threshold-based prediction.

    Levels:
    - 0: NO INDICATION OF INJURY
    - 1: REPORTED, NOT EVIDENT / NONINCAPACITATING INJURY
    - 2: INCAPACITATING INJURY
    - 3: FATAL

    Args:
        severity_value: Severity string.

    Returns:
        Ordinal severity level (0-3).
    """
    severity_upper = str(severity_value).upper()
    if "FATAL" in severity_upper:
        return 3
    elif "INCAPACITATING" in severity_upper:
        return 2
    elif "NO INDICATION" in severity_upper or "NO_INDICATION" in severity_upper:
        return 0
    else:
        return 1  # REPORTED/NONINCAPACITATING


def add_ordinal_severity(
    df: pd.DataFrame,
    severity_column: str = "MOST_SEVERE_INJURY",
) -> pd.DataFrame:
    """Add ordinal severity target for regression-based approach.

    Args:
        df: DataFrame with severity column.
        severity_column: Name of the column containing severity labels.

    Returns:
        DataFrame with SEVERITY_LEVEL column (0-3).
    """
    df = df.copy()

    if severity_column not in df.columns:
        raise ValueError(f"Severity column '{severity_column}' not found in DataFrame")

    df["SEVERITY_LEVEL"] = df[severity_column].apply(get_severity_level)

    return df


def engineer_all_features(
    df: pd.DataFrame,
    include_interactions: bool = True,
    include_clusters: bool = False,
    cluster_labels: np.ndarray | None = None,
) -> pd.DataFrame:
    """Apply all feature engineering transformations.

    Args:
        df: Input DataFrame (ideally from triple_merge).
        include_interactions: Whether to add interaction features.
        include_clusters: Whether to add cluster feature.
        cluster_labels: Pre-computed cluster labels (required if include_clusters=True).

    Returns:
        DataFrame with all engineered features.
    """
    # Temporal features
    df = add_temporal_features(df)

    # Vehicle age features (requires triple merge data)
    df = add_vehicle_age_features(df)

    # Interaction features
    if include_interactions:
        df = add_interaction_features(df)

    # Severity risk features
    df = add_severity_risk_features(df)

    # Spatial clusters
    if include_clusters:
        df = add_spatial_cluster_feature(df, cluster_labels=cluster_labels)

    return df


# Feature lists for easy configuration
ENGINEERED_CATEGORICAL_FEATURES = [
    "TIME_OF_DAY",
    "SEASON",
    "VEHICLE_AGE_CATEGORY",
    "LOCATION_CLUSTER",
]

ENGINEERED_NUMERICAL_FEATURES = [
    "IS_PEAK_HOUR",
    "IS_WEEKEND",
    "IS_NIGHT",
    "VEHICLE_AGE",
    "OLD_VEHICLE_FLAG",
    "ADVERSE_CONDITIONS_COUNT",
    "NIGHT_POOR_LIGHTING",
    "WET_ROAD",
    "IMPAIRED_DRIVER",
    "HIGH_SPEED_AREA",
    "MULTI_VEHICLE",
]

# From triple merge
VEHICLE_AGG_FEATURES = [
    "VEHICLE_COUNT",
    "OLDEST_VEHICLE_YEAR",
    "AVG_VEHICLE_YEAR",
    "ANY_SPEED_VIOLATION",
]

PEOPLE_AGG_FEATURES = [
    "PERSON_COUNT",
    "DRIVER_COUNT",
    "MIN_AGE",
    "MAX_AGE",
    "AVG_AGE",
    "MAX_BAC",
    "ANY_BAC_POSITIVE",
    "SEATBELT_USAGE_RATE",
    "ANY_CELL_PHONE_USE",
    "ANY_EJECTION",
]

WEATHER_FEATURES = [
    "Air Temperature",
    "Humidity",
    "Rain Intensity",
    "Wind Speed",
]

# Binary target columns for hierarchical classification
BINARY_TARGETS = [
    "IS_INJURY",
    "IS_SEVERE",
]

ORDINAL_TARGET = "SEVERITY_LEVEL"
