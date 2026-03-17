"""Merge crash data with people data, aggregating person-level features per crash.

This module creates crash-level features derived from the people involved in each
crash, enabling fatality prediction without requiring person-level modeling.
"""

import pandas as pd
from pathlib import Path

from data_preparation.helpers.csv_loaders import (
    get_traffic_crashes,
    get_crash_people,
)


def _aggregate_people_features(people_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate person-level data to crash-level features (optimized).

    Uses vectorized operations instead of apply() for ~100x speedup.

    Args:
        people_df: DataFrame with one row per person involved in crashes.

    Returns:
        DataFrame with one row per crash, containing aggregated features.
    """
    df = people_df.copy()

    # Pre-compute boolean columns BEFORE grouping (vectorized, fast)
    df["is_pedestrian"] = (df["PERSON_TYPE"] == "PEDESTRIAN").astype(int)
    df["is_cyclist"] = (df["PERSON_TYPE"] == "BICYCLE").astype(int)
    df["is_driver"] = (df["PERSON_TYPE"] == "DRIVER").astype(int)
    df["is_fatal"] = (df["INJURY_CLASSIFICATION"] == "FATAL").astype(int)
    df["is_incapacitating"] = (
        df["INJURY_CLASSIFICATION"] == "INCAPACITATING INJURY"
    ).astype(int)
    df["is_unbelted"] = df["SAFETY_EQUIPMENT"].str.contains(
        "NONE|NOT USED", case=False, na=False
    ).astype(int)
    df["is_alcohol"] = df["BAC_RESULT"].isin(
        ["POSITIVE", "TEST PERFORMED"]
    ).astype(int)
    df["is_elderly"] = (df["AGE"] >= 65).astype(int)
    df["is_young_driver"] = (
        (df["AGE"] <= 25) & (df["PERSON_TYPE"] == "DRIVER")
    ).astype(int)
    df["is_cellphone"] = (df["CELL_PHONE_USE"] == "Y").astype(int)

    # Single groupby with fast built-in aggregations (no lambdas!)
    result = df.groupby("CRASH_RECORD_ID", as_index=False).agg(
        num_occupants=("is_pedestrian", "count"),  # count any column
        num_pedestrians=("is_pedestrian", "sum"),
        num_cyclists=("is_cyclist", "sum"),
        num_drivers=("is_driver", "sum"),
        age_min=("AGE", "min"),
        age_max=("AGE", "max"),
        age_mean=("AGE", "mean"),
        has_fatality=("is_fatal", "max"),  # max of 0/1 = any()
        has_incapacitating_injury=("is_incapacitating", "max"),
        has_unbelted=("is_unbelted", "max"),
        has_alcohol=("is_alcohol", "max"),
        has_elderly=("is_elderly", "max"),
        has_young_driver=("is_young_driver", "max"),
        has_cellphone_use=("is_cellphone", "max"),
    )

    return result


def merge_crash_with_people(
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Merge crash data with aggregated people features.

    Joins the crash dataset with derived person-level features, creating a
    comprehensive dataset for fatality prediction.

    Args:
        output_path: Optional path to save merged CSV. If None, doesn't save.

    Returns:
        DataFrame with crash data enriched with people-derived features.

    Example:
        >>> df = merge_crash_with_people(
        ...     output_path=Path("data/processed/crashes_with_people.csv")
        ... )
    """
    # Load data
    crash_df = get_traffic_crashes()
    people_df = get_crash_people()

    print(f"Loaded {len(crash_df):,} crashes")
    print(f"Loaded {len(people_df):,} people records")

    # Aggregate people features
    people_agg = _aggregate_people_features(people_df)
    print(f"Aggregated to {len(people_agg):,} crash-level records")

    # Merge on CRASH_RECORD_ID
    merged_df = crash_df.merge(
        people_agg,
        on="CRASH_RECORD_ID",
        how="left",
    )

    # Report merge stats
    matched = merged_df["num_occupants"].notna().sum()
    unmatched = len(merged_df) - matched
    fatalities = merged_df["has_fatality"].sum()

    print(f"\nMerge Statistics:")
    print(f"  Total crashes: {len(merged_df):,}")
    print(f"  Matched with people data: {matched:,}")
    print(f"  Unmatched (no people data): {unmatched:,}")
    print(f"  Fatal crashes: {fatalities:,} ({fatalities/len(merged_df)*100:.2f}%)")
    print(f"  Output columns: {len(merged_df.columns)}")

    # Fill NaN for unmatched crashes (crashes without people records)
    fill_values = {
        "num_occupants": 0,
        "num_pedestrians": 0,
        "num_cyclists": 0,
        "num_drivers": 0,
        "has_fatality": 0,
        "has_incapacitating_injury": 0,
        "has_unbelted": 0,
        "has_alcohol": 0,
        "has_elderly": 0,
        "has_young_driver": 0,
        "has_cellphone_use": 0,
    }
    merged_df = merged_df.fillna(fill_values)

    # Save if output path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        print(f"\nSaved merged data to: {output_path}")

    return merged_df


def get_crash_with_people_features() -> pd.DataFrame:
    """Load or create crash data with people features.

    Convenience function that returns the merged dataset without saving.

    Returns:
        DataFrame with crash + people-derived features.
    """
    return merge_crash_with_people(output_path=None)


if __name__ == "__main__":
    # Run merge and save to processed folder
    output = Path(__file__).parent.parent / "data" / "processed" / "crashes_with_people.csv"
    merge_crash_with_people(output_path=output)
