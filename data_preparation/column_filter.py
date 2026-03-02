"""Filter out columns that are not useful for crash prediction."""

import pandas as pd
from pathlib import Path


def filter_columns(
    input_path: Path,
    output_path: Path,
    columns_to_drop: list[str]
) -> pd.DataFrame:
    """
    Load CSV, drop specified columns, and save filtered result.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to save filtered CSV
        columns_to_drop: List of column names to remove
    
    Returns:
        Filtered DataFrame
    
    Example:
        >>> from data_preparation.column_filter import filter_columns
        >>> from pathlib import Path
        
        >>> df = filter_columns(
        ...     input_path=Path("data/traffic_crashes.csv"),
        ...     output_path=Path("data/traffic_crashes_filtered.csv"),
        ...     columns_to_drop=["CRASH_RECORD_ID", "RD_NO", "LOCATION"]
        ... )
    """
    df = pd.read_csv(input_path)
    
    # Only drop columns that exist in the dataframe
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    missing_cols = [col for col in columns_to_drop if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: These columns not found in data: {missing_cols}")
    
    df_filtered = df.drop(columns=existing_cols)
    
    print(f"Original columns: {len(df.columns)}")
    print(f"Dropped columns: {len(existing_cols)}")
    print(f"Remaining columns: {len(df_filtered.columns)}")
    
    df_filtered.to_csv(output_path, index=False)
    print(f"Saved filtered data to: {output_path}")
    
    return df_filtered