"""Utility for generating timestamped CSV filenames with model names.

This module provides consistent filename generation for CSV exports across
the codebase, preventing overwrites and improving traceability.
"""

import re
from datetime import datetime
from pathlib import Path


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in filenames.
    
    Args:
        model_name: Raw model name (e.g., "Random Forest", "CatBoost-v2")
        
    Returns:
        Sanitized name (lowercase, special chars replaced with underscores)
    """
    # Convert to lowercase
    name = model_name.lower()
    # Replace spaces, dashes, and other special characters with underscores
    name = re.sub(r"[^a-z0-9]+", "_", name)
    # Remove leading/trailing underscores
    name = name.strip("_")
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    return name


def generate_csv_filename(
    base_name: str,
    model_name: str | None = None,
    timestamp: datetime | None = None,
    extension: str = ".csv",
) -> str:
    """Generate a timestamped filename with optional model name.
    
    Args:
        base_name: Base filename without extension (e.g., "baseline_comparison")
        model_name: Optional model name to include (e.g., "CatBoost", "Random Forest")
        timestamp: Optional datetime for the timestamp (defaults to now)
        extension: File extension (default: ".csv")
        
    Returns:
        Formatted filename string
        
    Examples:
        >>> generate_csv_filename("baseline_comparison", "CatBoost")
        'baseline_comparison_catboost_20260523_143022.csv'
        
        >>> generate_csv_filename("feature_importance", "Random Forest")
        'feature_importance_random_forest_20260523_143022.csv'
        
        >>> generate_csv_filename("training_history")
        'training_history_20260523_143022.csv'
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Build filename parts
    parts = [base_name]
    
    if model_name:
        sanitized = sanitize_model_name(model_name)
        if sanitized:  # Only add if not empty after sanitization
            parts.append(sanitized)
    
    parts.append(timestamp_str)
    
    # Ensure extension starts with a dot
    if not extension.startswith("."):
        extension = f".{extension}"
    
    return "_".join(parts) + extension


def generate_csv_path(
    output_dir: Path | str,
    base_name: str,
    model_name: str | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Generate a full path with timestamped filename.
    
    Args:
        output_dir: Directory for the output file
        base_name: Base filename without extension
        model_name: Optional model name to include
        timestamp: Optional datetime for the timestamp
        
    Returns:
        Full Path object for the CSV file
    """
    output_dir = Path(output_dir)
    filename = generate_csv_filename(base_name, model_name, timestamp)
    return output_dir / filename
