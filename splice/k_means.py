"""K-means clustering for geographic crash data.

This module provides K-means clustering with a scikit-learn compatible
fit/transform API for use in ML pipelines. Clustering crash locations
can reveal spatial patterns and serve as features for prediction.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from models.data_schemas.full.traffic_crashes import TrafficCrashesSchema


@dataclass
class LocationClusterer:
    """K-means clusterer for crash locations with fit/transform API.

    This class provides a scikit-learn compatible interface for clustering
    geographic data. Fit on training data, then transform any dataset
    to get consistent cluster assignments.

    Attributes:
        n_clusters: Number of clusters to create.
        n_iterations: Number of K-means iterations.
        random_state: Random seed for reproducibility.
        lat_col: Name of latitude column.
        lon_col: Name of longitude column.
        centroids: Fitted cluster centroids (shape: n_clusters x 2).
    """

    n_clusters: int = 10
    n_iterations: int = 100
    random_state: int = 42
    lat_col: str = "LATITUDE"
    lon_col: str = "LONGITUDE"
    centroids: torch.Tensor | None = None

    def fit(self, data: pd.DataFrame | NDArray) -> "LocationClusterer":
        """Fit the clusterer on training data.

        Args:
            data: DataFrame with lat/lon columns, or array of shape (n, 2).

        Returns:
            Self for method chaining.
        """
        # Set random seed
        torch.manual_seed(self.random_state)

        # Extract coordinates
        if isinstance(data, pd.DataFrame):
            coords = data[[self.lat_col, self.lon_col]].values
        else:
            coords = data

        # Remove NaN values for fitting
        valid_mask = ~np.isnan(coords).any(axis=1)
        coords = coords[valid_mask]

        tensor_data = torch.from_numpy(coords).float()

        # Initialize centroids randomly
        n_samples = tensor_data.size(0)
        if n_samples < self.n_clusters:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_clusters} clusters"
            )

        indices = torch.randperm(n_samples)[:self.n_clusters]
        self.centroids = tensor_data[indices].clone()

        # K-means iterations
        for _ in range(self.n_iterations):
            distances = torch.cdist(tensor_data, self.centroids)
            _, labels = torch.min(distances, dim=1)

            # Update centroids
            for i in range(self.n_clusters):
                cluster_mask = labels == i
                if torch.sum(cluster_mask) > 0:
                    self.centroids[i] = torch.mean(tensor_data[cluster_mask], dim=0)

        return self

    def transform(self, data: pd.DataFrame | NDArray) -> NDArray[np.int64]:
        """Assign cluster labels to data points.

        Args:
            data: DataFrame with lat/lon columns, or array of shape (n, 2).

        Returns:
            Array of cluster labels (0 to n_clusters-1).
            NaN coordinates get label -1.
        """
        if self.centroids is None:
            raise RuntimeError("Clusterer not fitted. Call fit() first.")

        # Extract coordinates
        if isinstance(data, pd.DataFrame):
            coords = data[[self.lat_col, self.lon_col]].values
        else:
            coords = data

        # Handle NaN values
        valid_mask = ~np.isnan(coords).any(axis=1)
        labels = np.full(len(coords), -1, dtype=np.int64)

        if valid_mask.sum() > 0:
            valid_coords = coords[valid_mask]
            tensor_coords = torch.from_numpy(valid_coords).float()

            distances = torch.cdist(tensor_coords, self.centroids)
            _, cluster_labels = torch.min(distances, dim=1)
            labels[valid_mask] = cluster_labels.numpy()

        return labels

    def fit_transform(self, data: pd.DataFrame | NDArray) -> NDArray[np.int64]:
        """Fit and transform in one step.

        Args:
            data: DataFrame with lat/lon columns, or array of shape (n, 2).

        Returns:
            Array of cluster labels.
        """
        self.fit(data)
        return self.transform(data)

    def add_cluster_column(
        self,
        df: pd.DataFrame,
        column_name: str = "LOCATION_CLUSTER",
    ) -> pd.DataFrame:
        """Add cluster labels as a new column to DataFrame.

        Args:
            df: DataFrame with lat/lon columns.
            column_name: Name for the new cluster column.

        Returns:
            DataFrame with cluster column added.
        """
        result = df.copy()
        result[column_name] = self.transform(df)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize clusterer to dictionary.

        Returns:
            Dictionary containing clusterer state.
        """
        return {
            "n_clusters": self.n_clusters,
            "n_iterations": self.n_iterations,
            "random_state": self.random_state,
            "lat_col": self.lat_col,
            "lon_col": self.lon_col,
            "centroids": self.centroids.numpy().tolist() if self.centroids is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocationClusterer":
        """Deserialize clusterer from dictionary.

        Args:
            data: Dictionary from to_dict().

        Returns:
            Reconstructed LocationClusterer.
        """
        clusterer = cls(
            n_clusters=data["n_clusters"],
            n_iterations=data["n_iterations"],
            random_state=data["random_state"],
            lat_col=data["lat_col"],
            lon_col=data["lon_col"],
        )
        if data["centroids"] is not None:
            clusterer.centroids = torch.tensor(data["centroids"], dtype=torch.float32)
        return clusterer


# =============================================================================
# Legacy Functions (backward compatibility)
# =============================================================================


def k_means(
    data: pd.DataFrame, centroids_count: int, num_iterations: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform K-means clustering on geographic data using latitude/longitude.

    Args:
        data: DataFrame containing at least LATITUDE and LONGITUDE columns.
              Only these two columns are used for clustering; any additional
              columns are ignored when computing centroids.
        centroids_count: Number of clusters to create
        num_iterations: Number of iterations to run the algorithm

    Returns:
        tuple: (centroids, labels)

    Notes:
        The returned centroids have shape (centroids_count, 2) and are
        directly compatible with `group_by_k_means`.
    """

    coords = data[
        [TrafficCrashesSchema.LATITUDE, TrafficCrashesSchema.LONGITUDE]
    ].values
    tensor_data = torch.from_numpy(coords).float()
    centroids = tensor_data[torch.randperm(tensor_data.size(0))[:centroids_count]]

    for _ in range(num_iterations):
        distances = torch.cdist(tensor_data, centroids)
        _, labels = torch.min(distances, dim=1)

        for i in range(centroids_count):
            if torch.sum(labels == i) > 0:
                centroids[i] = torch.mean(tensor_data[labels == i], dim=0)

    return centroids, labels


def group_by_k_means(centroids: torch.Tensor, data: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each datapoint to its nearest centroid based only on latitude/longitude.

    Args:
        centroids: Tensor of shape (k, 2) containing centroid locations (lat, lon)
        data: DataFrame with LATITUDE and LONGITUDE columns

    Returns:
        DataFrame with all original columns plus 'cluster' column
    """
    coords = data[
        [TrafficCrashesSchema.LATITUDE, TrafficCrashesSchema.LONGITUDE]
    ].values
    tensor_coords = torch.from_numpy(coords).float()

    distances = torch.cdist(tensor_coords, centroids)
    _, labels = torch.min(distances, dim=1)

    result = data.copy()
    result["cluster"] = labels.numpy()

    return result
