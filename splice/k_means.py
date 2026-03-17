import pandas as pd
import torch

from models.data_schemas.full.traffic_crashes import TrafficCrashesSchema


def k_means(data: pd.DataFrame, centroids_count: int, num_iterations: int):
    """
    Perform K-means clustering on geographic data.

    Args:
        data: DataFrame with numeric columns to cluster (e.g., LATITUDE, LONGITUDE)
        centroids_count: Number of clusters to create
        num_iterations: Number of iterations to run the algorithm

    Returns:
        tuple: (centroids, labels)
    """

    tensor_data = torch.from_numpy(data.values).float()
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
