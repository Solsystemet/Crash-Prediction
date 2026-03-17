"""Tests for k_means module."""

import pandas as pd
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "splice"))
from k_means import group_by_k_means, k_means


@pytest.fixture
def simple_data() -> pd.DataFrame:
    """Two clearly separated clusters of points."""
    cluster_a = pd.DataFrame(
        {"LATITUDE": [1.0, 1.1, 0.9], "LONGITUDE": [1.0, 1.1, 0.9]}
    )
    cluster_b = pd.DataFrame(
        {"LATITUDE": [10.0, 10.1, 9.9], "LONGITUDE": [10.0, 10.1, 9.9]}
    )
    return pd.concat([cluster_a, cluster_b], ignore_index=True)


@pytest.fixture
def large_data() -> pd.DataFrame:
    """100 data points spread across a grid."""
    import numpy as np

    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "LATITUDE": rng.uniform(0, 100, 100),
            "LONGITUDE": rng.uniform(0, 100, 100),
        }
    )


class TestKMeansReturnShape:
    def test_returns_two_values(self, simple_data: pd.DataFrame):
        centroids, labels = k_means(simple_data, centroids_count=2, num_iterations=10)
        assert centroids is not None
        assert labels is not None

    def test_centroids_shape(self, simple_data: pd.DataFrame):
        centroids, _ = k_means(simple_data, centroids_count=2, num_iterations=10)
        assert centroids.shape == (2, 2)

    def test_labels_length_matches_input(self, simple_data: pd.DataFrame):
        _, labels = k_means(simple_data, centroids_count=2, num_iterations=10)
        assert len(labels) == len(simple_data)

    def test_labels_are_valid_cluster_indices(self, large_data: pd.DataFrame):
        k = 5
        _, labels = k_means(large_data, centroids_count=k, num_iterations=10)
        assert labels.min().item() >= 0
        assert labels.max().item() < k

    def test_centroids_are_tensors(self, simple_data: pd.DataFrame):
        centroids, labels = k_means(simple_data, centroids_count=2, num_iterations=10)
        assert isinstance(centroids, torch.Tensor)
        assert isinstance(labels, torch.Tensor)


class TestKMeansClustering:
    def test_separates_two_distinct_clusters(self, simple_data: pd.DataFrame):
        """Points in cluster A and cluster B should get different labels."""
        torch.manual_seed(0)
        _, labels = k_means(simple_data, centroids_count=2, num_iterations=50)
        group_a = set(labels[:3].tolist())
        group_b = set(labels[3:].tolist())
        assert group_a.isdisjoint(group_b), "Distinct clusters should not share labels"

    def test_single_cluster(self, simple_data: pd.DataFrame):
        centroids, labels = k_means(simple_data, centroids_count=1, num_iterations=10)
        assert centroids.shape == (1, 2)
        assert torch.all(labels == 0)

    def test_centroids_count_equals_data_size(self):
        data = pd.DataFrame({"X": [1.0, 2.0, 3.0], "Y": [4.0, 5.0, 6.0]})
        centroids, labels = k_means(data, centroids_count=3, num_iterations=10)
        assert centroids.shape == (3, 2)
        assert len(labels) == 3

    def test_runs_with_many_iterations(self, large_data: pd.DataFrame):
        centroids, labels = k_means(large_data, centroids_count=10, num_iterations=100)
        assert centroids.shape == (10, 2)
        assert len(labels) == len(large_data)


class TestGroupByKMeans:
    def test_returns_dataframe(self, simple_data: pd.DataFrame):
        centroids, _ = k_means(simple_data, centroids_count=2, num_iterations=10)
        result = group_by_k_means(centroids, simple_data)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_original_columns(self, simple_data: pd.DataFrame):
        centroids, _ = k_means(simple_data, centroids_count=2, num_iterations=10)
        result = group_by_k_means(centroids, simple_data)
        assert "LATITUDE" in result.columns
        assert "LONGITUDE" in result.columns

    def test_adds_cluster_column(self, simple_data: pd.DataFrame):
        centroids, _ = k_means(simple_data, centroids_count=2, num_iterations=10)
        result = group_by_k_means(centroids, simple_data)
        assert "cluster" in result.columns

    def test_row_count_unchanged(self, simple_data: pd.DataFrame):
        centroids, _ = k_means(simple_data, centroids_count=2, num_iterations=10)
        result = group_by_k_means(centroids, simple_data)
        assert len(result) == len(simple_data)

    def test_cluster_values_are_valid_indices(self, simple_data: pd.DataFrame):
        k = 2
        centroids, _ = k_means(simple_data, centroids_count=k, num_iterations=10)
        result = group_by_k_means(centroids, simple_data)
        assert result["cluster"].min() >= 0
        assert result["cluster"].max() < k

    def test_assigns_distinct_clusters_to_separated_groups(
        self, simple_data: pd.DataFrame
    ):
        """Points far apart should be assigned to different clusters."""
        torch.manual_seed(0)
        centroids, _ = k_means(simple_data, centroids_count=2, num_iterations=50)
        result = group_by_k_means(centroids, simple_data)
        group_a = set(result["cluster"].iloc[:3])
        group_b = set(result["cluster"].iloc[3:])
        assert group_a.isdisjoint(group_b)

    def test_does_not_mutate_input(self, simple_data: pd.DataFrame):
        original_columns = list(simple_data.columns)
        centroids, _ = k_means(simple_data, centroids_count=2, num_iterations=10)
        group_by_k_means(centroids, simple_data)
        assert list(simple_data.columns) == original_columns

    def test_consistent_with_k_means_labels(self, simple_data: pd.DataFrame):
        """group_by_k_means with the same centroids should reproduce k_means labels."""
        torch.manual_seed(0)
        centroids, labels = k_means(simple_data, centroids_count=2, num_iterations=50)
        result = group_by_k_means(centroids, simple_data)
        assert list(result["cluster"]) == labels.tolist()

    def test_single_centroid_all_same_cluster(self, simple_data: pd.DataFrame):
        centroids, _ = k_means(simple_data, centroids_count=1, num_iterations=10)
        result = group_by_k_means(centroids, simple_data)
        assert (result["cluster"] == 0).all()

    def test_preserves_extra_columns(self):
        data = pd.DataFrame(
            {
                "LATITUDE": [1.0, 10.0],
                "LONGITUDE": [1.0, 10.0],
                "CRASH_TYPE": ["rear_end", "angle"],
            }
        )
        centroids = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
        result = group_by_k_means(centroids, data)
        assert "CRASH_TYPE" in result.columns
        assert list(result["CRASH_TYPE"]) == ["rear_end", "angle"]
