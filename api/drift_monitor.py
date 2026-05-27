"""
Drift monitoring for API inference.

Tracks feature distributions at inference time and compares against
training baselines to detect data drift.

Uses Population Stability Index (PSI) for drift detection:
- PSI < 0.1: No significant drift
- PSI 0.1-0.25: Moderate drift (monitor)
- PSI > 0.25: Significant drift (alert)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureDistribution:
    """Stores binned distribution for a feature."""

    name: str
    bins: List[float]  # Bin edges
    counts: List[int]  # Counts per bin
    total: int = 0

    def get_proportions(self) -> np.ndarray:
        """Get proportion in each bin."""
        if self.total == 0:
            return np.zeros(len(self.counts))
        return np.array(self.counts) / self.total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "bins": self.bins,
            "counts": self.counts,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureDistribution":
        return cls(
            name=data["name"],
            bins=data["bins"],
            counts=data["counts"],
            total=data["total"],
        )


def compute_psi(expected: np.ndarray, actual: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Compute Population Stability Index between two distributions.

    Args:
        expected: Expected (training) distribution proportions
        actual: Actual (inference) distribution proportions
        epsilon: Small value to avoid log(0)

    Returns:
        PSI value (0 = identical, higher = more drift)
    """
    expected = np.clip(expected, epsilon, 1 - epsilon)
    actual = np.clip(actual, epsilon, 1 - epsilon)

    psi = np.sum((actual - expected) * np.log(actual / expected))
    return float(psi)


@dataclass
class DriftAlert:
    """A drift detection alert."""

    feature: str
    psi: float
    severity: str  # "low", "moderate", "high"
    timestamp: str
    baseline_distribution: List[float]
    current_distribution: List[float]


class DriftMonitor:
    """
    Monitors feature drift between training and inference distributions.

    Usage:
        # At startup, load training baselines
        monitor = DriftMonitor()
        monitor.load_baselines("path/to/baselines.json")

        # During inference, record feature values
        monitor.record_features({"age": 25, "speed_limit": 30, ...})

        # Periodically check for drift
        alerts = monitor.check_drift()
    """

    def __init__(
        self,
        psi_threshold_moderate: float = 0.1,
        psi_threshold_high: float = 0.25,
        n_bins: int = 10,
    ):
        """
        Initialize drift monitor.

        Args:
            psi_threshold_moderate: PSI threshold for moderate drift warning
            psi_threshold_high: PSI threshold for high drift alert
            n_bins: Number of bins for numerical features
        """
        self.psi_threshold_moderate = psi_threshold_moderate
        self.psi_threshold_high = psi_threshold_high
        self.n_bins = n_bins

        # Training baselines (loaded from file)
        self.baselines: Dict[str, FeatureDistribution] = {}

        # Current inference distributions
        self.current: Dict[str, FeatureDistribution] = {}

        # Lock for thread safety
        self._lock = Lock()

        # Alert history
        self.alerts: List[DriftAlert] = []

        # Sample count
        self.n_samples = 0

    def load_baselines(self, path: str | Path) -> None:
        """Load training distribution baselines from JSON file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Baseline file not found: {path}")
            return

        with open(path) as f:
            data = json.load(f)

        self.baselines = {
            name: FeatureDistribution.from_dict(dist)
            for name, dist in data.get("distributions", {}).items()
        }
        logger.info(f"Loaded {len(self.baselines)} feature baselines from {path}")

    def save_baselines(self, path: str | Path) -> None:
        """Save current distributions as baselines."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_samples": self.n_samples,
            "distributions": {
                name: dist.to_dict() for name, dist in self.current.items()
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.current)} feature baselines to {path}")

    def record_features(self, features: Dict[str, Any]) -> None:
        """
        Record feature values from a single inference request.

        Args:
            features: Dictionary of feature name -> value
        """
        with self._lock:
            self.n_samples += 1

            for name, value in features.items():
                if value is None:
                    continue

                # Initialize distribution if needed
                if name not in self.current:
                    # Use baseline bins if available
                    if name in self.baselines:
                        bins = self.baselines[name].bins
                    else:
                        # Create default bins (will be refined)
                        bins = list(np.linspace(0, 100, self.n_bins + 1))

                    self.current[name] = FeatureDistribution(
                        name=name,
                        bins=bins,
                        counts=[0] * self.n_bins,
                    )

                # Update distribution
                dist = self.current[name]
                dist.total += 1

                # Find bin for value
                try:
                    value = float(value)
                    bin_idx = np.digitize(value, dist.bins[1:-1])
                    bin_idx = min(bin_idx, len(dist.counts) - 1)
                    dist.counts[bin_idx] += 1
                except (ValueError, TypeError):
                    pass  # Skip non-numeric values

    def check_drift(self) -> List[DriftAlert]:
        """
        Check for drift in all monitored features.

        Returns:
            List of drift alerts (empty if no drift detected)
        """
        alerts = []

        with self._lock:
            for name, baseline in self.baselines.items():
                if name not in self.current:
                    continue

                current = self.current[name]
                if current.total < 100:
                    continue  # Need sufficient samples

                # Compute PSI
                expected = baseline.get_proportions()
                actual = current.get_proportions()

                # Ensure same length
                if len(expected) != len(actual):
                    continue

                psi = compute_psi(expected, actual)

                # Check thresholds
                if psi >= self.psi_threshold_high:
                    severity = "high"
                elif psi >= self.psi_threshold_moderate:
                    severity = "moderate"
                else:
                    continue  # No alert needed

                alert = DriftAlert(
                    feature=name,
                    psi=psi,
                    severity=severity,
                    timestamp=datetime.utcnow().isoformat(),
                    baseline_distribution=expected.tolist(),
                    current_distribution=actual.tolist(),
                )
                alerts.append(alert)

                logger.warning(
                    f"Drift detected in '{name}': PSI={psi:.4f} ({severity})"
                )

        self.alerts.extend(alerts)
        return alerts

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift status for all features."""
        summary = {
            "n_samples": self.n_samples,
            "n_features_monitored": len(self.current),
            "n_baselines": len(self.baselines),
            "recent_alerts": len(self.alerts),
            "features": {},
        }

        with self._lock:
            for name, baseline in self.baselines.items():
                if name not in self.current:
                    summary["features"][name] = {"status": "no_data"}
                    continue

                current = self.current[name]
                expected = baseline.get_proportions()
                actual = current.get_proportions()

                if len(expected) != len(actual) or current.total < 100:
                    summary["features"][name] = {"status": "insufficient_data"}
                    continue

                psi = compute_psi(expected, actual)
                if psi >= self.psi_threshold_high:
                    status = "high_drift"
                elif psi >= self.psi_threshold_moderate:
                    status = "moderate_drift"
                else:
                    status = "stable"

                summary["features"][name] = {
                    "status": status,
                    "psi": round(psi, 4),
                    "n_samples": current.total,
                }

        return summary

    def reset(self) -> None:
        """Reset current distributions (keeps baselines)."""
        with self._lock:
            self.current = {}
            self.n_samples = 0
            self.alerts = []


# Global drift monitor instance
_drift_monitor: Optional[DriftMonitor] = None


def get_drift_monitor() -> DriftMonitor:
    """Get or create the global drift monitor instance."""
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()
    return _drift_monitor


def record_inference_features(features: Dict[str, Any]) -> None:
    """Record features from an inference request (convenience function)."""
    get_drift_monitor().record_features(features)
