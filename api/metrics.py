"""
API metrics collection and exposure.

Provides:
- Latency tracking per endpoint
- Prediction confidence distribution tracking
- Fallback rate monitoring
- Prometheus-style metrics endpoint
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Metrics router for /metrics endpoint
metrics_router = APIRouter(tags=["metrics"])


@dataclass
class LatencyBucket:
    """Tracks latency distribution in histogram buckets."""
    
    # Bucket boundaries in milliseconds
    boundaries: List[float] = field(default_factory=lambda: [10, 25, 50, 100, 250, 500, 1000, 2500, 5000])
    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_count: int = 0
    total_sum: float = 0.0
    
    def observe(self, latency_ms: float) -> None:
        """Record a latency observation."""
        self.total_count += 1
        self.total_sum += latency_ms
        
        # Find bucket
        for boundary in self.boundaries:
            if latency_ms <= boundary:
                self.counts[f"le_{boundary}"] += 1
                break
        else:
            self.counts["le_inf"] += 1
    
    def get_histogram(self) -> Dict[str, Any]:
        """Get histogram data."""
        return {
            "buckets": dict(self.counts),
            "count": self.total_count,
            "sum": self.total_sum,
            "avg": self.total_sum / self.total_count if self.total_count > 0 else 0,
        }


@dataclass 
class ConfidenceTracker:
    """Tracks prediction confidence distribution."""
    
    total_predictions: int = 0
    sum_confidence: float = 0.0
    low_confidence_count: int = 0  # Below threshold
    confidence_threshold: float = 0.6
    
    # Histogram buckets for confidence
    buckets: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def observe(self, confidence: float) -> None:
        """Record a confidence observation."""
        self.total_predictions += 1
        self.sum_confidence += confidence
        
        if confidence < self.confidence_threshold:
            self.low_confidence_count += 1
        
        # Bucket by decile
        bucket = min(int(confidence * 10), 9)  # 0-9
        self.buckets[f"0.{bucket}"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get confidence statistics."""
        return {
            "total_predictions": self.total_predictions,
            "avg_confidence": self.sum_confidence / self.total_predictions if self.total_predictions > 0 else 0,
            "low_confidence_count": self.low_confidence_count,
            "low_confidence_pct": (self.low_confidence_count / self.total_predictions * 100) if self.total_predictions > 0 else 0,
            "threshold": self.confidence_threshold,
            "distribution": dict(self.buckets),
        }


@dataclass
class FallbackTracker:
    """Tracks fallback events and reasons."""
    
    total_requests: int = 0
    fallback_count: int = 0
    reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def record_success(self) -> None:
        """Record successful prediction without fallback."""
        self.total_requests += 1
    
    def record_fallback(self, reason: str) -> None:
        """Record a fallback event."""
        self.total_requests += 1
        self.fallback_count += 1
        self.reasons[reason] += 1
        logger.warning(f"Prediction fallback: {reason}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fallback statistics."""
        return {
            "total_requests": self.total_requests,
            "fallback_count": self.fallback_count,
            "fallback_rate": (self.fallback_count / self.total_requests * 100) if self.total_requests > 0 else 0,
            "reasons": dict(self.reasons),
        }


class MetricsCollector:
    """Central metrics collection singleton."""
    
    _instance: Optional["MetricsCollector"] = None
    _lock = Lock()
    
    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize metrics collectors."""
        self.start_time = datetime.utcnow()
        
        # Per-endpoint latency tracking
        self.endpoint_latency: Dict[str, LatencyBucket] = defaultdict(LatencyBucket)
        
        # Prediction confidence tracking
        self.confidence = ConfidenceTracker()
        
        # Fallback tracking
        self.fallbacks = FallbackTracker()
        
        # Request counters
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Per-model prediction counts
        self.model_predictions: Dict[str, int] = defaultdict(int)
        
        self._lock = Lock()
    
    def record_request(self, endpoint: str, latency_ms: float, status_code: int) -> None:
        """Record an API request."""
        with self._lock:
            self.request_counts[endpoint] += 1
            self.endpoint_latency[endpoint].observe(latency_ms)
            
            if status_code >= 400:
                self.error_counts[endpoint] += 1
    
    def record_prediction(
        self,
        model_type: str,
        confidence: float,
        fallback: bool = False,
        fallback_reason: Optional[str] = None,
    ) -> None:
        """Record a prediction event."""
        with self._lock:
            self.model_predictions[model_type] += 1
            self.confidence.observe(confidence)
            
            if fallback and fallback_reason:
                self.fallbacks.record_fallback(fallback_reason)
            else:
                self.fallbacks.record_success()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "start_time": self.start_time.isoformat(),
            "requests": {
                "total": sum(self.request_counts.values()),
                "by_endpoint": dict(self.request_counts),
                "errors": dict(self.error_counts),
            },
            "latency": {
                endpoint: bucket.get_histogram()
                for endpoint, bucket in self.endpoint_latency.items()
            },
            "predictions": {
                "total": sum(self.model_predictions.values()),
                "by_model": dict(self.model_predictions),
                "confidence": self.confidence.get_stats(),
                "fallbacks": self.fallbacks.get_stats(),
            },
        }
    
    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        self._initialize()


# Global metrics instance
def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return MetricsCollector()


# Convenience functions
def record_prediction_metrics(
    model_type: str,
    confidence: float,
    fallback: bool = False,
    fallback_reason: Optional[str] = None,
) -> None:
    """Record prediction metrics (convenience function)."""
    get_metrics().record_prediction(
        model_type=model_type,
        confidence=confidence,
        fallback=fallback,
        fallback_reason=fallback_reason,
    )


@metrics_router.get("/metrics")
async def metrics_endpoint() -> JSONResponse:
    """Expose collected metrics as JSON.
    
    Returns metrics including:
    - Uptime and request counts
    - Per-endpoint latency histograms  
    - Prediction confidence distribution
    - Fallback rates and reasons
    """
    metrics = get_metrics().get_metrics()
    return JSONResponse(content=metrics)


@metrics_router.get("/metrics/prometheus")
async def prometheus_metrics() -> Response:
    """Expose metrics in Prometheus text format.
    
    Basic Prometheus-compatible output for scraping.
    """
    metrics = get_metrics().get_metrics()
    lines = []
    
    # Uptime
    lines.append(f"# HELP api_uptime_seconds Time since API started")
    lines.append(f"# TYPE api_uptime_seconds gauge")
    lines.append(f'api_uptime_seconds {metrics["uptime_seconds"]:.2f}')
    
    # Request totals
    lines.append(f"# HELP api_requests_total Total API requests")
    lines.append(f"# TYPE api_requests_total counter")
    for endpoint, count in metrics["requests"]["by_endpoint"].items():
        safe_endpoint = endpoint.replace("/", "_").strip("_")
        lines.append(f'api_requests_total{{endpoint="{endpoint}"}} {count}')
    
    # Predictions
    lines.append(f"# HELP api_predictions_total Total predictions made")
    lines.append(f"# TYPE api_predictions_total counter")
    lines.append(f'api_predictions_total {metrics["predictions"]["total"]}')
    
    # Confidence
    lines.append(f"# HELP api_prediction_confidence_avg Average prediction confidence")
    lines.append(f"# TYPE api_prediction_confidence_avg gauge")
    lines.append(f'api_prediction_confidence_avg {metrics["predictions"]["confidence"]["avg_confidence"]:.4f}')
    
    # Fallback rate
    lines.append(f"# HELP api_fallback_rate_percent Percentage of predictions using fallback")
    lines.append(f"# TYPE api_fallback_rate_percent gauge")
    lines.append(f'api_fallback_rate_percent {metrics["predictions"]["fallbacks"]["fallback_rate"]:.2f}')
    
    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@metrics_router.post("/metrics/reset")
async def reset_metrics() -> JSONResponse:
    """Reset all collected metrics.
    
    Useful for testing or starting fresh measurement periods.
    """
    get_metrics().reset()
    return JSONResponse(content={"status": "reset", "timestamp": datetime.utcnow().isoformat()})
