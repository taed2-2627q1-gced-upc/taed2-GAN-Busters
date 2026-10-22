"""Collect and summarize API request, prediction, and system metrics."""

from collections import Counter, deque
from threading import Lock
from time import time

import psutil

MAX_HISTORY = 10_000

_lock = Lock()
_requests = deque(maxlen=MAX_HISTORY)
_batches = deque(maxlen=MAX_HISTORY)
_status_counts = Counter()
_process = psutil.Process()


def record_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
) -> None:
    """Record an HTTP request's metadata, status code, and execution time."""
    with _lock:
        _requests.append(
            {
                "timestamp": time(),
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            }
        )
        _status_counts[str(status_code)] += 1


def record_batch(batch_size: int) -> None:
    """Record the number of images processed in a prediction batch."""
    with _lock:
        _batches.append(batch_size)


def get_snapshot() -> dict:
    """Return a snapshot of API performance and system resource usage.

    Include request counts, error rates, latency percentiles, prediction
    batch statistics, and current CPU and memory utilization.
    """
    with _lock:
        requests = list(_requests)
        batches = list(_batches)
        status_counts = dict(_status_counts)

    latencies = [r["duration_ms"] for r in requests]
    prediction_requests = [r for r in requests if r["path"] == "/predict"]
    prediction_latencies = [r["duration_ms"] for r in prediction_requests]

    def latency_summary(values):
        """Calculate summary statistics and latency percentiles in milliseconds."""
        if not values:
            return {
                "count": 0,
                "avg_ms": 0,
                "p50_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0,
                "max_ms": 0,
            }

        import numpy as np

        return {
            "count": len(values),
            "avg_ms": round(float(np.mean(values)), 2),
            "p50_ms": round(float(np.percentile(values, 50)), 2),
            "p95_ms": round(float(np.percentile(values, 95)), 2),
            "p99_ms": round(float(np.percentile(values, 99)), 2),
            "max_ms": round(max(values), 2),
        }

    now = time()
    recent_requests = [r for r in requests if now - r["timestamp"] <= 60]

    errors = sum(r["status_code"] >= 400 for r in requests)

    return {
        "requests": {
            "total_recorded": len(requests),
            "status_counts": status_counts,
            "error_count": errors,
            "error_rate_percent": round(100 * errors / len(requests), 2) if requests else 0,
            "requests_last_minute": len(recent_requests),
            "throughput_rpm": len(recent_requests),
        },
        "latency": {
            "all": latency_summary(latencies),
            "predictions": latency_summary(prediction_latencies),
        },
        "predictions": {
            "total_batches_recorded": len(batches),
            "total_images_recorded": sum(batches),
            "average_batch_size": round(sum(batches) / len(batches), 2) if batches else 0,
            "max_batch_size": max(batches, default=0),
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "process_memory_mb": round(_process.memory_info().rss / (1024**2), 2),
        },
        "history_limit": MAX_HISTORY,
    }
