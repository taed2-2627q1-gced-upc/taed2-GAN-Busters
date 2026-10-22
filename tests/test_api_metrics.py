import time
import pytest
from gan_busters.api import metrics

@pytest.fixture(autouse=True)
def clear_metrics():
    with metrics._lock:
        metrics._requests.clear()
        metrics._batches.clear()
        metrics._status_counts.clear()
    yield

def test_record_request_and_snapshot():
    metrics.record_request("GET", "/health/live", 200, 15.5)
    metrics.record_request("POST", "/predict", 400, 45.0)

    snapshot = metrics.get_snapshot()

    assert snapshot["requests"]["total_recorded"] == 2
    assert snapshot["requests"]["status_counts"] == {"200": 1, "400": 1}
    assert snapshot["requests"]["error_count"] == 1
    assert snapshot["requests"]["error_rate_percent"] == 50.0
    assert snapshot["latency"]["all"]["count"] == 2
    assert snapshot["latency"]["predictions"]["count"] == 1

def test_record_batch():
    metrics.record_batch(3)
    metrics.record_batch(5)

    snapshot = metrics.get_snapshot()

    assert snapshot["predictions"]["total_batches_recorded"] == 2
    assert snapshot["predictions"]["total_images_recorded"] == 8
    assert snapshot["predictions"]["average_batch_size"] == 4.0
    assert snapshot["predictions"]["max_batch_size"] == 5


def test_latency_summary_empty():
    summary = metrics.get_snapshot()["latency"]["all"]
    assert summary["count"] == 0
    assert summary["avg_ms"] == 0
    assert summary["p50_ms"] == 0
    assert summary["max_ms"] == 0


def test_metrics_history_limit():
    for i in range(10):
        metrics.record_request("GET", "/", 200, float(i))

    snapshot = metrics.get_snapshot()
    assert snapshot["requests"]["total_recorded"] == 10
    assert snapshot["history_limit"] == metrics.MAX_HISTORY