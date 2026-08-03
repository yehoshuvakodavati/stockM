"""Metrics route — GET /metrics (API performance counters)."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["monitoring"])


@router.get("/metrics", summary="API performance counters")
def metrics(request: Request) -> dict:
    """Return request counters + latency stats (Lesson 10's monitoring hook).

    The middleware (Lesson 9) increments counters on every request. This
    endpoint exposes them for scraping by a monitoring system (Prometheus-style
    via a future exporter, or a simple JSON pull for a dashboard).
    """
    counters = getattr(request.app.state, "metrics", None)
    if counters is None:
        return {"enabled": False}
    return {"enabled": True, **counters.snapshot()}


# ---------------------------------------------------------------------------
# A tiny in-process metrics counter (Lesson 10). Kept here so the middleware
# (Lesson 9) and the /metrics endpoint share one object via app.state.
# ---------------------------------------------------------------------------

class APIMetrics:
    """Thread-unsafe request counters (v1 single-worker). For multi-worker,
    a real metrics lib (prometheus_client) would be used; this is the honest
    minimal stand-in documented as such."""

    def __init__(self) -> None:
        self.total_requests = 0
        self.total_errors = 0
        self.total_predict_calls = 0
        self.total_predict_latency_ms = 0.0

    def record_request(self, latency_ms: float, is_error: bool, path: str) -> None:
        self.total_requests += 1
        if is_error:
            self.total_errors += 1
        if "/predict" in path:
            self.total_predict_calls += 1
            self.total_predict_latency_ms += latency_ms

    def snapshot(self) -> dict:
        avg = (self.total_predict_latency_ms / self.total_predict_calls
               if self.total_predict_calls else 0.0)
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "total_predict_calls": self.total_predict_calls,
            "avg_predict_latency_ms": round(avg, 2),
        }
