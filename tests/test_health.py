from __future__ import annotations
 
from fastapi.testclient import TestClient
 
import src.api.main as api_main
 
 
def _dependency(name: str, status: str, configured: bool = True) -> api_main.DependencyHealth:
    return api_main.DependencyHealth(
        name=name,
        status=status,
        configured=configured,
        latency_ms=1.0,
        message=f"{name} {status}",
    )
 
 
def test_detailed_health_reports_dependency_checks(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "check_s3_health",
        lambda: _dependency("s3", "healthy"),
    )
    monkeypatch.setattr(
        api_main,
        "check_snowflake_health",
        lambda: _dependency("snowflake", "skipped", configured=False),
    )
    monkeypatch.setattr(
        api_main,
        "check_redis_health",
        lambda: _dependency("redis", "unhealthy"),
    )
 
    client = TestClient(api_main.app)
    response = client.get("/health/detailed")
 
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unhealthy"
    assert payload["dependencies"]["s3"]["status"] == "healthy"
    assert payload["dependencies"]["snowflake"]["configured"] is False
    assert payload["dependencies"]["redis"]["message"] == "redis unhealthy"
 
 
def test_lightweight_health_links_to_detailed_health():
    client = TestClient(api_main.app)
    response = client.get("/health")
 
    assert response.status_code == 200
    assert response.json()["detailed_health_url"] == "/health/detailed"
 
 