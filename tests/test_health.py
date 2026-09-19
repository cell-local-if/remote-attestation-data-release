from fastapi.testclient import TestClient

from proof_release.app import app


def test_health_reports_ready() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
