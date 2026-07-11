from fastapi.testclient import TestClient

from doc2knowledge.api import create_app


def test_health_reports_service_status() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "doc2knowledge",
        "version": "0.1.0",
    }
