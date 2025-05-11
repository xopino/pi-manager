from fastapi.testclient import TestClient
import pytest

from src.api.main import app

client = TestClient(app)


def test_root_endpoint_returns_200():
    response = client.get("/")
    
    assert response.status_code == 200
    assert "name" in response.json()
    assert "version" in response.json()
    assert "endpoints" in response.json() 