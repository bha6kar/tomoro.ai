import json

import pytest

from src.main.api import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ------------------ Unit Tests ------------------


@pytest.mark.unit
def test_health_endpoint(client):
    """Test the health check endpoint returns a correct message."""
    response = client.get("/")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "status" in data
    assert data["status"] == "ok"
    assert "message" in data
    assert data["message"] == "ConvFinQA API is running"


# ------------------ Integration Tests: Basic Functionality ------------------


@pytest.mark.integration
def test_predict_endpoint(client):
    """Test the prediction endpoint with the target question."""
    target_q = "What is the percentage change in total gross amount of unrecognized tax benefits from 2010 to 2011 "

    response = client.post("/predict", json={"question": target_q})

    assert response.status_code == 200
    data = response.json
    assert data["question"] == target_q
    assert data["answer"] == "4.3%"


@pytest.mark.integration
def test_predict_with_variations(client):
    """Test the prediction endpoint with different question phrasings."""
    variations = [
        "What is the percentage change in total gross amount of unrecognized tax benefits from 2010 to 2011",
        "can you tell me the percentage change in total gross amount of unrecognized tax benefits from 2010 to 2011? (as a percentage)",
        "what  is  the  percentage  change  in  total  gross  amount  of  unrecognized  tax  benefits  from  2010  to  2011?",
    ]

    for question in variations:
        response = client.post("/predict", json={"question": question})

        assert response.status_code == 200
        data = response.json
        assert data["question"] == question
        assert data["answer"] == "4.3%"


# ------------------ Integration Tests: Error Handling ------------------


@pytest.mark.integration
def test_invalid_requests(client):
    """Test handling of invalid requests."""
    response = client.post("/predict", json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "Question is required" in data["error"]

    response = client.post("/predict", data="not json", content_type="application/json")
    assert response.status_code in [400, 500]
    response = client.get("/nonexistent-endpoint")
    assert response.status_code == 404


@pytest.mark.integration
def test_exception_handling(client):
    """Test handling of exceptions in the API."""

    response = client.post("/predict", json={"a": "error question"})

    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "Question is required" == data["error"]


if __name__ == "__main__":
    pytest.main()
