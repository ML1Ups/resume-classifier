from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient


async def test_process_batch_preserves_order_and_reports_version(client: AsyncClient) -> None:
    response = await client.post("/process", json={"texts": ["Python FastAPI", "SQL PostgreSQL"]})
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == [
        {"index": 0, "category": "backend"},
        {"index": 1, "category": "backend"},
    ]
    assert body["model"]["version"] == "1"
    assert body["model"]["run_id"] == "unit-run"
    assert body["model"]["model_uri"] == "models:/resume-classifier/1"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"texts": []},
        {"texts": ["   "]},
        {"texts": [42]},
        {"texts": ["x" * 20001]},
        {"texts": ["a"] * 33},
        {"texts": ["a"], "unexpected": True},
    ],
)
async def test_process_rejects_invalid_payloads(client: AsyncClient, payload: dict) -> None:
    response = await client.post("/process", json=payload)
    assert response.status_code == 422


async def test_model_endpoint_returns_loaded_snapshot(client: AsyncClient) -> None:
    response = await client.get("/api/v1/model")
    assert response.status_code == 200
    assert response.json()["version"] == "1"


async def test_process_uses_existing_model(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    from resume_classifier.model import ModelService

    def unexpected_load(settings):
        raise AssertionError("Request must not reload the model")

    monkeypatch.setattr(ModelService, "load", unexpected_load)
    for _ in range(3):
        assert (await client.post("/process", json={"texts": ["Python"]})).status_code == 200


async def test_inference_failure_returns_generic_error(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from resume_classifier.model import ModelService

    monkeypatch.setattr(ModelService, "predict", AsyncMock(side_effect=RuntimeError("private")))
    response = await client.post("/process", json={"texts": ["Python"]})
    assert response.status_code == 500
    assert response.json() == {"detail": "Model inference failed"}
