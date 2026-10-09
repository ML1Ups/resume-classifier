import pytest
from httpx import AsyncClient

from resume_classifier import main
from resume_classifier.config import Settings
from resume_classifier.model import LoadedModel


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
async def test_process_rejects_invalid_payloads(
    client: AsyncClient,
    payload: dict[str, object],
) -> None:
    response = await client.post("/process", json=payload)

    assert response.status_code == 422


async def test_process_uses_model_loaded_at_startup(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_load(settings: Settings) -> LoadedModel:
        raise AssertionError("Request must not reload the model")

    monkeypatch.setattr(main, "load_model", unexpected_load)

    for _ in range(3):
        response = await client.post("/process", json={"texts": ["Python"]})
        assert response.status_code == 200


async def test_model_endpoint_returns_loaded_version(client: AsyncClient) -> None:
    response = await client.get("/api/v1/model")

    assert response.status_code == 200
    assert response.json() == {
        "name": "resume-classifier",
        "alias": "champion",
        "version": "1",
        "run_id": "unit-run",
        "model_uri": "models:/resume-classifier/1",
    }


@pytest.mark.usefixtures("missing_model")
async def test_process_returns_503_without_model(client: AsyncClient) -> None:
    response = await client.post("/process", json={"texts": ["Python"]})

    assert response.status_code == 503
    assert response.json() == {"detail": "Model is not loaded"}


@pytest.mark.usefixtures("missing_model")
async def test_model_endpoint_returns_503_without_model(client: AsyncClient) -> None:
    response = await client.get("/api/v1/model")

    assert response.status_code == 503


@pytest.mark.usefixtures("missing_model")
async def test_service_endpoints_work_without_model(client: AsyncClient) -> None:
    assert (await client.get("/healthz")).status_code == 200
    assert (await client.get("/api/v1/version")).status_code == 200
    assert (await client.get("/api/v1/health")).status_code == 200
