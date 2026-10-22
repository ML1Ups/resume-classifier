import pytest
from httpx import AsyncClient
from mlflow.exceptions import MlflowException

from resume_classifier import main
from resume_classifier.api.v1 import model as model_endpoint
from resume_classifier.config import Settings
from resume_classifier.model import LoadedModel
from resume_classifier.schemas import ModelInfo


async def test_process_batch_preserves_order_and_reports_version(client: AsyncClient) -> None:
    response = await client.post("/process", json={"texts": ["Python FastAPI", "SQL PostgreSQL"]})

    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == [
        {"index": 0, "category": "backend"},
        {"index": 1, "category": "backend"},
    ]
    assert body["model"]["alias"] == "champion"
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


async def test_process_rejects_unknown_model_alias(client: AsyncClient) -> None:
    response = await client.post("/process", json={"model": "missing", "texts": ["Python"]})

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Model alias 'missing' is not loaded. Available aliases: champion"
    }


async def test_process_uses_models_loaded_at_startup(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_load(settings: Settings) -> dict[str, LoadedModel]:
        raise AssertionError("Request must not reload models")

    monkeypatch.setattr(main, "load_models", unexpected_load)

    for _ in range(3):
        response = await client.post("/process", json={"texts": ["Python"]})
        assert response.status_code == 200


async def test_model_endpoint_lists_registered_versions(
    client: AsyncClient,
    model: LoadedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second = ModelInfo(
        name="resume-classifier",
        alias=None,
        version="2",
        run_id="run-2",
        model_uri="models:/resume-classifier/2",
        dataset_version="v2",
        train_rows=14,
        validation_metrics={"f1_macro": 0.5},
    )
    monkeypatch.setattr(model_endpoint, "list_models", lambda settings: [model.info, second])

    response = await client.get("/api/v1/model")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "resume-classifier"
    assert [item["version"] for item in body["models"]] == ["1", "2"]
    assert body["models"][1]["dataset_version"] == "v2"
    assert body["models"][1]["validation_metrics"] == {"f1_macro": 0.5}


async def test_model_endpoint_returns_503_when_registry_is_unavailable(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(settings: Settings) -> list[ModelInfo]:
        raise MlflowException("registry is down")

    monkeypatch.setattr(model_endpoint, "list_models", unavailable)

    response = await client.get("/api/v1/model")

    assert response.status_code == 503
    assert response.json() == {"detail": "Model Registry is unavailable"}


@pytest.mark.usefixtures("missing_model")
async def test_process_returns_503_without_models(client: AsyncClient) -> None:
    response = await client.post("/process", json={"texts": ["Python"]})

    assert response.status_code == 503
    assert response.json() == {"detail": "No models are loaded"}


@pytest.mark.usefixtures("missing_model")
async def test_service_endpoints_work_without_models(client: AsyncClient) -> None:
    assert (await client.get("/healthz")).status_code == 200
    assert (await client.get("/api/v1/version")).status_code == 200
    assert (await client.get("/api/v1/health")).status_code == 200
