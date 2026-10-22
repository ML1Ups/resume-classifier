from pathlib import Path

import pandas as pd
import pytest
from httpx import AsyncClient
from mlflow.exceptions import MlflowException

from resume_classifier.api.v1 import model as model_endpoint
from resume_classifier.config import Settings
from resume_classifier.model import LoadedModel
from resume_classifier.schemas import ModelInfo


@pytest.fixture
def evaluation_dir(tmp_path: Path, settings: Settings) -> Path:
    pd.DataFrame(
        {
            "record_id": ["1", "2", "3"],
            "text": ["Python", "SQL", "Go"],
            "category": ["backend", "backend", "frontend"],
        }
    ).to_csv(tmp_path / "v1_test.csv", index=False)
    settings.evaluation_data_dir = tmp_path
    return tmp_path


@pytest.mark.usefixtures("evaluation_dir")
async def test_evaluation_reports_selected_metrics_and_sample(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/evaluations",
        json={"dataset": "v1", "metrics": ["accuracy", "f1_weighted"], "sample_size": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"]["alias"] == "champion"
    assert body["dataset"] == {"version": "v1", "rows": 3}
    assert set(body["metrics"]) == {"accuracy", "f1_weighted"}
    assert body["metrics"]["accuracy"] == pytest.approx(2 / 3)
    assert body["predictions"] == [
        {"index": 0, "category": "backend", "expected": "backend"},
        {"index": 1, "category": "backend", "expected": "backend"},
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"dataset": "../v1"},
        {"dataset": "v1", "metrics": []},
        {"dataset": "v1", "metrics": ["accuracy", "f1_macro", "f1_weighted"]},
        {"dataset": "v1", "metrics": ["accuracy", "accuracy"]},
        {"dataset": "v1", "metrics": ["recall"]},
        {"dataset": "v1", "sample_size": 21},
        {"dataset": "v1", "unexpected": True},
    ],
)
async def test_evaluation_rejects_invalid_payloads(
    client: AsyncClient,
    payload: dict[str, object],
) -> None:
    response = await client.post("/api/v1/evaluations", json=payload)

    assert response.status_code == 422


@pytest.mark.usefixtures("evaluation_dir")
async def test_evaluation_rejects_unknown_dataset_and_model(client: AsyncClient) -> None:
    dataset = await client.post("/api/v1/evaluations", json={"dataset": "v9"})
    model = await client.post("/api/v1/evaluations", json={"dataset": "v1", "model": "missing"})

    assert dataset.status_code == 404
    assert dataset.json() == {"detail": "Dataset version 'v9' is not found"}
    assert model.status_code == 404
    assert model.json() == {
        "detail": "Model alias 'missing' is not loaded. Available aliases: champion"
    }


@pytest.mark.usefixtures("missing_model")
async def test_evaluation_returns_503_without_models(client: AsyncClient) -> None:
    response = await client.post("/api/v1/evaluations", json={"dataset": "v1"})

    assert response.status_code == 503
    assert response.json() == {"detail": "No models are loaded"}


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
async def test_service_endpoints_work_without_models(client: AsyncClient) -> None:
    assert (await client.get("/healthz")).status_code == 200
    assert (await client.get("/api/v1/version")).status_code == 200
    assert (await client.get("/api/v1/health")).status_code == 200
