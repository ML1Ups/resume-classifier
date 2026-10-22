import pytest
from httpx import AsyncClient

from resume_classifier import main
from resume_classifier.config import Settings
from resume_classifier.model import LoadedModel


async def test_process_batch_preserves_order_and_reports_version(client: AsyncClient) -> None:
    response = await client.post(
        "/process",
        json={"model": "champion", "texts": ["Python FastAPI", "SQL PostgreSQL"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == [
        {"index": 0, "category": "backend", "expected": None},
        {"index": 1, "category": "backend", "expected": None},
    ]
    assert body["model"]["version"] == "1"
    assert body["model"]["run_id"] == "unit-run"
    assert body["model"]["model_uri"] == "models:/resume-classifier/1"
    assert body["metrics"] == {}
    assert body["dataset"] is None


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
        {"texts": ["a"], "dataset": "v1"},
        {"dataset": "v1", "metrics": []},
        {"dataset": "v1", "metrics": ["accuracy", "f1_macro", "f1_weighted"]},
        {"dataset": "v1", "metrics": ["accuracy", "accuracy"]},
    ],
)
async def test_process_rejects_invalid_payloads(
    client: AsyncClient,
    payload: dict[str, object],
) -> None:
    response = await client.post("/process", json=payload)

    assert response.status_code == 422


async def test_process_rejects_unknown_model(client: AsyncClient) -> None:
    response = await client.post(
        "/process",
        json={"model": "missing", "texts": ["Python"]},
    )

    assert response.status_code == 404


async def test_process_uses_model_loaded_at_startup(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_load(settings: Settings) -> LoadedModel:
        raise AssertionError("Request must not reload the model")

    monkeypatch.setattr(main, "load_model", unexpected_load)

    for _ in range(3):
        response = await client.post("/process", json={"model": "champion", "texts": ["Python"]})
        assert response.status_code == 200


async def test_model_endpoint_returns_registered_models(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    version = SimpleNamespace(
        version="7",
        run_id="run-7",
        tags={"candidate": "logreg_words", "dataset.version": "v1", "dataset.train_rows": "100"},
    )
    client_mock = Mock()
    client_mock.get_registered_model.return_value.aliases = {"demo-v1": "7"}
    client_mock.search_model_versions.return_value = [version]
    client_mock.get_run.return_value.data.metrics = {
        "validation_accuracy": 0.9,
        "validation_f1_macro": 0.8,
        "test_accuracy": 0.88,
    }
    monkeypatch.setattr(
        "resume_classifier.model.MlflowClient",
        Mock(return_value=client_mock),
    )

    response = await client.get("/api/v1/model")

    assert response.status_code == 200
    assert response.json() == {
        "name": "resume-classifier",
        "models": [
            {
                "name": "resume-classifier",
                "alias": "demo-v1",
                "version": "7",
                "run_id": "run-7",
                "model_uri": "models:/resume-classifier/7",
                "candidate": "logreg_words",
                "dataset_version": "v1",
                "train_rows": 100,
                "validation_metrics": {
                    "accuracy": 0.9,
                    "f1_macro": 0.8,
                },
                "test_metrics": {
                    "accuracy": 0.88,
                },
            }
        ],
    }


@pytest.mark.usefixtures("missing_model")
async def test_process_returns_503_without_model(client: AsyncClient) -> None:
    response = await client.post("/process", json={"model": "champion", "texts": ["Python"]})

    assert response.status_code == 503
    assert response.json() == {"detail": "No models are loaded"}


@pytest.mark.usefixtures("missing_model")
async def test_model_endpoint_still_lists_registry_without_loaded_model(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "resume_classifier.model.list_models",
        lambda settings: [],
    )
    response = await client.get("/api/v1/model")
    assert response.status_code == 503


@pytest.mark.usefixtures("missing_model")
async def test_service_endpoints_work_without_model(client: AsyncClient) -> None:
    assert (await client.get("/healthz")).status_code == 200
    assert (await client.get("/api/v1/version")).status_code == 200
    assert (await client.get("/api/v1/health")).status_code == 200


async def test_process_evaluates_dataset_and_selected_metrics(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import pandas as pd

    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    pd.DataFrame(
        {
            "record_id": ["1", "2", "3"],
            "text": ["Python", "SQL", "Go"],
            "category": ["backend", "backend", "backend"],
        }
    ).to_csv(evaluation_dir / "test_v1.csv", index=False)
    settings.evaluation_data_dir = str(evaluation_dir)

    response = await client.post(
        "/process",
        json={
            "model": "champion",
            "dataset": "v1",
            "metrics": ["accuracy", "f1_macro"],
            "sample_size": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == {"version": "v1", "rows": 3}
    assert body["metrics"] == {"accuracy": 1.0, "f1_macro": 1.0}
    assert body["predictions"] == [
        {"index": 0, "category": "backend", "expected": "backend"},
        {"index": 1, "category": "backend", "expected": "backend"},
    ]
