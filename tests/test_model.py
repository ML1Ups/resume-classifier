from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from faststream.rabbit import TestRabbitBroker
from mlflow.exceptions import MlflowException

from resume_classifier import model as module
from resume_classifier.config import Settings
from resume_classifier.main import create_app
from resume_classifier.model import LoadedModel


def registry_version(version: str, dataset: str) -> SimpleNamespace:
    return SimpleNamespace(
        name="resume-classifier",
        version=version,
        run_id=f"run-{version}",
        tags={"candidate": "logreg_words", "dataset.version": dataset, "dataset.train_rows": "100"},
    )


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    client.get_registered_model.return_value.aliases = {
        "champion": "7",
        "demo-v1": "7",
        "demo-v2": "8",
    }
    client.get_model_version.side_effect = lambda name, version: registry_version(
        version, "v1" if version == "7" else "v2"
    )
    client.search_model_versions.return_value = [
        registry_version("8", "v2"),
        registry_version("7", "v1"),
    ]
    client.get_run.return_value.data.metrics = {
        "validation_f1_macro": 0.8,
        "test_f1_macro": 0.7,
        "fit_seconds": 1.5,
    }
    monkeypatch.setattr(module, "MlflowClient", Mock(return_value=client))
    return client


def test_load_models_loads_every_alias_and_each_version_once(
    settings: Settings,
    registry: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = Mock(side_effect=lambda uri: f"pipeline {uri}")
    monkeypatch.setattr(module.mlflow.sklearn, "load_model", loader)

    loaded = module.load_models(settings)

    assert sorted(loaded) == ["champion", "demo-v1", "demo-v2"]
    assert loader.call_count == 2
    assert loaded["champion"].pipeline == "pipeline models:/resume-classifier/7"
    assert loaded["demo-v2"].info.dataset_version == "v2"
    assert loaded["demo-v2"].info.train_rows == 100
    assert loaded["demo-v1"].info.validation_metrics == {"f1_macro": 0.8}
    assert loaded["demo-v1"].info.test_metrics == {"f1_macro": 0.7}


def test_load_models_returns_nothing_when_registry_is_unavailable(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.get_registered_model.side_effect = MlflowException("registry is down")
    monkeypatch.setattr(module, "MlflowClient", Mock(return_value=client))

    assert module.load_models(settings) == {}


def test_list_models_returns_versions_in_order_with_aliases(
    settings: Settings,
    registry: Mock,
) -> None:
    models = module.list_models(settings)

    assert [model.version for model in models] == ["7", "8"]
    assert models[1].alias == "demo-v2"
    assert models[1].model_uri == "models:/resume-classifier/8"


async def test_startup_keeps_loaded_models_in_app_state(
    settings: Settings,
    model: LoadedModel,
) -> None:
    app = create_app(settings)

    async with TestRabbitBroker(app.state.broker), app.router.lifespan_context(app):
        assert app.state.models == {"champion": model}
