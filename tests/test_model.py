from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException

from resume_classifier import model as module
from resume_classifier.config import Settings
from resume_classifier.main import create_app
from resume_classifier.model import LoadedModel


def test_load_model_resolves_alias_once_and_loads_that_version(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="7", run_id="run-7")
    pipeline = Mock()
    loader = Mock(return_value=pipeline)
    monkeypatch.setattr(module, "MlflowClient", Mock(return_value=client))
    monkeypatch.setattr(module.mlflow.sklearn, "load_model", loader)

    loaded = module.load_model(settings)

    client.get_model_version_by_alias.assert_called_once_with("resume-classifier", "champion")
    loader.assert_called_once_with("models:/resume-classifier/7")
    assert loaded.pipeline is pipeline
    assert loaded.info.version == "7"
    assert loaded.info.run_id == "run-7"
    assert loaded.info.model_uri == "models:/resume-classifier/7"


def test_load_model_returns_none_when_alias_is_missing(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.get_model_version_by_alias.side_effect = MlflowException("alias not found")
    monkeypatch.setattr(module, "MlflowClient", Mock(return_value=client))

    assert module.load_model(settings) is None


def test_load_models_loads_all_registry_aliases(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.get_registered_model.return_value.aliases = {"demo-v1": "7", "demo-v2": "8"}
    client.get_model_version_by_alias.side_effect = [
        SimpleNamespace(
            version="7",
            run_id="run-7",
            tags={
                "candidate": "logreg_words",
                "dataset.version": "v1",
                "dataset.train_rows": "100",
            },
        ),
        SimpleNamespace(
            version="8",
            run_id="run-8",
            tags={"candidate": "svm_bigrams", "dataset.version": "v2", "dataset.train_rows": "10"},
        ),
    ]
    client.get_run.return_value.data.metrics = {"validation_f1_macro": 0.8}
    pipeline = Mock()
    monkeypatch.setattr(module, "MlflowClient", Mock(return_value=client))
    monkeypatch.setattr(module.mlflow.sklearn, "load_model", Mock(return_value=pipeline))

    loaded = module.load_models(settings)

    assert sorted(loaded) == ["demo-v1", "demo-v2"]
    assert loaded["demo-v1"].info.dataset_version == "v1"
    assert loaded["demo-v2"].info.candidate == "svm_bigrams"


async def test_startup_keeps_loaded_models_in_app_state(
    settings: Settings,
    model: LoadedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "load_models", lambda settings: {"champion": model})
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        assert app.state.models == {"champion": model}
        assert app.state.model is model
