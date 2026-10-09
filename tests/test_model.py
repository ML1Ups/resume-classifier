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


async def test_startup_keeps_loaded_model_in_app_state(
    settings: Settings,
    model: LoadedModel,
) -> None:
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        assert app.state.model is model
