import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from resume_classifier.config import Settings
from resume_classifier.model import ModelService
from resume_classifier.schemas import ModelInfo

REAL_LOAD = ModelService.load


def test_alias_resolved_once_and_immutable_version_loaded(settings, monkeypatch):
    import resume_classifier.model as module

    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(version="7", run_id="run-7")
    factory = Mock(return_value=client)
    loader = Mock(return_value=Mock())
    monkeypatch.setattr(module, "MlflowClient", factory)
    monkeypatch.setattr(module.mlflow.sklearn, "load_model", loader)
    service = REAL_LOAD(settings)
    client.get_model_version_by_alias.assert_called_once_with("resume-classifier", "champion")
    loader.assert_called_once_with("models:/resume-classifier/7")
    assert service.info.version == "7"
    assert service.info.run_id == "run-7"


async def test_startup_loads_once_and_closes_pool_on_shutdown(settings, monkeypatch):
    from resume_classifier import main

    service = Mock(info=SimpleNamespace(version="5"))
    loader = Mock(return_value=service)
    pool = Mock(close=AsyncMock())
    monkeypatch.setattr(main, "create_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(main.ModelService, "load", loader)
    app = main.create_app(settings)
    async with app.router.lifespan_context(app):
        assert app.state.model_service is service
        loader.assert_called_once_with(settings)
    pool.close.assert_awaited_once()


async def test_missing_alias_fails_startup_and_closes_pool(settings, monkeypatch):
    from resume_classifier import main

    pool = Mock(close=AsyncMock())
    monkeypatch.setattr(main, "create_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(main.ModelService, "load", Mock(side_effect=RuntimeError("no alias")))
    app = main.create_app(settings)
    with pytest.raises(RuntimeError, match="no alias"):
        async with app.router.lifespan_context(app):
            pytest.fail("Application must not start without a model")
    pool.close.assert_awaited_once()


async def test_wrong_model_output_size_is_rejected():
    info = ModelInfo(name="a", alias="b", version="1", run_id=None, model_uri="models:/a/1")
    service = ModelService(Mock(predict=Mock(return_value=[])), info, asyncio.Semaphore(1))
    with pytest.raises(ValueError, match="number of predictions"):
        await service.predict(["text"])


def test_config_rejects_zero_concurrency(settings: Settings):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(**{**settings.model_dump(), "inference_max_concurrency": 0})
