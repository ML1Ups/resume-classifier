import logging
import socket
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import pytest
import structlog
from httpx import ASGITransport, AsyncClient
from sklearn.dummy import DummyClassifier

from resume_classifier import main
from resume_classifier.config import Settings
from resume_classifier.main import create_app
from resume_classifier.model import LoadedModel
from resume_classifier.schemas import ModelInfo


@asynccontextmanager
async def running_client(settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        yield client


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def model() -> LoadedModel:
    pipeline = DummyClassifier(strategy="constant", constant="backend").fit([[0]], ["backend"])
    info = ModelInfo(
        name="resume-classifier",
        alias="champion",
        version="1",
        run_id="unit-run",
        model_uri="models:/resume-classifier/1",
    )
    return LoadedModel(pipeline=pipeline, info=info)


@pytest.fixture(autouse=True)
def loaded_model(monkeypatch: pytest.MonkeyPatch, model: LoadedModel) -> None:
    monkeypatch.setattr(main, "load_model", lambda settings: model)


@pytest.fixture
def missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "load_model", lambda settings: None)


@pytest.fixture
def unreachable_settings(settings: Settings) -> Settings:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        free_port = sock.getsockname()[1]
    return settings.model_copy(update={"postgres_host": "127.0.0.1", "postgres_port": free_port})


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    async with running_client(settings) as client:
        yield client


@pytest.fixture
async def unreachable_client(unreachable_settings: Settings) -> AsyncIterator[AsyncClient]:
    async with running_client(unreachable_settings) as client:
        yield client


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    level = root.level
    yield
    for handler in root.handlers[:]:
        if isinstance(handler.formatter, structlog.stdlib.ProcessorFormatter):
            root.removeHandler(handler)
    root.setLevel(level)
    structlog.reset_defaults()
