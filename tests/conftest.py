import logging
import socket
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
import structlog
from httpx import ASGITransport, AsyncClient

from resume_classifier.config import Settings
from resume_classifier.main import create_app
from resume_classifier.model import ModelService
from resume_classifier.schemas import ModelInfo


@pytest.fixture(autouse=True)
def external_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests are offline; real DB and MLflow are checked by smoke tests."""
    from resume_classifier import main

    for name, value in {
        "POSTGRES_HOST": "localhost",
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "test",
        "POSTGRES_DB": "test",
    }.items():
        monkeypatch.setenv(name, value)

    async def fake_pool(settings: Settings):
        @asynccontextmanager
        async def acquire():
            if settings.postgres_host == "127.0.0.1":
                raise ConnectionRefusedError
            yield type("Connection", (), {"fetchval": AsyncMock(return_value="17.0")})()

        return type("Pool", (), {"acquire": staticmethod(acquire), "close": AsyncMock()})()

    import asyncio

    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="constant", constant="backend")
    model.fit([[0]], ["backend"])
    info = ModelInfo(
        name="resume-classifier",
        alias="champion",
        version="1",
        run_id="unit-run",
        model_uri="models:/resume-classifier/1",
    )
    monkeypatch.setattr(main, "create_pool", fake_pool)
    monkeypatch.setattr(
        ModelService,
        "load",
        lambda settings: ModelService(
            model=model,
            info=info,
            limiter=asyncio.Semaphore(2),
        ),
    )


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
