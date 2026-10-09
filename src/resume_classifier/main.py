from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from resume_classifier import __version__
from resume_classifier.api import healthz, process, v1
from resume_classifier.config import Settings, get_settings
from resume_classifier.db import create_pool
from resume_classifier.logs import log_requests
from resume_classifier.model import load_model

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.db_pool = await create_pool(app.state.settings)
    app.state.model = await run_in_threadpool(load_model, app.state.settings)
    logger.info("application_started", version=__version__)
    yield
    await app.state.db_pool.close()
    logger.info("application_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Resume Classifier", version=__version__, lifespan=lifespan)
    app.state.settings = settings or get_settings()
    app.middleware("http")(log_requests)
    app.include_router(healthz.router)
    app.include_router(process.router)
    app.include_router(v1.router)
    return app
