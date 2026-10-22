import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

import asyncpg
import structlog
from faststream import AckPolicy, Context, ContextRepo
from faststream.asgi import AsgiFastStream, make_ping_asgi
from faststream.exceptions import RejectMessage
from faststream.rabbit import Channel, RabbitBroker

from resume_classifier.config import get_settings
from resume_classifier.db import create_pool, create_schema, save_result
from resume_classifier.logs import configure_logging
from resume_classifier.model import LoadedModel, load_models, predict_labels
from resume_classifier.queues import declare_queues, tasks_queue
from resume_classifier.schemas import TaskMessage

logger = structlog.get_logger(__name__)

settings = get_settings()
broker = RabbitBroker(
    settings.rabbitmq_url.get_secret_value(),
    default_channel=Channel(prefetch_count=settings.worker_prefetch),
)
executor = ThreadPoolExecutor(max_workers=settings.worker_threads)


def infer(model: LoadedModel, text: str) -> str:
    time.sleep(settings.inference_seconds)
    return str(predict_labels(model, [text])[0])


@broker.subscriber(tasks_queue(settings), ack_policy=AckPolicy.REJECT_ON_ERROR)
async def handle(
    task: TaskMessage,
    models: Annotated[dict[str, LoadedModel], Context()],
    pool: Annotated[asyncpg.Pool, Context()],
) -> None:
    try:
        model = models[task.model]
        category = await asyncio.get_running_loop().run_in_executor(
            executor, infer, model, task.text
        )
        await save_result(pool, task, category, model.info.version)
    except Exception as exc:
        logger.warning("task_failed", task_id=str(task.id), error=repr(exc))
        raise RejectMessage(requeue=True) from exc


async def startup(context: ContextRepo) -> None:
    configure_logging(settings.log_level, settings.log_json)
    models = await asyncio.to_thread(load_models, settings)
    if not models:
        raise RuntimeError("No models are loaded from MLflow")
    pool = await create_pool(settings)
    await create_schema(pool)
    context.set_global("models", models)
    context.set_global("pool", pool)
    await broker.connect()
    await declare_queues(broker, settings)
    logger.info(
        "worker_started",
        threads=settings.worker_threads,
        prefetch=settings.worker_prefetch,
        inference_seconds=settings.inference_seconds,
        model_aliases=sorted(models),
    )


async def shutdown(pool: Annotated[asyncpg.Pool, Context()]) -> None:
    executor.shutdown()
    await pool.close()
    logger.info("worker_stopped")


app = AsgiFastStream(
    broker,
    asgi_routes=[("/health", make_ping_asgi(broker, timeout=5.0))],
    on_startup=[startup],
    after_shutdown=[shutdown],
)
