import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import asyncpg
import pytest
from faststream.rabbit import RabbitMessage, TestRabbitBroker

from resume_classifier import worker
from resume_classifier.model import LoadedModel
from resume_classifier.schemas import TaskMessage


def new_task() -> TaskMessage:
    return TaskMessage(id=uuid4(), text="Python", model="champion", created_at=datetime.now(UTC))


@pytest.fixture
async def broker(
    pool: asyncpg.Pool,
    model: LoadedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[TestRabbitBroker]:
    monkeypatch.setattr(worker.settings, "inference_seconds", 0.0)
    async with TestRabbitBroker(worker.broker) as broker:
        broker.context.set_global("models", {"champion": model})
        broker.context.set_global("pool", pool)
        yield broker


@pytest.fixture
def reject(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    reject = AsyncMock()
    monkeypatch.setattr(RabbitMessage, "reject", reject)
    return reject


async def test_worker_writes_result_once_when_message_is_redelivered(
    broker: TestRabbitBroker,
    pool: asyncpg.Pool,
) -> None:
    task = new_task()

    await broker.publish(task, queue="tasks")
    async with pool.acquire() as connection:
        first = await connection.fetchrow("SELECT * FROM tasks WHERE id = $1", task.id)
    await broker.publish(task, queue="tasks")

    async with pool.acquire() as connection:
        rows = await connection.fetch("SELECT * FROM tasks")
    assert len(rows) == 1
    assert rows[0]["status"] == "DONE"
    assert rows[0]["category"] == "backend"
    assert rows[0]["model_version"] == "1"
    assert rows[0]["finished_at"] == first["finished_at"]


async def test_worker_requeues_task_when_result_cannot_be_saved(
    broker: TestRabbitBroker,
    reject: AsyncMock,
) -> None:
    failing_pool = Mock()
    failing_pool.acquire.side_effect = OSError("database is down")
    broker.context.set_global("pool", failing_pool)

    await broker.publish(new_task(), queue="tasks")

    reject.assert_awaited_once_with(requeue=True)


async def test_worker_rejects_invalid_message_without_requeue(
    broker: TestRabbitBroker,
    reject: AsyncMock,
) -> None:
    with pytest.raises(Exception, match="validation error"):
        await broker.publish({"id": "not-a-uuid"}, queue="tasks")

    reject.assert_awaited_once_with()


async def test_inference_runs_in_thread_pool_without_blocking_event_loop(
    broker: TestRabbitBroker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker.settings, "inference_seconds", 0.2)
    lags = []

    async def watch_event_loop() -> None:
        for _ in range(15):
            started = time.perf_counter()
            await asyncio.sleep(0.01)
            lags.append(time.perf_counter() - started - 0.01)

    started = time.perf_counter()
    await asyncio.gather(
        watch_event_loop(),
        *(broker.publish(new_task(), queue="tasks") for _ in range(worker.settings.worker_threads)),
    )

    assert time.perf_counter() - started < 0.6
    assert max(lags) < 0.1
