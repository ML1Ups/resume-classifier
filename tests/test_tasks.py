from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import asyncpg
import pytest
from aiormq.exceptions import DeliveryError
from fastapi import FastAPI
from httpx import AsyncClient
from pamqp.commands import Basic

from resume_classifier.db import save_result
from resume_classifier.schemas import TaskMessage


@pytest.fixture
def publish(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    publish = AsyncMock()
    monkeypatch.setattr(app.state.broker, "publish", publish)
    return publish


async def count_tasks(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as connection:
        return await connection.fetchval("SELECT count(*) FROM tasks")


async def test_create_task_stores_pending_task_and_publishes_it(
    client: AsyncClient,
    pool: asyncpg.Pool,
    publish: AsyncMock,
) -> None:
    response = await client.post("/api/v1/tasks", json={"text": "Python FastAPI"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "PENDING"
    assert body["model"] == "champion"
    assert response.headers["Location"] == f"/api/v1/tasks/{body['id']}"
    message = publish.await_args.args[0]
    assert message.id == UUID(body["id"])
    assert message.text == "Python FastAPI"
    assert publish.await_args.kwargs == {"queue": "tasks", "persist": True}
    async with pool.acquire() as connection:
        status = await connection.fetchval("SELECT status FROM tasks WHERE id = $1", message.id)
    assert status == "PENDING"


async def test_full_queue_returns_503_and_leaves_no_pending_task(
    client: AsyncClient,
    pool: asyncpg.Pool,
    publish: AsyncMock,
) -> None:
    publish.side_effect = DeliveryError(None, Basic.Nack(delivery_tag=1))

    response = await client.post("/api/v1/tasks", json={"text": "Python FastAPI"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Task queue is full"}
    assert response.headers["Retry-After"] == "13"
    assert await count_tasks(pool) == 0


@pytest.mark.usefixtures("publish")
async def test_task_status_shows_result_written_by_worker(
    client: AsyncClient,
    pool: asyncpg.Pool,
) -> None:
    created = (await client.post("/api/v1/tasks", json={"text": "Python"})).json()
    task = TaskMessage(
        id=created["id"], text="Python", model="champion", created_at=created["created_at"]
    )
    await save_result(pool, task, "backend", "1")

    response = await client.get(f"/api/v1/tasks/{created['id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "DONE"
    assert body["category"] == "backend"
    assert body["model_version"] == "1"
    assert body["finished_at"] is not None


@pytest.mark.usefixtures("pool")
async def test_unknown_task_returns_404(client: AsyncClient) -> None:
    response = await client.get(f"/api/v1/tasks/{uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Task not found"}


@pytest.mark.parametrize(
    "payload",
    [{}, {"text": "   "}, {"text": "x" * 20001}, {"text": "Python", "unexpected": True}],
)
async def test_create_task_rejects_invalid_payloads(
    client: AsyncClient,
    payload: dict[str, object],
) -> None:
    response = await client.post("/api/v1/tasks", json=payload)

    assert response.status_code == 422


async def test_create_task_rejects_unknown_model(
    client: AsyncClient,
    pool: asyncpg.Pool,
    publish: AsyncMock,
) -> None:
    response = await client.post("/api/v1/tasks", json={"text": "Python", "model": "missing"})

    assert response.status_code == 404
    publish.assert_not_awaited()
    assert await count_tasks(pool) == 0


@pytest.mark.usefixtures("missing_model")
async def test_create_task_returns_503_without_models(client: AsyncClient) -> None:
    response = await client.post("/api/v1/tasks", json={"text": "Python"})

    assert response.status_code == 503


async def test_redelivered_result_does_not_overwrite_finished_task(pool: asyncpg.Pool) -> None:
    task = TaskMessage(id=uuid4(), text="Python", model="champion", created_at=datetime.now(UTC))

    await save_result(pool, task, "backend", "1")
    await save_result(pool, task, "frontend", "2")

    async with pool.acquire() as connection:
        rows = await connection.fetch("SELECT category, model_version FROM tasks")
    assert [tuple(row) for row in rows] == [("backend", "1")]
