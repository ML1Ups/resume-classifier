import asyncio
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import pairwise
from uuid import uuid4

import asyncpg
import pytest
from aiormq.exceptions import DeliveryError
from faststream import AckPolicy
from faststream.exceptions import RejectMessage
from faststream.rabbit import RabbitBroker
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from testcontainers.community.rabbitmq import RabbitMqContainer

from resume_classifier.config import Settings
from resume_classifier.main import create_app
from resume_classifier.queues import DEAD_LETTER_QUEUE, declare_queues, tasks_queue
from resume_classifier.schemas import TaskMessage
from resume_classifier.source import publish_until_accepted

RABBITMQ_IMAGE = "rabbitmq:4.3-management-alpine"


def new_task() -> TaskMessage:
    return TaskMessage(id=uuid4(), text="Python", model="champion", created_at=datetime.now(UTC))


async def publish_until_rejected(broker: RabbitBroker, queue: str) -> int:
    for accepted in range(10):
        try:
            await broker.publish(new_task(), queue=queue, persist=True)
        except DeliveryError:
            return accepted
    return 10


def retry_gaps(moments: list[float]) -> list[float]:
    return [round(later - earlier, 1) for earlier, later in pairwise(moments)]


@pytest.fixture(scope="module")
def rabbitmq_url() -> Iterator[str]:
    with RabbitMqContainer(RABBITMQ_IMAGE) as rabbitmq:
        host, port = rabbitmq.get_container_host_ip(), rabbitmq.get_exposed_port(5672)
        yield f"amqp://guest:guest@{host}:{port}/"


@pytest.fixture
def bounded(settings: Settings, rabbitmq_url: str) -> Settings:
    return settings.model_copy(
        update={
            "rabbitmq_url": SecretStr(rabbitmq_url),
            "tasks_queue": f"tasks-{uuid4().hex[:8]}",
            "queue_max_length": 3,
        }
    )


async def test_full_queue_rejects_publishes_and_source_waits_for_free_space(
    bounded: Settings,
) -> None:
    async with RabbitBroker(bounded.rabbitmq_url.get_secret_value(), logger=None) as broker:
        await declare_queues(broker, bounded)
        accepted = await publish_until_rejected(broker, bounded.tasks_queue)
        assert bounded.queue_max_length <= accepted <= bounded.queue_max_length + 1

        waiting = asyncio.create_task(
            publish_until_accepted(broker, new_task(), bounded.tasks_queue)
        )
        await asyncio.sleep(1)
        assert not waiting.done()

        queue = await broker.declare_queue(tasks_queue(bounded))
        for _ in range(2):
            await queue.get(no_ack=True)
        assert await asyncio.wait_for(waiting, timeout=10) > 0


async def test_api_sheds_load_with_503_and_keeps_no_pending_tasks(
    bounded: Settings,
    pool: asyncpg.Pool,
) -> None:
    app = create_app(bounded)
    transport = ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        responses = [await client.post("/api/v1/tasks", json={"text": "Python"}) for _ in range(6)]

    accepted = [response for response in responses if response.status_code == 202]
    assert responses[-1].status_code == 503
    assert responses[-1].headers["Retry-After"] == str(bounded.retry_after_seconds)
    assert bounded.queue_max_length <= len(accepted) <= bounded.queue_max_length + 1
    async with pool.acquire() as connection:
        stored = await connection.fetchval("SELECT count(*) FROM tasks WHERE status = 'PENDING'")
    assert stored == len(accepted)


async def test_failing_and_invalid_messages_end_up_in_dead_letter_queue(
    bounded: Settings,
) -> None:
    broker = RabbitBroker(bounded.rabbitmq_url.get_secret_value(), logger=None)
    deliveries = []
    delivered_at = []

    @broker.subscriber(tasks_queue(bounded), ack_policy=AckPolicy.REJECT_ON_ERROR)
    async def always_fails(task: TaskMessage) -> None:
        deliveries.append(task.id)
        delivered_at.append(time.monotonic())
        raise RejectMessage(requeue=True)

    async with broker:
        await declare_queues(broker, bounded)
        await broker.start()
        await broker.publish(new_task(), queue=bounded.tasks_queue)
        await broker.publish({"id": "not-a-uuid"}, queue=bounded.tasks_queue)
        await asyncio.sleep(5)
        dead_letters = await broker.declare_queue(DEAD_LETTER_QUEUE)
        reasons = sorted(
            [
                (await dead_letters.get(no_ack=True)).headers["x-death"][0]["reason"]
                for _ in range(2)
            ]
        )

    assert len(deliveries) == bounded.delivery_limit + 1
    assert retry_gaps(delivered_at) == [0.5, 1.0, 1.5]
    assert reasons == ["delivery_limit", "rejected"]
