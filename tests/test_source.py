from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from aiormq.exceptions import DeliveryError
from pamqp.commands import Basic

from resume_classifier import source
from resume_classifier.config import Settings
from resume_classifier.schemas import TaskMessage

REJECTED = DeliveryError(None, Basic.Nack(delivery_tag=1))


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def record(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(source.asyncio, "sleep", record)
    return delays


async def test_source_waits_with_growing_delay_until_broker_accepts(sleeps: list[float]) -> None:
    broker = AsyncMock()
    broker.publish.side_effect = [REJECTED] * 7 + [None]

    task = TaskMessage(id=uuid4(), text="Python", model="champion", created_at=datetime.now(UTC))

    rejected = await source.publish_until_accepted(broker, task, "tasks")

    assert rejected == 7
    assert broker.publish.await_count == 8
    assert sleeps == [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 2.0]


@pytest.mark.usefixtures("restore_logging")
async def test_source_publishes_every_row_once(
    tmp_path: Path,
    settings: Settings,
    sleeps: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "rows.csv"
    data.write_text("text,category\nPython,backend\nSQL,backend\nFigma,design\n", "utf-8")
    broker = MagicMock()
    broker.__aenter__ = AsyncMock(return_value=broker)
    broker.__aexit__ = AsyncMock(return_value=None)
    broker.declare_queue = AsyncMock()
    broker.publish = AsyncMock(side_effect=[None, REJECTED, None, None])
    monkeypatch.setattr(source, "RabbitBroker", MagicMock(return_value=broker))

    await source.run(settings, data, "champion")

    texts = [call.args[0].text for call in broker.publish.await_args_list]
    assert texts == ["Python", "SQL", "SQL", "Figma"]
    assert sleeps == [0.05]
