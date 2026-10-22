import argparse
import asyncio
import csv
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import structlog
from aiormq.exceptions import DeliveryError
from faststream.rabbit import RabbitBroker

from resume_classifier.config import Settings, get_settings
from resume_classifier.logs import configure_logging
from resume_classifier.queues import declare_queues
from resume_classifier.schemas import TaskMessage

FIRST_DELAY_SECONDS = 0.05
MAX_DELAY_SECONDS = 2.0
REPORT_SECONDS = 5.0

logger = structlog.get_logger(__name__)


async def publish_until_accepted(broker: RabbitBroker, task: TaskMessage, queue: str) -> int:
    rejected = 0
    delay = FIRST_DELAY_SECONDS
    while True:
        try:
            await broker.publish(task, queue=queue, persist=True)
        except DeliveryError:
            rejected += 1
            await asyncio.sleep(delay)
            delay = min(delay * 2, MAX_DELAY_SECONDS)
        else:
            return rejected


async def run(settings: Settings, data: Path, model: str) -> None:
    published = rejected = window_published = 0
    async with RabbitBroker(settings.rabbitmq_url.get_secret_value(), logger=None) as broker:
        await declare_queues(broker, settings)
        window_started = time.monotonic()
        with data.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                task = TaskMessage(
                    id=uuid4(), text=row["text"], model=model, created_at=datetime.now(UTC)
                )
                rejected += await publish_until_accepted(broker, task, settings.tasks_queue)
                published += 1
                window_published += 1
                elapsed = time.monotonic() - window_started
                if elapsed >= REPORT_SECONDS:
                    logger.info(
                        "source_progress",
                        published=published,
                        rejected=rejected,
                        rate=round(window_published / elapsed, 2),
                    )
                    window_started, window_published = time.monotonic(), 0
    logger.info("source_finished", published=published, rejected=rejected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish every CSV row as a scoring task")
    parser.add_argument("--data", type=Path, default=Path("data/evaluation/full.csv"))
    parser.add_argument("--model", default="champion")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    asyncio.run(run(settings, args.data, args.model))


if __name__ == "__main__":
    main()
