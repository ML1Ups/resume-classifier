from faststream.rabbit import QueueType, RabbitBroker, RabbitQueue

from resume_classifier.config import Settings

DEAD_LETTER_QUEUE = RabbitQueue("tasks.dlq", queue_type=QueueType.QUORUM)
RETRY_DELAY_MIN_MS = 500
RETRY_DELAY_MAX_MS = 5000


def tasks_queue(settings: Settings) -> RabbitQueue:
    arguments = {
        "x-delivery-limit": settings.delivery_limit,
        "x-dead-letter-exchange": "",
        "x-dead-letter-routing-key": DEAD_LETTER_QUEUE.name,
        "x-delayed-retry-type": "failed",
        "x-delayed-retry-min": RETRY_DELAY_MIN_MS,
        "x-delayed-retry-max": RETRY_DELAY_MAX_MS,
    }
    if settings.queue_max_length:
        arguments |= {"x-max-length": settings.queue_max_length, "x-overflow": "reject-publish"}
    return RabbitQueue(settings.tasks_queue, queue_type=QueueType.QUORUM, arguments=arguments)


async def declare_queues(broker: RabbitBroker, settings: Settings) -> None:
    await broker.declare_queue(DEAD_LETTER_QUEUE)
    await broker.declare_queue(tasks_queue(settings))
