from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from aiormq.exceptions import DeliveryError
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from resume_classifier.api.dependencies import loaded_models, selected_model
from resume_classifier.db import fetch_task
from resume_classifier.model import LoadedModel
from resume_classifier.schemas import TaskMessage, TaskRequest, TaskResponse

router = APIRouter(tags=["tasks"])


@router.post(
    "/tasks",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Model alias is not loaded"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Task queue is full"},
    },
)
async def create_task(
    payload: TaskRequest,
    request: Request,
    response: Response,
    models: Annotated[dict[str, LoadedModel], Depends(loaded_models)],
) -> TaskResponse:
    selected_model(payload.model, models)
    settings = request.app.state.settings
    task = TaskMessage(
        id=uuid4(), text=payload.text, model=payload.model, created_at=datetime.now(UTC)
    )
    async with request.app.state.db_pool.acquire() as connection, connection.transaction():
        await connection.execute(
            "INSERT INTO tasks (id, text, model, status, created_at) "
            "VALUES ($1, $2, $3, 'PENDING', $4)",
            task.id,
            task.text,
            task.model,
            task.created_at,
        )
        try:
            await request.app.state.broker.publish(task, queue=settings.tasks_queue, persist=True)
        except DeliveryError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Task queue is full",
                headers={"Retry-After": str(settings.retry_after_seconds)},
            ) from exc
    response.headers["Location"] = f"/api/v1/tasks/{task.id}"
    return TaskResponse(
        id=task.id,
        status="PENDING",
        model=task.model,
        category=None,
        model_version=None,
        created_at=task.created_at,
        finished_at=None,
    )


@router.get("/tasks/{task_id}", responses={status.HTTP_404_NOT_FOUND: {"description": "No task"}})
async def get_task(task_id: UUID, request: Request) -> TaskResponse:
    row = await fetch_task(request.app.state.db_pool, task_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Task not found")
    return TaskResponse.model_validate(dict(row))
