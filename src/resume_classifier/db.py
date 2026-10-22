from uuid import UUID

import asyncpg

from resume_classifier.config import Settings
from resume_classifier.schemas import TaskMessage


async def create_pool(settings: Settings) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
        database=settings.postgres_db,
        min_size=0,
        max_size=settings.postgres_pool_max_size,
    )


async def fetch_postgres_version(pool: asyncpg.Pool) -> str:
    async with pool.acquire() as connection:
        return await connection.fetchval("SHOW server_version")


async def create_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection:
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id uuid PRIMARY KEY,
                text text NOT NULL,
                model text NOT NULL,
                status text NOT NULL,
                category text,
                model_version text,
                created_at timestamptz NOT NULL,
                finished_at timestamptz
            )
            """
        )


async def save_result(pool: asyncpg.Pool, task: TaskMessage, category: str, version: str) -> None:
    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO tasks (id, text, model, status, category, model_version, created_at,
                               finished_at)
            VALUES ($1, $2, $3, 'DONE', $4, $5, $6, now())
            ON CONFLICT (id) DO UPDATE
            SET status = 'DONE', category = EXCLUDED.category,
                model_version = EXCLUDED.model_version, finished_at = now()
            WHERE tasks.status = 'PENDING'
            """,
            task.id,
            task.text,
            task.model,
            category,
            version,
            task.created_at,
        )


async def fetch_task(pool: asyncpg.Pool, task_id: UUID) -> asyncpg.Record | None:
    async with pool.acquire() as connection:
        return await connection.fetchrow("SELECT * FROM tasks WHERE id = $1", task_id)
