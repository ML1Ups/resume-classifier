import argparse
import asyncio
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import asyncpg
import httpx

MODES = {
    "limited": {"QUEUE_MAX_LENGTH": "100", "WORKER_PREFETCH": "4"},
    "unlimited": {"QUEUE_MAX_LENGTH": "0", "WORKER_PREFETCH": "0"},
}
UNITS = {"B": 1 / 1024**2, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024}
SAMPLE_SECONDS = 2


def compose(env: dict[str, str], *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["docker", "compose", *args],
        env=env,
        check=check,
        capture_output=True,
        text=True,
    ).stdout


def source_progress(env: dict[str, str], started_at: float) -> list[dict[str, float]]:
    logs = compose(env, "--profile", "load", "logs", "--no-color", "--no-log-prefix", "source")
    events = [json.loads(line) for line in logs.splitlines() if '"source_progress"' in line]
    return [
        {
            "seconds": round(
                datetime.fromisoformat(event["timestamp"]).timestamp() - started_at, 1
            ),
            "rate": event["rate"],
            "published": event["published"],
            "rejected": event["rejected"],
        }
        for event in events
    ]


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def memory_mb(project: str) -> dict[str, float]:
    stats = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{json .}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    usage = {"worker": 0.0, "source": 0.0, "rabbitmq": 0.0}
    for line in stats.splitlines():
        container = json.loads(line)
        service = container["Name"].removeprefix(f"{project}-").rsplit("-", 1)[0]
        if service in usage:
            value = container["MemUsage"].split(" / ")[0]
            unit = value.lstrip("0123456789.")
            usage[service] += float(value.removesuffix(unit)) * UNITS[unit]
    return usage


async def sample(
    env: dict[str, str], client: httpx.AsyncClient, pool: asyncpg.Pool, workers: int, started: float
) -> dict[str, float]:
    queue = (await client.get("/api/queues/%2F/tasks")).json()
    async with pool.acquire() as connection:
        done, latency = await connection.fetchrow(
            "SELECT count(*), avg(extract(epoch FROM finished_at - created_at)) FROM tasks "
            "WHERE finished_at > now() - make_interval(secs => $1)",
            SAMPLE_SECONDS,
        )
    memory = memory_mb(env["COMPOSE_PROJECT_NAME"])
    return {
        "seconds": round(time.monotonic() - started, 1),
        "workers": workers,
        "ready": queue["messages_ready"],
        "unacked": queue["messages_unacknowledged"],
        "done_rate": done / SAMPLE_SECONDS,
        "latency_seconds": float(latency or 0),
        "worker_mb": round(memory["worker"], 1),
        "source_mb": round(memory["source"], 1),
        "rabbitmq_mb": round(memory["rabbitmq"], 1),
    }


async def run(mode: str, seconds: int, scale_at: int) -> tuple[list[dict], list[dict]]:
    env = os.environ | MODES[mode]
    compose(env, "stop", "source", "worker", "app")
    compose(env, "exec", "-T", "rabbitmq", "rabbitmqctl", "delete_queue", "tasks", check=False)
    compose(
        env,
        "exec",
        "-T",
        "db",
        "psql",
        "-U",
        env["POSTGRES_USER"],
        "-d",
        env["POSTGRES_DB"],
        "-c",
        "TRUNCATE tasks",
        check=False,
    )
    compose(env, "up", "-d", "--no-deps", "--wait", "--scale", "worker=1", "worker", "app")
    started_at = time.time()
    compose(env, "--profile", "load", "up", "-d", "--no-deps", "--force-recreate", "source")

    auth = (env["RABBITMQ_USER"], env["RABBITMQ_PASSWORD"])
    rows = []
    pool = await asyncpg.create_pool(
        host="127.0.0.1",
        port=int(env["POSTGRES_PORT"]),
        user=env["POSTGRES_USER"],
        password=env["POSTGRES_PASSWORD"],
        database=env["POSTGRES_DB"],
    )
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{env['RABBITMQ_UI_PORT']}", auth=auth
    ) as client:
        started = time.monotonic()
        workers = 1
        while (elapsed := time.monotonic() - started) < seconds:
            if workers == 1 and elapsed >= scale_at:
                compose(
                    env, "up", "-d", "--no-deps", "--no-recreate", "--scale", "worker=2", "worker"
                )
                workers = 2
            rows.append(await sample(env, client, pool, workers, started))
            print(rows[-1], flush=True)
            await asyncio.sleep(SAMPLE_SECONDS)
    await pool.close()
    compose(env, "stop", "source")
    return rows, source_progress(env, started_at)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the pipeline with and without limits")
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--scale-at", type=int, default=90)
    parser.add_argument("--output", type=Path, default=Path("docs/experiment"))
    args = parser.parse_args()
    rows, source_rows = asyncio.run(run(args.mode, args.seconds, args.scale_at))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / f"{args.mode}.csv", rows)
    write_csv(args.output / f"{args.mode}_source.csv", source_rows)


if __name__ == "__main__":
    main()
