"""Run a real HTTP MLflow + Registry + API check without Docker.

The app database is replaced with a pool stub here; the full Docker smoke check
also exercises PostgreSQL and the persistent artifact volume.
The model and MLflow are real in this check.
"""

import argparse
import asyncio
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
from mlflow import MlflowClient

from resume_classifier import main as app_module
from resume_classifier.config import Settings
from resume_classifier.ml.train import run_training


async def check_api(settings: Settings, alternate: str, client: MlflowClient) -> dict:
    pool = type("Pool", (), {"close": AsyncMock()})()
    original_pool = app_module.create_pool
    app_module.create_pool = AsyncMock(return_value=pool)
    try:
        app = app_module.create_app(settings)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                payload = {
                    "texts": [
                        "Python FastAPI PostgreSQL API транзакции",
                        "React TypeScript CSS компоненты интерфейса",
                    ]
                }
                response = await http.post("/process", json=payload)
                response.raise_for_status()
                before = response.json()
                old_version = before["model"]["version"]
                client.set_registered_model_alias(settings.mlflow_model_name, "champion", alternate)
                response = await http.post("/process", json=payload)
                response.raise_for_status()
                if response.json()["model"]["version"] != old_version:
                    raise AssertionError("Running app must keep its startup model snapshot")
                if (await http.post("/process", json={"texts": [" "]})).status_code != 422:
                    raise AssertionError("Blank input must be rejected")
        restarted = app_module.create_app(settings)
        async with restarted.router.lifespan_context(restarted):
            transport = httpx.ASGITransport(app=restarted)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                response = await http.post("/process", json=payload)
                response.raise_for_status()
                after = response.json()
                if after["model"]["version"] != alternate:
                    raise AssertionError("Restarted app must resolve the new alias version")
        client.set_registered_model_alias(settings.mlflow_model_name, "champion", old_version)
        return {
            "before_alias_change": before,
            "after_restart": after,
            "existing_process_kept_version": old_version,
            "blank_input_status": 422,
            "postgres": "stubbed in this check",
        }
    finally:
        app_module.create_pool = original_pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--data", type=Path, default=Path("data/resumes_demo.csv"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="resume-integration-") as temp:
        root = Path(temp)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        uri = f"http://127.0.0.1:{port}"
        command = [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            f"sqlite:///{root / 'tracking.db'}",
            "--artifacts-destination",
            str(root / "artifacts"),
            "--serve-artifacts",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
        ]
        with (root / "server.log").open("w") as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603
            try:
                with httpx.Client(timeout=2, trust_env=False) as http:
                    for _ in range(100):
                        if server.poll() is not None:
                            raise RuntimeError((root / "server.log").read_text())
                        try:
                            response = http.get(uri + "/health")
                            if response.status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(0.2)
                    else:
                        raise TimeoutError("MLflow server did not start")
                summary = run_training(args.data, uri, output=args.output)
                settings = Settings(
                    _env_file=None,
                    postgres_host="unused",
                    postgres_user="unused",
                    postgres_password="unused",
                    postgres_db="unused",
                    mlflow_tracking_uri=uri,
                )
                client = MlflowClient(tracking_uri=uri)
                # Keep a small, reviewable artifact snapshot alongside the source archive.
                for run_id, artifact_path in [
                    (summary["eda_run_id"], "eda"),
                    (summary["champion"]["run_id"], "data"),
                    (summary["champion"]["run_id"], "validation"),
                    (summary["champion"]["run_id"], "test"),
                ]:
                    client.download_artifacts(
                        run_id, artifact_path, dst_path=str(args.output.resolve())
                    )
                # Preparation is logged by the EDA run; preserve it in the snapshot too.
                (args.output / "data" / "preparation.json").write_text(
                    json.dumps(summary["dataset"]["preparation"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                alternate = next(
                    row["version"]
                    for row in summary["runs"]
                    if row["version"] and row["version"] != summary["champion"]["version"]
                )
                evidence = asyncio.run(check_api(settings, alternate, client))
                evidence["mlflow_backend"] = "SQLite temporary database"
                evidence["mlflow_artifacts"] = "local artifact store through HTTP proxy"
                evidence["docker_compose_executed"] = False
                (args.output / "integration_check.json").write_text(
                    json.dumps(evidence, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(json.dumps(evidence, ensure_ascii=False, indent=2))
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


if __name__ == "__main__":
    main()
