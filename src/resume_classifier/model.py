"""One immutable model snapshot per application process."""

import asyncio
from dataclasses import dataclass
from typing import Any

import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from starlette.concurrency import run_in_threadpool

from resume_classifier.config import Settings
from resume_classifier.schemas import ModelInfo, Prediction, ProcessResponse


@dataclass
class ModelService:
    model: Any
    info: ModelInfo
    limiter: asyncio.Semaphore

    @classmethod
    def load(cls, settings: Settings) -> "ModelService":
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_registry_uri(settings.mlflow_tracking_uri)
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        # Resolve the alias once, then load an immutable version. This avoids a race
        # if someone moves the alias between reading metadata and downloading weights.
        version = client.get_model_version_by_alias(
            settings.mlflow_model_name,
            settings.mlflow_model_alias,
        )
        uri = f"models:/{settings.mlflow_model_name}/{version.version}"
        model = mlflow.sklearn.load_model(uri)
        info = ModelInfo(
            name=settings.mlflow_model_name,
            alias=settings.mlflow_model_alias,
            version=str(version.version),
            run_id=version.run_id,
            model_uri=uri,
        )
        return cls(
            model=model,
            info=info,
            limiter=asyncio.Semaphore(
                settings.inference_max_concurrency,
            ),
        )

    def _predict(self, texts: list[str]) -> list[Prediction]:
        # This is the same DataFrame schema as the logged MLflow signature.
        labels = self.model.predict(pd.DataFrame({"text": texts}))
        if len(labels) != len(texts):
            raise ValueError("Model returned an unexpected number of predictions")
        return [Prediction(index=i, category=str(label)) for i, label in enumerate(labels)]

    async def predict(self, texts: list[str]) -> ProcessResponse:
        # sklearn is synchronous: it must not run on FastAPI's event loop.
        async with self.limiter:
            predictions = await run_in_threadpool(self._predict, texts)
        return ProcessResponse(predictions=predictions, model=self.info)
