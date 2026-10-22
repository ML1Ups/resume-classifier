from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import structlog
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from resume_classifier.config import Settings
from resume_classifier.schemas import ModelInfo, Prediction

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Pipeline
    info: ModelInfo


def metrics_with_prefix(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {
        key.removeprefix(prefix): value for key, value in metrics.items() if key.startswith(prefix)
    }


def model_info(client: MlflowClient, version: ModelVersion, alias: str | None) -> ModelInfo:
    number = str(version.version)
    metrics = client.get_run(version.run_id).data.metrics
    return ModelInfo(
        name=version.name,
        alias=alias,
        version=number,
        run_id=version.run_id,
        model_uri=f"models:/{version.name}/{number}",
        candidate=version.tags.get("candidate"),
        dataset_version=version.tags.get("dataset.version"),
        train_rows=version.tags.get("dataset.train_rows"),
        validation_metrics=metrics_with_prefix(metrics, "validation_"),
        test_metrics=metrics_with_prefix(metrics, "test_"),
    )


def load_models(settings: Settings) -> dict[str, LoadedModel]:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    name = settings.mlflow_model_name
    try:
        aliases = client.get_registered_model(name).aliases
        pipelines = {
            version: mlflow.sklearn.load_model(f"models:/{name}/{version}")
            for version in set(aliases.values())
        }
        models = {
            alias: LoadedModel(
                pipeline=pipelines[version],
                info=model_info(client, client.get_model_version(name, version), alias),
            )
            for alias, version in aliases.items()
        }
    except (MlflowException, OSError) as exc:
        logger.warning("models_not_loaded", model=name, error=repr(exc))
        return {}
    for alias, model in models.items():
        logger.info(
            "model_loaded",
            model=name,
            alias=alias,
            version=model.info.version,
            run_id=model.info.run_id,
        )
    return models


def list_models(settings: Settings) -> list[ModelInfo]:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    name = settings.mlflow_model_name
    aliases = {
        version: alias for alias, version in client.get_registered_model(name).aliases.items()
    }
    versions = sorted(
        client.search_model_versions(f"name='{name}'"), key=lambda version: int(version.version)
    )
    return [model_info(client, version, aliases.get(str(version.version))) for version in versions]


def predict_labels(model: LoadedModel, texts: list[str]) -> np.ndarray:
    return model.pipeline.predict(pd.DataFrame({"text": texts}))


def predict(model: LoadedModel, texts: list[str]) -> list[Prediction]:
    labels = predict_labels(model, texts)
    return [Prediction(index=index, category=str(label)) for index, label in enumerate(labels)]
