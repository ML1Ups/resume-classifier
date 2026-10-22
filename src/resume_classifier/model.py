from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import structlog
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from resume_classifier.config import Settings
from resume_classifier.schemas import ModelInfo, Prediction

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Pipeline
    info: ModelInfo


def _metrics_by_prefix(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {
        key.removeprefix(prefix): float(value)
        for key, value in metrics.items()
        if key.startswith(prefix)
    }


def _int_tag(tags: dict[str, str], name: str) -> int | None:
    value = tags.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _model_info(
    client: MlflowClient,
    model_name: str,
    alias: str | None,
    version: object,
) -> ModelInfo:
    tags = getattr(version, "tags", {}) or {}
    run_id = getattr(version, "run_id", None)
    run_metrics: dict[str, float] = {}
    if run_id:
        try:
            raw_metrics = getattr(getattr(client.get_run(run_id), "data", None), "metrics", {})
            run_metrics = dict(raw_metrics or {})
        except (AttributeError, MlflowException, TypeError):
            logger.warning("model_metrics_not_loaded", run_id=run_id)

    version_number = str(version.version)
    return ModelInfo(
        name=model_name,
        alias=alias,
        version=version_number,
        run_id=run_id,
        model_uri=f"models:/{model_name}/{version_number}",
        candidate=tags.get("candidate"),
        dataset_version=tags.get("dataset.version"),
        train_rows=_int_tag(tags, "dataset.train_rows"),
        validation_metrics=_metrics_by_prefix(run_metrics, "validation_"),
        test_metrics=_metrics_by_prefix(run_metrics, "test_"),
    )


def load_model(settings: Settings, alias: str | None = None) -> LoadedModel | None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    name = settings.mlflow_model_name
    selected_alias = alias or settings.mlflow_model_alias
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(name, selected_alias)
        uri = f"models:/{name}/{version.version}"
        pipeline = mlflow.sklearn.load_model(uri)
        info = _model_info(client, name, selected_alias, version)
    except (MlflowException, OSError) as exc:
        logger.warning(
            "model_not_loaded",
            model=name,
            alias=selected_alias,
            error=repr(exc),
        )
        return None

    logger.info(
        "model_loaded",
        model=name,
        alias=selected_alias,
        version=info.version,
        run_id=info.run_id,
    )
    return LoadedModel(pipeline=pipeline, info=info)


def load_models(settings: Settings) -> dict[str, LoadedModel]:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    name = settings.mlflow_model_name
    client = MlflowClient()
    try:
        registered = client.get_registered_model(name)
    except MlflowException as exc:
        logger.warning("models_not_loaded", model=name, error=repr(exc))
        return {}

    loaded: dict[str, LoadedModel] = {}
    pipelines: dict[str, Pipeline] = {}
    for alias in sorted(registered.aliases):
        try:
            version = client.get_model_version_by_alias(name, alias)
            uri = f"models:/{name}/{version.version}"
            pipeline = pipelines.get(uri)
            if pipeline is None:
                pipeline = mlflow.sklearn.load_model(uri)
                pipelines[uri] = pipeline
            info = _model_info(client, name, alias, version)
            loaded[alias] = LoadedModel(pipeline=pipeline, info=info)
        except (MlflowException, OSError) as exc:
            logger.warning(
                "model_alias_not_loaded",
                model=name,
                alias=alias,
                error=repr(exc),
            )
    return loaded


def list_models(settings: Settings) -> list[ModelInfo]:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    name = settings.mlflow_model_name
    client = MlflowClient()
    registered = client.get_registered_model(name)
    aliases = {str(version): alias for alias, version in registered.aliases.items()}
    versions = client.search_model_versions(f"name='{name}'")
    versions = sorted(versions, key=lambda item: int(item.version))
    return [
        _model_info(client, name, aliases.get(str(version.version)), version)
        for version in versions
    ]


def predict_labels(model: LoadedModel, texts: list[str]) -> np.ndarray:
    return model.pipeline.predict(pd.DataFrame({"text": texts}))


def predict(model: LoadedModel, texts: list[str]) -> list[Prediction]:
    labels = predict_labels(model, texts)
    return [
        Prediction(index=index, category=str(label))
        for index, label in enumerate(labels)
    ]
