from dataclasses import dataclass

import mlflow.sklearn
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


def load_model(settings: Settings) -> LoadedModel | None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    name = settings.mlflow_model_name
    try:
        version = MlflowClient().get_model_version_by_alias(name, settings.mlflow_model_alias)
        uri = f"models:/{name}/{version.version}"
        pipeline = mlflow.sklearn.load_model(uri)
    except (MlflowException, OSError) as exc:
        logger.warning("model_not_loaded", model=name, error=repr(exc))
        return None
    info = ModelInfo(
        name=name,
        alias=settings.mlflow_model_alias,
        version=str(version.version),
        run_id=version.run_id,
        model_uri=uri,
    )
    logger.info("model_loaded", model=name, version=info.version, run_id=info.run_id)
    return LoadedModel(pipeline=pipeline, info=info)


def predict(model: LoadedModel, texts: list[str]) -> list[Prediction]:
    labels = model.pipeline.predict(pd.DataFrame({"text": texts}))
    return [Prediction(index=index, category=str(label)) for index, label in enumerate(labels)]
