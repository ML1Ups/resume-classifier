import numpy as np
from fastapi import APIRouter, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from resume_classifier.ml.artifacts import quality_metrics
from resume_classifier.ml.dataset import load_evaluation_test
from resume_classifier.model import LoadedModel, predict, predict_labels
from resume_classifier.schemas import DatasetInfo, ProcessRequest, ProcessResponse, Prediction

router = APIRouter(tags=["inference"])

MODEL_NOT_LOADED = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model is not loaded"}
}


def loaded_models(request: Request) -> dict[str, LoadedModel]:
    models: dict[str, LoadedModel] = request.app.state.models
    if not models:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models are loaded",
        )
    return models


def selected_model(
    payload: ProcessRequest,
    models: dict[str, LoadedModel],
) -> LoadedModel:
    model = models.get(payload.model)
    if model is None:
        available = sorted(models)
        detail = f"Model alias '{payload.model}' is not loaded"
        if available:
            detail += f". Available aliases: {', '.join(available)}"
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=detail)
    return model


def _evaluation_predictions(
    frame,
    predicted: np.ndarray,
    sample_size: int,
) -> list[Prediction]:
    return [
        Prediction(
            index=int(index),
            category=str(predicted[index]),
            expected=str(frame.iloc[index]["category"]),
        )
        for index in range(min(sample_size, len(frame)))
    ]


@router.post("/process", responses=MODEL_NOT_LOADED)
async def process(
    payload: ProcessRequest,
    request: Request,
) -> ProcessResponse:
    model = selected_model(payload, loaded_models(request))
    if payload.texts is not None:
        predictions = await run_in_threadpool(predict, model, payload.texts)
        return ProcessResponse(predictions=predictions, model=model.info)

    dataset_version = payload.dataset
    if dataset_version is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Dataset version is required",
        )

    frame = await run_in_threadpool(
        load_evaluation_test,
        request.app.state.settings.evaluation_data_dir,
        dataset_version,
    )
    predicted = await run_in_threadpool(predict_labels, model, frame["text"].tolist())
    all_metrics = quality_metrics(frame["category"], predicted)
    metrics = {name: all_metrics[name] for name in payload.metrics}
    predictions = _evaluation_predictions(frame, predicted, payload.sample_size)
    return ProcessResponse(
        predictions=predictions,
        model=model.info,
        dataset=DatasetInfo(version=dataset_version, rows=len(frame)),
        metrics=metrics,
    )
