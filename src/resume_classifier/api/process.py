from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from resume_classifier.ml.dataset import load_evaluation_test
from resume_classifier.ml.metrics import quality_metrics
from resume_classifier.model import LoadedModel, predict, predict_labels
from resume_classifier.schemas import DatasetInfo, Prediction, ProcessRequest, ProcessResponse

router = APIRouter(tags=["inference"])


def loaded_models(request: Request) -> dict[str, LoadedModel]:
    models: dict[str, LoadedModel] = request.app.state.models
    if not models:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="No models are loaded")
    return models


@router.post(
    "/process",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Model alias or dataset version not found"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "No models are loaded"},
    },
)
async def process(
    payload: ProcessRequest,
    request: Request,
    models: Annotated[dict[str, LoadedModel], Depends(loaded_models)],
) -> ProcessResponse:
    if payload.model not in models:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Model alias '{payload.model}' is not loaded. "
            f"Available aliases: {', '.join(sorted(models))}",
        )
    model = models[payload.model]
    if payload.texts:
        predictions = await run_in_threadpool(predict, model, payload.texts)
        return ProcessResponse(predictions=predictions, model=model.info)

    try:
        frame = await run_in_threadpool(
            load_evaluation_test, request.app.state.settings.evaluation_data_dir, payload.dataset
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"Dataset version '{payload.dataset}' is not found"
        ) from exc
    predicted = await run_in_threadpool(predict_labels, model, frame["text"].tolist())
    metrics = quality_metrics(frame["category"], predicted)
    return ProcessResponse(
        predictions=[
            Prediction(index=index, category=str(predicted[index]), expected=expected)
            for index, expected in enumerate(frame["category"].head(payload.sample_size))
        ],
        model=model.info,
        dataset=DatasetInfo(version=payload.dataset, rows=len(frame)),
        metrics={name: metrics[name] for name in payload.metrics},
    )
