from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from resume_classifier.api.dependencies import loaded_models, selected_model
from resume_classifier.ml.dataset import load_evaluation_test
from resume_classifier.ml.metrics import quality_metrics
from resume_classifier.model import LoadedModel, predict_labels
from resume_classifier.schemas import (
    DatasetInfo,
    EvaluationRequest,
    EvaluationResponse,
    Prediction,
)

router = APIRouter(tags=["inference"])


@router.post(
    "/evaluations",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Model alias or dataset version not found"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "No models are loaded"},
    },
)
async def evaluate(
    payload: EvaluationRequest,
    request: Request,
    models: Annotated[dict[str, LoadedModel], Depends(loaded_models)],
) -> EvaluationResponse:
    model = selected_model(payload.model, models)
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
    return EvaluationResponse(
        model=model.info,
        dataset=DatasetInfo(version=payload.dataset, rows=len(frame)),
        metrics={name: metrics[name] for name in payload.metrics},
        predictions=[
            Prediction(index=index, category=str(predicted[index]), expected=expected)
            for index, expected in enumerate(frame["category"].head(payload.sample_size))
        ],
    )
