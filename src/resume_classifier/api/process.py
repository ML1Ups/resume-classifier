from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from resume_classifier.model import LoadedModel, predict
from resume_classifier.schemas import ProcessRequest, ProcessResponse

router = APIRouter(tags=["inference"])

MODEL_NOT_LOADED = {status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model is not loaded"}}


def loaded_model(request: Request) -> LoadedModel:
    model: LoadedModel | None = request.app.state.model
    if model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded")
    return model


@router.post("/process", responses=MODEL_NOT_LOADED)
async def process(
    payload: ProcessRequest,
    model: Annotated[LoadedModel, Depends(loaded_model)],
) -> ProcessResponse:
    predictions = await run_in_threadpool(predict, model, payload.texts)
    return ProcessResponse(predictions=predictions, model=model.info)
