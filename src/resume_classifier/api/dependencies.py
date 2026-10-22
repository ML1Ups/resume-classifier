from fastapi import HTTPException, Request, status

from resume_classifier.model import LoadedModel


def loaded_models(request: Request) -> dict[str, LoadedModel]:
    models: dict[str, LoadedModel] = request.app.state.models
    if not models:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="No models are loaded")
    return models


def selected_model(alias: str, models: dict[str, LoadedModel]) -> LoadedModel:
    if alias not in models:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Model alias '{alias}' is not loaded. "
            f"Available aliases: {', '.join(sorted(models))}",
        )
    return models[alias]
