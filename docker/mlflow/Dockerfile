FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-default-groups --group web --group ml

RUN groupadd --system --gid 10001 mlflow \
    && useradd --system --uid 10001 --gid mlflow --no-create-home mlflow \
    && mkdir /mlartifacts \
    && chown -R mlflow:mlflow /app /mlartifacts
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
USER mlflow
EXPOSE 5000
