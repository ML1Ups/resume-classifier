#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "File .env not found. Create it first: cp .env.example .env" >&2
  exit 1
fi

docker compose up -d --wait db
uv run pytest "$@"
