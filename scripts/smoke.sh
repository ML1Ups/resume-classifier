#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export POSTGRES_USER=smoke
export POSTGRES_PASSWORD=smoke
export POSTGRES_DB=smoke
export APP_PORT=18000
export POSTGRES_PORT=55432
export MLFLOW_PORT=15000
export MLFLOW_DB_PASSWORD=smoke
export TRAIN_DATA_PATH=/input/resumes_demo.csv

compose() {
  docker compose --project-name resume-classifier-smoke "$@"
}

cleanup() {
  compose down --volumes --remove-orphans >/dev/null 2>&1 || true
}

fail() {
  echo "FAIL $1" >&2
  compose logs --no-color >&2 || true
  exit 1
}

check() {
  local path=$1 expected_status=$2 expected_body=$3
  local response status body
  response=$(curl -sS -w '\n%{http_code}' "http://127.0.0.1:${APP_PORT}${path}") \
    || fail "$path is unreachable"
  status=$(tail -n1 <<<"$response")
  body=$(sed '$d' <<<"$response")
  [[ $status == "$expected_status" ]] || fail "$path returned $status, expected $expected_status: $body"
  [[ $body == *"$expected_body"* ]] || fail "$path body does not contain $expected_body: $body"
  echo "OK   GET $path -> $status $body"
}

trap cleanup EXIT

version=$(uv version --short)

compose up -d --build --wait --wait-timeout 600 || fail "stack did not become healthy"

[[ $(compose exec -T app id -u) != 0 ]] || fail "app container runs as root"
echo "OK   app container runs as non-root user"

check /healthz 200 '"status":"ok"'
check /api/v1/version 200 "\"version\":\"${version}\""
check /api/v1/health 200 '"name":"postgres","status":"ok"'
check /api/v1/model 200 '"alias":"champion"'

curl -fsS "http://127.0.0.1:${MLFLOW_PORT}/health" >/dev/null || fail "MLflow is unreachable"
echo "OK   MLflow is reachable"

response=$(curl -sS -X POST "http://127.0.0.1:${APP_PORT}/process" \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Python FastAPI PostgreSQL API"]}') || fail "/process is unreachable"
[[ $response == *'"predictions":[{"index":0,"category":"'* ]] \
  || fail "/process returned unexpected body: $response"
echo "OK   POST /process -> $response"

old_version=$(curl -fsS "http://127.0.0.1:${APP_PORT}/api/v1/model" \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["version"])')
new_version=$((old_version % 3 + 1))
compose exec -T app python -m resume_classifier.ml.registry --version "$new_version" >/dev/null \
  || fail "champion alias was not moved"
check /api/v1/model 200 "\"version\":\"${old_version}\""
compose restart app >/dev/null
compose up -d --no-deps --wait --wait-timeout 60 app || fail "app did not restart"
check /api/v1/model 200 "\"version\":\"${new_version}\""

compose stop db >/dev/null
check /healthz 200 '"status":"ok"'
check /api/v1/health 503 '"name":"postgres","status":"error"'

echo "Smoke test passed"
