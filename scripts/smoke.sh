#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export POSTGRES_USER=smoke
export POSTGRES_PASSWORD=smoke
export POSTGRES_DB=smoke
export APP_PORT=18000
export POSTGRES_PORT=55432
export MLFLOW_PORT=15000
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

compose up -d --build --wait --wait-timeout 300 || fail "stack did not become healthy"

[[ $(compose exec -T app id -u) != 0 ]] || fail "app container runs as root"
echo "OK   app container runs as non-root user"

check /healthz 200 '"status":"ok"'
check /api/v1/version 200 "\"version\":\"${version}\""
check /api/v1/health 200 '"name":"postgres","status":"ok"'
check /api/v1/model 200 '"alias":"champion"'

curl --fail --silent --show-error "http://127.0.0.1:${MLFLOW_PORT}/health" \
  || fail "MLflow is unreachable"
curl --fail --silent --show-error -X POST "http://127.0.0.1:${APP_PORT}/process" \
  -H 'Content-Type: application/json' \
  --data '{"texts":["Python FastAPI PostgreSQL API"]}' \
  | python3 -c 'import sys,json; r=json.load(sys.stdin); assert len(r["predictions"])==1; assert r["model"]["version"]; assert r["model"]["run_id"]' \
  || fail "POST /process failed"

old_version=$(curl -fsS "http://127.0.0.1:${APP_PORT}/api/v1/model" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["version"])')
alternate=$(compose exec -T app python -c 'import os; from mlflow import MlflowClient; c=MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"]); n=os.environ["MLFLOW_MODEL_NAME"]; old=str(c.get_model_version_by_alias(n,"champion").version); print(next(str(v.version) for v in c.search_model_versions() if v.name==n and str(v.version)!=old))')
compose exec -T app python -m resume_classifier.ml.registry --version "$alternate" >/dev/null
check /api/v1/model 200 "\"version\":\"${old_version}\""
compose restart app
compose up -d --no-deps --wait --wait-timeout 60 app
check /api/v1/model 200 "\"version\":\"${alternate}\""

compose stop db >/dev/null
check /healthz 200 '"status":"ok"'
check /api/v1/health 503 '"name":"postgres","status":"error"'

echo "Smoke test passed"
