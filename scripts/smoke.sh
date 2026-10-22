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
export RABBITMQ_USER=smoke
export RABBITMQ_PASSWORD=smoke
export RABBITMQ_PORT=55672
export RABBITMQ_UI_PORT=25672
export QUEUE_MAX_LENGTH=2
export INFERENCE_SECONDS=0.1

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

submit() {
  curl -sS -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:${APP_PORT}/api/v1/tasks" \
    -H 'Content-Type: application/json' -d '{"text":"Python FastAPI PostgreSQL API"}'
}

processed_version() {
  local task_id status
  task_id=$(curl -fsS -X POST "http://127.0.0.1:${APP_PORT}/api/v1/tasks" \
    -H 'Content-Type: application/json' -d '{"text":"Python FastAPI PostgreSQL API"}' \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])') || fail "task was not accepted"
  for _ in $(seq 1 60); do
    status=$(curl -fsS "http://127.0.0.1:${APP_PORT}/api/v1/tasks/${task_id}")
    if [[ $status == *'"status":"DONE"'* ]]; then
      python3 -c 'import json, sys; print(json.loads(sys.argv[1])["model_version"])' "$status"
      return
    fi
    sleep 0.5
  done
  fail "task ${task_id} was not processed: $status"
}

old_version=$(processed_version)
echo "OK   task processed by worker with model version ${old_version}"

new_version=$((old_version % 3 + 1))
compose exec -T worker python -m resume_classifier.ml.registry --version "$new_version" \
  >/dev/null || fail "champion alias was not moved"
[[ $(processed_version) == "$old_version" ]] || fail "worker changed model without restart"
echo "OK   worker keeps model version ${old_version} until restart"
compose restart worker >/dev/null
compose up -d --no-deps --wait --wait-timeout 120 worker || fail "worker did not restart"
[[ $(processed_version) == "$new_version" ]] || fail "worker did not load version ${new_version}"
echo "OK   worker loads model version ${new_version} after restart"

compose stop worker >/dev/null
codes=$(for _ in 1 2 3 4; do submit; echo; done | tr '\n' ' ')
[[ $codes == *503* ]] || fail "full queue did not return 503: $codes"
retry_after=$(curl -sS -D - -o /dev/null -X POST "http://127.0.0.1:${APP_PORT}/api/v1/tasks" \
  -H 'Content-Type: application/json' -d '{"text":"Python"}' | tr -d '\r' \
  | awk -F': ' 'tolower($1) == "retry-after" {print $2}')
[[ -n $retry_after ]] || fail "503 without Retry-After"
echo "OK   full queue -> ${codes}with Retry-After: ${retry_after}"
compose start worker >/dev/null

compose stop db >/dev/null
check /healthz 200 '"status":"ok"'
check /api/v1/health 503 '"name":"postgres","status":"error"'

echo "Smoke test passed"
