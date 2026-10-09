# feature/mlflow-server — MLflow Tracking Server

Часть проекта [Resume Classifier](../../tree/main). Влита в `main` через [PR #18](../../pull/18).
Вторая из четырёх веток с MLflow.

## Что сделано

- в `docker-compose.yml` добавлены два сервиса: `mlflow` — Tracking Server и `mlflow-db` — его
  PostgreSQL;
- у обоих healthcheck, лимиты CPU и памяти, ротация логов, общая сеть `backend`;
- в `.env.example` добавлены `MLFLOW_PORT` и `MLFLOW_DB_PASSWORD`;
- smoke-тест проверяет, что MLflow отвечает на `/health`.

## Где что хранится

| Хранилище | Что в нём | Где лежит |
|---|---|---|
| Backend Store | эксперименты, запуски, параметры, метрики, реестр моделей | PostgreSQL `mlflow-db`, том `mlflow-pgdata` |
| Artifact Store | графики, отчёты, файлы моделей | том `mlflow-artifacts`, каталог `/mlartifacts` |

Хранилища разделены: метаданные лежат в базе, файлы — на отдельном томе. Сервер запущен с
`--serve-artifacts`, поэтому клиенты отправляют и получают артефакты через него по HTTP и доступ
к тому им не нужен.

## Решения

**Официальный образ.** Используется `ghcr.io/mlflow/mlflow:v3.10.0-full`: в нём уже есть драйвер
PostgreSQL, свой Dockerfile для сервера не нужен. Версия закреплена тегом, а не `latest`.

**Отдельная база.** У MLflow свой PostgreSQL, а не база приложения: их можно останавливать и
очищать независимо друг от друга.

**Порт 5050.** Снаружи MLflow открыт на `127.0.0.1:5050`. Порт 5000 на macOS занят AirPlay:
`localhost:5000` отвечает 403 от системной службы, а не от MLflow. Порт меняется переменной
`MLFLOW_PORT`.

**`--allowed-hosts`.** MLflow 3 проверяет заголовок `Host` и по умолчанию принимает только
`localhost` и частные IP-адреса. Остальные сервисы обращаются к серверу по имени `mlflow`,
поэтому оно добавлено в список разрешённых.

**База MLflow не видна снаружи.** У `mlflow-db` нет открытых портов: к ней обращается только
сервер MLflow по внутренней сети.

**Пароль базы.** `MLFLOW_DB_PASSWORD` обязателен: без него `docker compose` не запустится и
сообщит, какой переменной не хватает. Значения по умолчанию нет.

## Как проверить

```bash
cp .env.example .env
docker compose up -d --build
docker compose ps
./scripts/smoke.sh
```

Интерфейс MLflow — http://127.0.0.1:5050. Пока он пустой: эксперименты появляются в следующей
ветке, [`feature/ml-training`](../../tree/feature/ml-training).
