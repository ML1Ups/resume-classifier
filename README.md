# feature/task-queue — очередь задач и воркер

Часть проекта [Resume Classifier](../../tree/main). Влита в `main` через [PR #27](../../pull/27).
Первая из трёх веток ДЗ 4 (противодавление на FastStream и RabbitMQ).

## Что сделано

- **RabbitMQ 4.3** в `docker-compose.yml`: веб-интерфейс на http://127.0.0.1:15672, данные на томе.
- **Очередь `tasks`** — quorum, лимит 100 сообщений, при заполнении брокер отказывает в приёме
  (`x-overflow: reject-publish`). Очередь «мёртвых» сообщений — `tasks.dlq`.
- **Воркер** (`worker.py`, отдельный сервис `worker`) на FastStream: один раз при старте загружает
  модели из MLflow по alias, берёт задачи из очереди, пишет результат в PostgreSQL.
- **`POST /api/v1/tasks`** вместо `/process`: ставит задачу в очередь и сразу отвечает `202` со
  ссылкой на задачу. Статус и результат — `GET /api/v1/tasks/{id}`.
- **`POST /api/v1/evaluations`** — оценка модели по версии датасета, переехала из `/process`.
- Таблица `tasks` в PostgreSQL, её создаёт воркер при старте.
- 20 новых тестов, из них два интеграционных на настоящем RabbitMQ через Testcontainers.

```
POST /api/v1/tasks ──► очередь tasks (quorum, ≤ 100) ──► worker (prefetch 4, 4 потока) ──► PostgreSQL
        │ полна: 503 + Retry-After            │ 3 неудачи или неверный формат
        ▼                                     ▼
   клиент повторит позже                 очередь tasks.dlq
```

## Настройки

| Переменная | По умолчанию | Что задаёт |
|---|---:|---|
| `QUEUE_MAX_LENGTH` | 100 | ёмкость очереди, 0 — без лимита |
| `WORKER_THREADS` | 4 | размер пула потоков воркера |
| `WORKER_PREFETCH` | 4 | `prefetch_count`, 0 — без лимита |
| `INFERENCE_SECONDS` | 0.5 | длительность inference, для воспроизводимых замеров |
| `DELIVERY_LIMIT` | 3 | сколько раз повторить задачу перед DLQ |

## Противодавление «брокер → воркер»

**`prefetch_count` = 4 = размер пула потоков.** Брокер держит у воркера не больше 4
неподтверждённых сообщений, и для каждого сразу есть свободный поток. Один воркер обрабатывает
4 / 0,5 с = 8 задач в секунду.

- Если prefetch больше пула, лишние сообщения лежат в памяти воркера и ждут поток. Другим воркерам
  они недоступны, а при падении воркера все они получают +1 к счётчику неудачных доставок.
- Если меньше — потоки простаивают, пропускная способность падает.

**Inference не блокирует event loop.** Обработчик асинхронный, а предсказание модели вместе с
имитацией длительности (`time.sleep(INFERENCE_SECONDS)`) выполняется в `ThreadPoolExecutor` через
`run_in_executor`. Доказательство — воркер отвечает на `GET /health` во время обработки; тест
`test_inference_runs_in_thread_pool_without_blocking_event_loop` проверяет, что 4 задачи по 0,2 с
выполняются параллельно, а задержка event loop остаётся меньше 0,1 с.

## Граница системы: 503 и Retry-After

Внешний клиент вне нашего контура: замедлить его нельзя. Если очередь полна, брокер отклоняет
публикацию (`basic.nack`, в коде — `DeliveryError`), и API сразу отвечает `503` с заголовком
`Retry-After`. Запрос не ждёт освобождения места.

**Значение Retry-After** = ⌈`QUEUE_MAX_LENGTH` × `INFERENCE_SECONDS` / `WORKER_THREADS`⌉ =
⌈100 × 0,5 / 4⌉ = 13 с — за это время один воркер разбирает полную очередь. Повтор раньше почти
наверняка снова упрётся в лимит.

**В базе не остаётся PENDING.** Запись задачи и публикация идут в одной транзакции PostgreSQL:
сначала `INSERT`, потом публикация с подтверждением. Если брокер отказал, транзакция
откатывается. Если воркер успел взять сообщение раньше `COMMIT`, его `INSERT … ON CONFLICT`
дождётся транзакции API по уникальному ключу.

**Почему это load shedding, а не backpressure.** Противодавление замедляет того, кто производит
задачи: сигнал идёт назад по цепочке, и производитель ждёт. Клиента API мы замедлить не можем,
поэтому лишнюю нагрузку сбрасываем — отказываем сразу и говорим, когда прийти снова.
Противодавление в системе есть у источника из следующей ветки,
[`feature/source`](../../tree/feature/source).

## Ошибки и надёжность

**`ack_policy=REJECT_ON_ERROR`, ack — после записи результата.** FastStream подтверждает сообщение
после выхода из обработчика, а запись в PostgreSQL — внутри обработчика. Если воркер упадёт до
записи, сообщение вернётся в очередь.

| Ситуация | Что происходит |
|---|---|
| Ошибка обработки (база недоступна, сбой модели) | `RejectMessage(requeue=True)`: сообщение возвращается, через 0,5 / 1 / 1,5 с повтор; после 3 повторов — в `tasks.dlq` с причиной `delivery_limit` |
| Сообщение неверного формата | reject без возврата: сразу в `tasks.dlq` с причиной `rejected`, повторять бессмысленно |
| Воркер убит во время обработки | неподтверждённые сообщения возвращаются в очередь и обрабатываются заново |

Задержку между повторами даёт сама quorum-очередь (`x-delayed-retry-type: failed`, RabbitMQ 4.3).

**Почему не другие политики:**

- `ACK_FIRST` — подтверждение до обработки, при падении задача теряется.
- `ACK` — ошибки тоже подтверждаются, задача теряется.
- `NACK_ON_ERROR` — в RabbitMQ 4.3 `basic.nack` не увеличивает счётчик неудачных доставок, и
  упавшее сообщение возвращалось бы бесконечно. Проверено на живом брокере: 30 доставок подряд со
  счётчиком 0. `basic.reject` счётчик увеличивает.
- `MANUAL` — сообщение, которое не прошло проверку формата, осталось бы неподтверждённым и заняло
  место в prefetch.

**Идемпотентность — на уровне базы.** `id` задачи — первичный ключ, результат пишется через
`INSERT … ON CONFLICT (id) DO UPDATE … WHERE status = 'PENDING'`. Повторная доставка не создаёт
дубль и не перезаписывает готовый результат; гонки нет, потому что уникальность проверяет
PostgreSQL, а не код.

**Graceful shutdown.** По SIGTERM uvicorn останавливает приложение FastStream: оно перестаёт брать
новые сообщения, ждёт завершения начатых (до 15 с) и только потом закрывает пул PostgreSQL. В
compose `stop_grace_period: 30s`.

## Как проверить

```bash
cp .env.example .env
TRAIN_DATA_PATH=/input/resumes_demo.csv docker compose up -d --build
curl -X POST http://localhost:8000/api/v1/tasks -H 'Content-Type: application/json' -d '{"text": "Python, FastAPI, PostgreSQL"}'
curl http://localhost:8000/api/v1/tasks/<id>
```

| На защите просят | Команда |
|---|---|
| второй воркер | `docker compose up -d --scale worker=2 worker` |
| другой prefetch | `WORKER_PREFETCH=8 docker compose up -d worker` |
| другой лимит очереди | аргументы очереди не меняются на лету: `docker compose exec rabbitmq rabbitmqctl delete_queue tasks`, затем `QUEUE_MAX_LENGTH=20 docker compose up -d worker app`; уменьшить без пересоздания — `docker compose exec rabbitmq rabbitmqctl set_policy tasks-limit '^tasks$' '{"max-length":20}' --apply-to queues` |
| убить воркер во время обработки | `docker compose kill -s SIGKILL worker` |
| graceful shutdown | `docker compose stop worker` (SIGTERM) |
| некорректное сообщение | в веб-интерфейсе RabbitMQ: Queues → `tasks` → Publish message, тело `{"id": "oops"}` |
| health-check во время обработки | `docker compose exec worker python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').status)"` |

Тесты: `./scripts/test.sh` (для интеграционных нужен Docker).
