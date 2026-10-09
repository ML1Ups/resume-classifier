# Пояснения к защите домашнего задания 3

Работа демонстрирует полный цикл MLOps на классификации текста резюме. Данные настоящие: используется присланный cv_target.csv с 28 935 резюме. Цель — профессиональная сфера из professionList.codeProfessionalSphere. При запуске создаётся Experiment, пять Runs и три зарегистрированные версии модели. Версия для API выбирается alias champion.

## Что показать

1. MLflow UI → Experiment resume-classifier-hw3 → Run eda. Покажи Dataset inputs, распределение классов, длины текстов, частотные токены и выводы. Анализ выполнен на train.
2. Один модельный Run: входы training/validation, параметры TF-IDF и модели, validation_f1_macro, дополнительные метрики, confusion matrix и classification report.
3. Выдели четыре модельных Runs. Сравни validation_f1_macro, validation_accuracy, fit_seconds. Dummy — нижняя граница; остальные Runs меняют модель и представление текста.
4. Models → resume-classifier: версии, исходные Runs, alias champion.
5. Swagger → POST /process: ответ, model.version, model.run_id, model.model_uri.
6. Переключи alias через UI или CLI. До перезапуска используется старая версия. После `docker compose restart app` — новая, без изменения inference-кода.

## Сущности MLflow

**Dataset** описывает входные данные и связывается с Run через `mlflow.log_input`. Context обозначает назначение: source, eda, training, validation или testing.

**Experiment** объединяет запуски одной задачи. **Run** — отдельный запуск с параметрами, метриками, входами и артефактами. EDA и каждый вариант классификатора имеют свои Runs.

**Параметры** заданы перед обучением: seed, тип модели, C, analyzer, ngram range, размеры split. **Метрики** — измеренные результаты: macro F1, accuracy, balanced accuracy, weighted F1, время fit.

**Артефакты** — файлы: графики, отчёты, lineage и сохранённые модели. Файлы находятся в Artifact Store; метаданные и ссылки — в Backend Store.

**Logged Model** сохраняет sklearn Pipeline и signature входа/выхода. **Model Registry** объединяет версии под одним именем и связывает их с Runs. **Alias** — изменяемый указатель на версию. `models:/resume-classifier@champion` обозначает alias, `models:/resume-classifier/2` — конкретную версию.

## Source digest и lineage

**Source** — происхождение данных, например `file:///input/cv_target.csv`. Для внешних данных `--source` задаёт устойчивый исходный URI. Это ссылка происхождения, а не автоматическая копия всего CSV в MLflow.

**Digest** вычисляет MLflow Dataset для различения содержимого. Он не гарантирует криптографический хеш всех байтов файла. Дополнительно вычисляется SHA-256 полного исходного CSV, сохраняемый в теге dataset.sha256 и lineage.

**Lineage** — цепочка CSV → обработка → train/validation/test → Run → версия → API. В data/lineage.json записаны parent digest, SHA-256, source, seed, доли split, digest частей и ID входящих строк. Тег eda.run_id связывает обучение с анализом. Версия Registry ссылается на Run; ответ API сообщает фактическую версию и Run ID.

Признак модели — только text, собранный из должности, навыков, опыта и образования. category извлекается из professionList, record_id и group_id служат учёту происхождения; в TF-IDF они не попадают. Один соискатель и одинаковый текст не пересекаются между split. Отчёт подготовки объясняет удаление 1 неразмеченной строки, 1 558 повторов и 10 строк двух редких сфер. Рабочий набор содержит 27 366 текстов.

## Метрики и выбор модели

Macro F1 усредняет F1 классов с одинаковым весом, чтобы качество редких направлений не скрывалось за многочисленным классом. Accuracy показывает общую долю правильных ответов, balanced accuracy — средний recall классов, weighted F1 учитывает число примеров. Матрица ошибок показывает, какие направления путаются.

На настоящем наборе 34 класса с сильным дисбалансом. Фактические результаты находятся в results/comparison.csv; они получены на grouped holdout, а не на синтетических примерах. Dummy показывает качество самого частого класса. CSV с ошибками и перепутанными парами помогает объяснить ограничения модели.

Выбор выполняется на validation. При равенстве предпочитается более простой word LogisticRegression. Test оценивается только после выбора. Дальнейший ручной подбор по тому же test превратил бы его в validation; для реальных исследований нужен независимый финальный holdout.

## Архитектура

| Компонент | Назначение | Состояние |
|---|---|---|
| mlflow | HTTP API, UI, Registry, прокси артефактов | PostgreSQL и отдельный том файлов |
| mlflow-db | Backend Store | mlflow-pgdata |
| artifact-store | Artifact Store | том, смонтированный в /mlartifacts |
| train | EDA, обучение, регистрация, champion | запись результатов в MLflow |
| app | FastAPI и inference | модель в памяти |
| db | PostgreSQL исходного приложения | pgdata |

Artifact Store не смешан с таблицами PostgreSQL и не находится в слое контейнера. Сервер использует `--serve-artifacts --artifacts-destination /mlartifacts`; новые Experiments получают proxy URI. Клиент общается с MLflow по HTTP и не монтирует хранилище файлов.

## Однократная загрузка и асинхронность

Lifespan создаёт DB pool и вызывает ModelService.load в рабочем потоке. Сначала alias разрешается в номер версии, затем загружается неизменяемый URI этой версии. Если alias переместили между чтением метаданных и загрузкой, модель и сообщаемая версия остаются согласованными.

Модель хранится в app.state.model_service. POST /process не обращается к MLflow и выполняет predict существующего Pipeline. Препроцессинг и классификатор — один артефакт, поэтому преобразования обучения и inference совпадают.

`async def` не делает sklearn асинхронным. run_in_threadpool переносит синхронный predict из event loop; Semaphore ограничивает одновременные операции. Каждая реплика загружает свой снимок на startup.

При ошибке загрузки startup завершается с ошибкой, DB pool закрывается. Запасная модель не используется. /healthz — liveness; успешный старт гарантирует наличие модели. /api/v1/health сохраняет проверку PostgreSQL.

## Горячее обновление

Перемещение alias не заменяет модель в памяти работающих процессов. После частичного перезапуска реплики могут использовать разные версии. Загрузка alias на каждом запросе увеличивает задержку, связывает доступность inference с MLflow и может нарушать согласованность версий. Замена общего объекта в памяти во время обработки тоже требует синхронизации.

В этой работе alias разрешается на startup, обновление требует явного перезапуска. Это не горячая замена без остановки.

Для production лучше зафиксировать номер версии, поднять новые реплики параллельно старым, проверить signature, зависимости, загрузку и тестовые запросы, затем постепенно переключить трафик. Canary или blue-green deployment позволяют наблюдать ошибки и откатить трафик. Старые реплики завершают текущие запросы. Alias служит указателем для следующего развёртывания, а сам deployment получает неизменяемую версию.

## Доказательства проверки

results/comparison.csv и training_summary.json содержат фактические метрики. integration_check.json фиксирует реальные ответы API до смены alias и после нового startup. VERIFICATION.md перечисляет выполненные проверки и ограничения. Снимок относится к временному серверу; собственный запуск создаст актуальные Runs в MLflow.

Документация: [Tracking Server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/), [Model Registry](https://mlflow.org/docs/latest/ml/model-registry/), [Dataset Tracking](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.data.html).
