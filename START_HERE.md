# Домашнее задание 3 MLflow на настоящем датасете

Проект использует присланный `cv_target.csv`: 28 935 резюме и 48 колонок. Реализован полный путь от подготовки данных до MLflow Model Registry и асинхронного `POST /process`. По умолчанию приложение обучается на этом датасете. Учебный набор остался только для быстрых тестов и CI.

## Запуск

Нужен Docker Desktop с Linux containers или Docker Engine с Compose v2. Python и uv на компьютере для запуска через Docker не требуются.

Распакуй архив, открой терминал в папке проекта. В PowerShell:

```powershell
Copy-Item .env.example .env
docker compose up -d --build --wait --wait-timeout 900
```

В Linux или WSL:

```bash
cp .env.example .env
docker compose up -d --build --wait --wait-timeout 900
```

Первая сборка скачивает зависимости. После неё Compose поднимает PostgreSQL и MLflow, выполняет подготовку, EDA и четыре модельных Runs, регистрирует три версии модели, назначает `champion` и запускает API. Скорость обучения зависит от компьютера; прогресс виден через `docker compose logs -f train`.

- MLflow UI — [http://localhost:5000](http://localhost:5000).
- Swagger — [http://localhost:8000/docs](http://localhost:8000/docs).
- Загруженная версия модели — [http://localhost:8000/api/v1/model](http://localhost:8000/api/v1/model).

```bash
docker compose ps -a
docker compose logs train
docker compose logs app
```

Статус `train` — `Exited (0)` означает успешное завершение. API запускается после обучения; запасная модель при ошибке не подставляется.

CSV уже включён в архив и лежит в `data/cv_target.csv`. Он монтируется в обучающий контейнер как `/input/cv_target.csv`; его путь задан `TRAIN_DATA_PATH`. Данные не включаются в Docker-образы. В Git настоящий CSV игнорируется; CI использует маленькую тестовую фикстуру.

## Что именно классифицируется

Цель — `professionList[*].codeProfessionalSphere`: профессиональная сфера из исходной разметки. Из 36 исходных сфер поддерживаются 34. Исходные коды сохранены, в том числе написание `BuldindRealty`. Это общие направления: например, `InformationTechnology`, `Medicine`, `AccountingTaxesManagement`, `Sales`.

`typicalPosition` почти пустая и не используется как цель. Профессия `codeProfession` и название `positionName` тоже не подменяют целевую метку. `positionName` используется как доступный входной текст наряду с навыками, опытом и образованием.

### Подготовка и split

- 1 строка не имеет целевой сферы и исключается.
- 1 558 повторов нормализованного текста с одинаковой меткой исключаются после построения групп.
- `Logistic` — 9 строк и `Entertainment` — 1 строка: слишком мало для раздельных train/validation/test. Они исключены и перечислены в отчёте подготовки.
- Остаётся 27 366 резюме в 34 классах. Это весь пригодный набор, без семплирования.
- Train: 16 332, validation: 5 579, test: 5 455. Пропорции близки к 60/20/20; размеры могут немного отличаться из-за неделимых групп.

Группа связывает все резюме одного `candidateId` и одинаковые нормализованные тексты, включая транзитивные связи. Группы не пересекаются между частями. Затем используется StratifiedGroupKFold с seed 42. Это защищает от попадания одного соискателя или одинакового текста одновременно в обучение и проверку.

Текст собирается из желаемой должности, hard/soft skills, должностей/обязанностей/достижений на прошлых работах, квалификации/специальности и уровня образования. `professionList`, идентификаторы, `innerInfo`, возраст, пол, названия компаний и вузов не являются признаками. Максимальная длина подготовленного текста — 20 000 символов, как у API.

EDA и TF-IDF vocabulary используют только train. Champion выбирается по macro F1 на validation; при равенстве предпочтение: word LogisticRegression, LinearSVC, character LogisticRegression. Test оценивается только для выбранной модели и не участвует в выборе.

Полный отчёт исключений — `data/preparation.json` в артефактах Run eda и `results/data/preparation.json`. Коды классов и распределение — `data/README.md` и результаты EDA.

## Проверка API

В Swagger выбери `POST /process`, Try it out и отправь:

```json
{
  "texts": [
    "Бухгалтер. Бухгалтерский учет, налоговая отчетность, 1С, расчёт заработной платы.",
    "Медицинская сестра. Уход за пациентами, выполнение назначений врача, перевязки."
  ]
}
```

Или в PowerShell:

```powershell
$body = @{
    texts = @(
        "Бухгалтер. Учет, налоговая отчетность, 1С, расчёт зарплаты.",
        "Медицинская сестра. Уход за пациентами и выполнение назначений врача."
    )
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/process" `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

В Linux:

```bash
curl -X POST http://localhost:8000/process \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Бухгалтерский учет и налоговая отчетность, 1С","Уход за пациентами, медицинская сестра"]}'
```

`predictions` содержит индекс и предсказанный код сферы. `model` сообщает имя, alias, фактически загруженную версию, Run ID и URI. Индексы начинаются с нуля. API принимает 1–32 текста длиной 1–20 000 символов после удаления пробелов по краям. Неправильные типы, пустые строки и лишние поля дают 422.

## Переключение версии

```bash
docker compose exec app python -m resume_classifier.ml.registry
docker compose exec app python -m resume_classifier.ml.registry --version 2
```

В свежем Registry версии 1, 2, 3 соответствуют `logreg_words`, `svm_bigrams`, `logreg_characters`. После следующих обучений номера растут: проверяй вывод команды. Alias также можно изменить через UI.

Работающее приложение продолжает использовать старую модель. Перезапусти только API:

```bash
docker compose restart app
```

После завершения старта `/api/v1/model` и `/process` покажут выбранную версию. Не запускай весь стек повторно для этого шага: повторное выполнение train снова выбирает champion.

## MLflow и требования задания

| Требование | Реализация |
|---|---|
| Tracking Server и UI | mlflow, порт 5000 |
| Backend Store | отдельный PostgreSQL mlflow-db, том mlflow-pgdata |
| Artifact Store | отдельный постоянный том artifact-store в /mlartifacts, HTTP-прокси MLflow |
| EDA | отдельный Run eda: классы, длины текстов, заполненность полей, токены и выводы |
| Dataset Tracking | from_pandas, log_input, source, digest, SHA-256 исходного CSV |
| Lineage | подготовка, исключения, seed, group split, ID строк и групп |
| Runs | Dummy baseline и три обучаемых варианта TF-IDF Pipeline |
| Метрики | macro F1, accuracy, balanced accuracy, weighted F1 и fit_seconds |
| Диагностика | confusion matrix, classification report, CSV ошибок и перепутанных пар |
| Registry | три версии, связь с Run, теги, описания, alias champion |
| Inference | POST /process, модель загружается на startup; predict в рабочем потоке |
| Версия в ответе | GET /api/v1/model и model в ответе POST /process |
| Проверки | pytest, Ruff, настоящая HTTP-интеграция MLflow/API, Docker smoke и CI |

Метаданные и бинарные артефакты физически разделены. Клиенты общаются с Tracking Server по HTTP, а не монтируют Artifact Store. Файлы сохраняются после остановки контейнера.

## Результаты и проверки

`results/comparison.csv` содержит фактическое сравнение на твоём датасете; `training_summary.json` — выбранную версию, метрики test, SHA-256 и сведения о подготовке. В `results/eda`, `validation`, `test`, `data` лежат графики и артефакты проверочного запуска. `integration_check.json` фиксирует реальные ответы API и смену версии после startup. Метрики настоящего набора не заменены показателями учебных данных.

Run IDs и URI снимка относятся к временному проверочному серверу. Собственный запуск Compose создаёт новые Runs и постоянный Registry.

Для разработки нужен uv:

```bash
uv sync --locked
uv run ruff check
uv run ruff format --check
uv run pytest --cov-report=term-missing
uv run python scripts/integration_check.py --data data/cv_target.csv --output results
```

Без `--data` интеграционный скрипт использует маленький demo только для быстрого CI. Обычный training CLI и Compose по умолчанию используют `cv_target.csv`.

Полная Docker-проверка в Linux/WSL:

```bash
bash scripts/smoke.sh
```

Smoke использует отдельный Compose-проект и demo, проверяет PostgreSQL, inference и смену alias, затем удаляет только свои тестовые контейнеры и тома. Фактически выполненные проверки перечислены в `results/VERIFICATION.md`. Docker Engine в среде подготовки отсутствовал, поэтому контейнерная сборка и smoke здесь не запускались.

## Повторное обучение и остановка

```bash
docker compose run --rm --no-deps train
docker compose restart app
```

Каждое обучение создаёт новые Runs и версии. Скопировать результаты стандартного train-контейнера:

```bash
docker compose cp train:/app/results/. results/
```

Другой CSV положи в `data/` и передай `--data /input/имя.csv`. Поддерживаются исходный формат cv_target и готовый UTF-8 CSV с text/category. Для плоского формата действуют проверки непустых значений, повторов и минимум 10 примеров каждого класса.

Остановка с сохранением данных:

```bash
docker compose down
```

`docker compose down -v` удаляет тома проекта с Runs, моделями и базами; это команда полного сброса.

Пояснения для защиты: [docs/DEFENSE.md](docs/DEFENSE.md). Изменения: [docs/CHANGES.md](docs/CHANGES.md).
