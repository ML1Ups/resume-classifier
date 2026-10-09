# Изменения для домашнего задания 3

Исходный архив содержал FastAPI, служебные эндпоинты, PostgreSQL, логирование, тесты, Docker и CI/CD. Модель, обучение и сами данные отсутствовали. Версия пакета изменена с 0.1.1 на 0.2.0.

| Файл или каталог | Изменение |
|---|---|
| ml/dataset.py | CSV, проверки, стратифицированный split, lineage |
| ml/artifacts.py | EDA, метрики, confusion matrix, отчёты и ошибки |
| ml/train.py | четыре модельных Runs, Dataset Tracking, Registry и champion |
| ml/registry.py | просмотр версий, смена alias через CLI |
| model.py | разрешение alias, однократная загрузка, inference |
| api/process.py | POST /process, GET /api/v1/model |
| schemas.py | запросы, предсказания, метаданные версии |
| main.py | startup, маршруты, освобождение DB pool при ошибке |
| config.py | настройки MLflow и inference concurrency |
| pyproject.toml и uv.lock | ML-зависимости, группа ml, версия пакета |
| Dockerfile | API-образ и обучающий образ через WITH_ML |
| docker/mlflow/Dockerfile | Tracking Server, права для Artifact Store |
| docker-compose.yml | Backend Store, постоянный Artifact Store, MLflow и train-before-app |
| .env.example | настройки MLflow и модели |
| .gitignore и .dockerignore | demo CSV включён; реальные данные и временные хранилища исключены |
| data/ и generate_demo_data.py | учебный набор и генератор |
| tests/ | inference, validation, startup, split, обучение и Registry |
| integration_check.py | реальный HTTP MLflow, inference и смена alias |
| smoke.sh | Docker-проверка сервисов и смены версии |
| test.sh | unit-проверки без обязательного Docker |
| CI | добавлена HTTP-интеграция |
| START_HERE.md и docs/ | запуск, защита, перечень изменений |
| results/ | фактический снимок проверочного запуска |

Пути ml/, model.py, schemas.py и других Python-модулей в таблице относятся к src/resume_classifier/. Скрипты находятся в scripts/. Исторический README и правила веток сохранены; актуальная инструкция — START_HERE.md.

## Обновление после получения настоящего датасета

Добавлены ml/prepare.py и тесты: извлечение цели из professionList, сбор текста из вложенных списков, отчёт исключений и группировка по соискателям/одинаковым текстам. Dataset loader автоматически распознаёт cv_target. Default training и Compose переключены на cv_target.csv; данные монтируются отдельно и исключены из образов. EDA и диагностика адаптированы к 34 неравномерным классам. Синтетические метрики и графики заменены результатами настоящего набора.
