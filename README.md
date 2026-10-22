# feature/dataset-versions — версии датасета

Часть проекта [Resume Classifier](../../tree/main). Влита в `main` через [PR #23](../../pull/23).
Первая из трёх веток с доработками ДЗ 3 после защиты.

## Что сделано

- **`scripts/build_evaluation_datasets.py`** собирает из `cv_target.csv` две версии датасета с
  общими validation и test: `v1` — весь train (16 332 резюме), `v2` — стратифицированные 10%
  (1 633). Рядом пишется `manifest.json`: источник, SHA-256 исходного файла и каждой части, размеры.
- **Обучение на версии:** `TRAIN_DATASET_VERSION=v1 TRAIN_ALIAS=demo-v1 docker compose up train`.
  Модель регистрируется под указанным alias, у версии в реестре теги `dataset.version` и размеры
  частей.
- Расчёт метрик вынесен в `ml/metrics.py`: он нужен и обучению, и приложению, а модуль с графиками
  тянет matplotlib.
- Тесты: сборка версий из учебного набора, обучение на версии и теги в реестре.

## Зачем

Validation и test у версий одинаковые, поэтому разница моделей показывает влияние объёма
обучающих данных, а не другой тестовой выборки. На настоящих данных `logreg_characters` даёт
macro-F1 0,365 на `v1` и 0,292 на `v2`.

## Как проверить

```bash
uv run python scripts/build_evaluation_datasets.py
TRAIN_DATASET_VERSION=v1 TRAIN_ALIAS=demo-v1 docker compose up train
TRAIN_DATASET_VERSION=v2 TRAIN_ALIAS=demo-v2 docker compose up train
```

В MLflow у моделей появятся alias `demo-v1` и `demo-v2`.
