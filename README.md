# feature/evaluation — оценка модели на версии датасета

Часть проекта [Resume Classifier](../../tree/main). Влита в `main` через [PR #26](../../pull/26),
релиз `v0.3.0`. Третья из трёх веток с доработками ДЗ 3 после защиты.

## Что сделано

- `/process` умеет считать метрики выбранной модели на test выбранной версии датасета:

```json
{"model": "demo-v1", "dataset": "v1", "metrics": ["accuracy", "f1_macro"], "sample_size": 5}
```

- В ответе: метрики (не больше двух из `accuracy`, `balanced_accuracy`, `f1_macro`,
  `f1_weighted`), размер выборки и первые `sample_size` предсказаний рядом с правильным ответом.
- Версия датасета проверяется по шаблону `v<число>`: она становится частью пути к файлу.
  Неизвестная версия — `404`.
- Папка `data/evaluation` подключается к приложению только на чтение.

## Как проверить

```bash
curl -X POST http://localhost:8000/process -H 'Content-Type: application/json' -d '{"model": "demo-v1", "dataset": "v1", "metrics": ["accuracy", "f1_macro"]}'
```

В ДЗ 4 эта оценка переехала в `POST /api/v1/evaluations`.
