# feature/cd — публикация образа в Docker Hub

Часть проекта [Resume Classifier](../../tree/main). Влита в `main` через [PR #3](../../pull/3).

## Что сделано

CD на GitHub Actions: сначала целиком прогоняется CI, и только после его успеха образ
собирается и публикуется в Docker Hub —
[`asebaie/resume-classifier`](https://hub.docker.com/r/asebaie/resume-classifier).

| Событие | Теги образа |
|---|---|
| push ветки, pull request | ничего не публикуется, только CI |
| merge в `main` | `main`, `sha-<commit>` |
| тег `vX.Y.Z` | `X.Y.Z`, `X.Y`, `latest` |

**Почему так.** Релиз — осознанное действие с неизменяемым тегом версии. Образ из `main` даёт
свежую сборку для проверки, а `sha-<commit>` однозначно связывает образ с коммитом. Из веток
ничего не публикуется, чтобы непроверенный код не попадал в реестр.

**Проверка версии.** При публикации по тегу CD сверяет тег с версией в `pyproject.toml` и
падает, если они не совпадают. Поэтому образ `X.Y.Z` всегда отвечает `{"version": "X.Y.Z"}`
на `/api/v1/version`.

## Настройка

В Settings → Secrets and variables → Actions репозитория:

- переменная `DOCKERHUB_USERNAME` — логин на Docker Hub;
- секрет `DOCKERHUB_TOKEN` — access token Docker Hub с правами Read & Write.

Первые запуски CD упали на входе в Docker Hub, потому что их ещё не было. После добавления
запуски перезапустили, и образы `main` и `0.1.0` опубликовались.

## Выпуск версии

Поднять версию в ветке (`uv version --bump patch`), влить pull request и поставить тег на `main`:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

В этой ветке образ собирается только под `linux/amd64`. Сборка под `arm64` добавлена в
[`feature/multiarch`](../../tree/feature/multiarch).
