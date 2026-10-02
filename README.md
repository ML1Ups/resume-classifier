# feature/multiarch — образ под amd64 и arm64

Часть проекта [Resume Classifier](../../tree/main). Влита в `main` через [PR #4](../../pull/4),
релиз `v0.1.1`.

## Что сделано

- CD собирает образ сразу под две платформы: `linux/amd64` и `linux/arm64`. Сборщик GitHub
  работает на amd64, вариант для arm64 собирается через эмуляцию QEMU.
- Оба варианта хранятся в Docker Hub под одним тегом, `docker pull` сам выбирает подходящий —
  например, на Mac с Apple Silicon скачивается arm64 без флага `--platform`.
- Версия поднята до `0.1.1`: образ изменился, значит, это новый релиз. Опубликованный `0.1.0`
  не перезаписывается — теги версий неизменяемы.

## Результат

| Теги | Платформы |
|---|---|
| `0.1.1`, `0.1`, `latest`, `main` | amd64, arm64 |
| `0.1.0` | amd64 |

## Как проверить

```bash
docker buildx imagetools inspect asebaie/resume-classifier:0.1.1
docker run --rm -p 8000:8000 -e POSTGRES_HOST=db -e POSTGRES_USER=u -e POSTGRES_PASSWORD=p -e POSTGRES_DB=d asebaie/resume-classifier:0.1.1
```

Первая команда показывает обе платформы. Вторая запускает сервис без БД: `/healthz` отвечает
`{"status": "ok"}`, `/api/v1/version` — `{"version": "0.1.1"}`, а `/api/v1/health` — 503,
потому что базы нет.
