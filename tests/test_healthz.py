from httpx import AsyncClient


async def test_healthz_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "hello world"}


async def test_healthz_does_not_depend_on_database(unreachable_client: AsyncClient) -> None:
    response = await unreachable_client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "hello world"}
