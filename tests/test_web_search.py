from __future__ import annotations

import httpx

from abmcts_explorer import (
    SearXNGSearchClient,
    SearXNGSearchConfig,
    parse_searxng_engines,
)


def test_parse_searxng_engines_accepts_bracketed_and_plain_values() -> None:
    assert parse_searxng_engines("[google,brave,bing]") == [
        "google",
        "brave",
        "bing",
    ]
    assert parse_searxng_engines("google, brave") == ["google", "brave"]
    assert parse_searxng_engines(None) == []


def test_search_normalizes_and_sorts_results(monkeypatch) -> None:
    def fake_get(*args, **kwargs):
        request = httpx.Request("GET", "http://localhost:8080/search")
        response = httpx.Response(
            200,
            request=request,
            json={
                "results": [
                    {
                        "title": "low",
                        "url": "https://example.com/low",
                        "content": "low score",
                        "score": 0.1,
                    },
                    {
                        "title": "high",
                        "url": "https://example.com/high",
                        "content": "high score",
                        "score": 0.9,
                    },
                ]
            },
        )
        return response

    monkeypatch.setattr(httpx, "get", fake_get)

    client = SearXNGSearchClient(
        SearXNGSearchConfig(base_url="http://localhost:8080")
    )
    response = client.search("ab-mcts", limit=2)

    assert response.success is True
    assert response.error is None
    assert [result.title for result in response.results] == ["high", "low"]
    assert response.results[0].position == 1


def test_search_falls_back_to_next_engine(monkeypatch) -> None:
    calls: list[str | None] = []

    def fake_get(*args, **kwargs):
        engine = kwargs["params"].get("engines")
        calls.append(engine)
        request = httpx.Request("GET", "http://localhost:8080/search")

        if engine == "bad":
            return httpx.Response(500, request=request)

        return httpx.Response(
            200,
            request=request,
            json={
                "results": [
                    {
                        "title": "ok",
                        "url": "https://example.com/ok",
                        "content": "fallback result",
                        "score": 1.0,
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "get", fake_get)

    client = SearXNGSearchClient(
        SearXNGSearchConfig(
            base_url="http://localhost:8080",
            engines=("bad", "good"),
        )
    )
    response = client.search("ab-mcts", limit=1)

    assert response.success is True
    assert calls == ["bad", "good"]
    assert response.results[0].title == "ok"


def test_search_sends_language_parameter(monkeypatch) -> None:
    captured_params = {}

    def fake_get(*args, **kwargs):
        captured_params.update(kwargs["params"])
        request = httpx.Request("GET", "http://localhost:4866/search")
        return httpx.Response(200, request=request, json={"results": []})

    monkeypatch.setattr(httpx, "get", fake_get)

    client = SearXNGSearchClient(
        SearXNGSearchConfig(
            base_url="http://127.0.0.1:4866",
            engines=("google",),
            language="ja",
        )
    )
    response = client.search("東京 天気", limit=1)

    assert response.success is True
    assert captured_params["q"] == "東京 天気"
    assert captured_params["engines"] == "google"
    assert captured_params["language"] == "ja"
