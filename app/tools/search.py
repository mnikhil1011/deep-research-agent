"""Web search via Tavily."""

from functools import lru_cache

from tavily import TavilyClient


@lru_cache
def _client() -> TavilyClient:
    return TavilyClient()  # reads TAVILY_API_KEY


def web_search(query: str, max_results: int) -> list[dict]:
    """Return results as dicts with `url`, `title` and `content` keys."""
    response = _client().search(query, max_results=max_results)
    return [
        {"url": r["url"], "title": r.get("title") or r["url"], "content": r.get("content") or ""}
        for r in response.get("results", [])
    ]
