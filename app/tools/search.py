"""Web access via Tavily: search, and reading a page in full."""

from functools import lru_cache

from tavily import TavilyClient


@lru_cache
def _client() -> TavilyClient:
    return TavilyClient()  # reads TAVILY_API_KEY


def web_search(query: str, max_results: int) -> list[dict]:
    """Return results as dicts with `url`, `title` and `content` (a snippet) keys."""
    response = _client().search(query, max_results=max_results)
    return [
        {"url": r["url"], "title": r.get("title") or r["url"], "content": r.get("content") or ""}
        for r in response.get("results", [])
    ]


def read_page(url: str) -> str:
    """Return the page's main text. Raises if the page cannot be fetched."""
    response = _client().extract(url)
    results = response.get("results", [])
    if not results or not results[0].get("raw_content"):
        raise RuntimeError(f"Could not read {url}")
    return results[0]["raw_content"]
