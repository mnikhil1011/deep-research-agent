"""Scholarly lookups: arXiv, Crossref and OpenAlex, plus full-text retrieval.

None of the three needs a key, but each has its own limit, enforced here:

- arXiv asks for at most one request every three seconds.
- Crossref is free and unmetered, but allowed only one request a second when
  this was written. It states its limit in each response, and the pacing here
  follows whatever it states.
- OpenAlex is metered. Without a key it allowed 1,000 credits a day when this
  was written (October 2026); a search costs 10 credits and a filtered lookup 1.
  The credits used and remaining are tracked in `openalex_credits`.

Every function returns `Work` records in one shape, whichever service answered.
"""

import hashlib
import io
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import requests
from pydantic import BaseModel
from pypdf import PdfReader

from app import config
from app.text import normalize

TIMEOUT_S = 40
MAX_DOWNLOAD_BYTES = 30_000_000
ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?")
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_OPENALEX_FIELDS = (
    "id,doi,title,publication_year,cited_by_count,is_retracted,authorships,"
    "open_access,best_oa_location,abstract_inverted_index"
)

# Seconds between requests to each service.
_MIN_INTERVAL = {"arxiv": 3.1, "crossref": 1.05, "openalex": 0.11, "download": 1.0}
_next_slot = dict.fromkeys(_MIN_INTERVAL, 0.0)
_slot_lock = threading.Lock()

openalex_credits: dict[str, int | None] = {"used": 0, "remaining": None}


class ScholarError(RuntimeError):
    """A scholarly service could not be reached or refused the request."""


class Work(BaseModel):
    """One scholarly work, as a service described it."""

    title: str
    authors: list[str] = []
    year: int | None = None
    doi: str = ""  # bare and lower-case, without https://doi.org/
    arxiv_id: str = ""  # without the version suffix
    openalex_id: str = ""  # such as W2626778328
    abstract: str = ""
    pdf_url: str = ""  # an open-access copy, when one is known
    cited_by_count: int | None = None
    is_retracted: bool = False
    source: str = ""  # the service that returned this record

    @property
    def ref(self) -> str:
        """A short handle that identifies the work across calls."""
        if self.openalex_id:
            return f"openalex:{self.openalex_id}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi}"
        return f"title:{normalize(self.title)}"


def same_work(a: Work, b: Work) -> bool:
    """True if two records describe the same work: a shared id, or the same title."""
    for field in ("doi", "arxiv_id", "openalex_id"):
        if getattr(a, field) and getattr(a, field) == getattr(b, field):
            return True
    return bool(a.title) and normalize(a.title) == normalize(b.title)


# --- HTTP --------------------------------------------------------------------


@lru_cache
def _session() -> requests.Session:
    session = requests.Session()
    contact = f" (mailto:{config.SCHOLAR_MAILTO})" if config.SCHOLAR_MAILTO else ""
    session.headers["User-Agent"] = f"paper-audit/0.1{contact}"
    return session


def _wait_for_slot(service: str) -> None:
    with _slot_lock:
        now = time.monotonic()
        slot = max(now, _next_slot[service])
        _next_slot[service] = slot + _MIN_INTERVAL[service]
    time.sleep(slot - now)


def _get(service: str, url: str, params: dict | None = None) -> requests.Response:
    """GET with per-service pacing and retries. Returns any final response, 404 included."""
    last_error = "no attempt made"
    for attempt in range(4):
        _wait_for_slot(service)
        try:
            response = _session().get(url, params=params, timeout=TIMEOUT_S)
        except requests.RequestException as error:
            last_error = f"{type(error).__name__}: {error}"
        else:
            if service == "openalex":
                _track_openalex(response)
            if service == "crossref":
                _follow_crossref_limit(response)
            if response.status_code not in (429, 500, 502, 503, 504):
                return response
            last_error = f"HTTP {response.status_code}"
            if service == "openalex" and openalex_credits["remaining"] == 0:
                raise ScholarError(
                    "OpenAlex's daily allowance is used up. It resets each day. Set "
                    "OPENALEX_FALLBACK=false to check references without it."
                )
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                time.sleep(min(int(retry_after), 30))
        time.sleep(2**attempt + random.random())
    raise ScholarError(f"{service} request failed after retries ({last_error}): {url}")


def _follow_crossref_limit(response: requests.Response) -> None:
    """Pace Crossref requests at the rate its response headers allow."""
    limit = response.headers.get("x-rate-limit-limit", "")
    interval = response.headers.get("x-rate-limit-interval", "").removesuffix("s")
    if limit.isdigit() and interval.isdigit() and int(limit) > 0:
        with _slot_lock:
            _MIN_INTERVAL["crossref"] = 1.05 * int(interval) / int(limit)


def _track_openalex(response: requests.Response) -> None:
    used = response.headers.get("X-RateLimit-Credits-Used")
    remaining = response.headers.get("X-RateLimit-Remaining")
    with _slot_lock:
        if used and used.isdigit():
            openalex_credits["used"] = (openalex_credits["used"] or 0) + int(used)
        if remaining and remaining.isdigit():
            openalex_credits["remaining"] = int(remaining)


# --- arXiv -------------------------------------------------------------------


def _arxiv_work(entry: ET.Element) -> Work | None:
    match = re.search(r"arxiv\.org/abs/(.+?)(?:v\d+)?$", entry.findtext("a:id", "", _ATOM))
    if not match:
        return None  # arXiv reports errors as entries without a paper id
    arxiv_id = match.group(1)
    published = entry.findtext("a:published", "", _ATOM)
    return Work(
        title=" ".join(entry.findtext("a:title", "", _ATOM).split()),
        authors=[a.findtext("a:name", "", _ATOM) for a in entry.findall("a:author", _ATOM)],
        year=int(published[:4]) if published[:4].isdigit() else None,
        doi=(entry.findtext("arxiv:doi", "", _ATOM) or "").lower(),
        arxiv_id=arxiv_id,
        abstract=" ".join(entry.findtext("a:summary", "", _ATOM).split()),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        source="arxiv",
    )


def _arxiv_query(params: dict) -> list[Work]:
    response = _get("arxiv", "https://export.arxiv.org/api/query", params)
    if response.status_code != 200:
        raise ScholarError(f"arXiv returned HTTP {response.status_code}")
    try:
        feed = ET.fromstring(response.text)
    except ET.ParseError as error:
        raise ScholarError(f"arXiv returned something that is not a feed: {error}") from error
    works = [_arxiv_work(entry) for entry in feed.findall("a:entry", _ATOM)]
    return [work for work in works if work]


def arxiv_by_ids(arxiv_ids: list[str]) -> dict[str, Work]:
    """Look up papers by arXiv id, 50 per request. Unknown ids are simply absent."""
    found: dict[str, Work] = {}
    ids = list(dict.fromkeys(arxiv_ids))
    for start in range(0, len(ids), 50):
        chunk = ids[start : start + 50]
        works = _arxiv_query({"id_list": ",".join(chunk), "max_results": len(chunk)})
        found.update({work.arxiv_id: work for work in works})
    return found


def arxiv_search(query: str, max_results: int = 5) -> list[Work]:
    """Search arXiv. `query` uses arXiv's syntax, for example ti:"exact title"."""
    return _arxiv_query({"search_query": query, "max_results": max_results, "sortBy": "relevance"})


def arxiv_title_query(title: str) -> str:
    """Build an arXiv query that matches a title as a phrase."""
    return 'ti:"' + re.sub(r'["\\]', " ", title).strip() + '"'


# --- Crossref ----------------------------------------------------------------


def _crossref_work(item: dict) -> Work:
    dates = (item.get("issued") or {}).get("date-parts") or [[None]]
    year = dates[0][0] if dates[0] else None
    title = " ".join((item.get("title") or [""])[0].split())
    retraction_notices = [
        update
        for update in item.get("updated-by") or []
        if update.get("type") in ("retraction", "withdrawal", "removal")
    ]
    return Work(
        title=title,
        authors=[
            " ".join(part for part in (a.get("given"), a.get("family")) if part)
            for a in item.get("author") or []
        ],
        year=year if isinstance(year, int) else None,
        doi=(item.get("DOI") or "").lower(),
        abstract=" ".join(re.sub(r"<[^>]+>", " ", item.get("abstract") or "").split()),
        cited_by_count=item.get("is-referenced-by-count"),
        is_retracted=bool(retraction_notices) or title.upper().startswith("RETRACTED"),
        source="crossref",
    )


def crossref_by_doi(doi: str) -> Work | None:
    """Look up a DOI. Returns None when Crossref has no record of it."""
    response = _get("crossref", f"https://api.crossref.org/works/{quote(doi, safe='')}")
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise ScholarError(f"Crossref returned HTTP {response.status_code} for {doi}")
    return _crossref_work(response.json()["message"])


def crossref_search(reference_text: str, rows: int = 5) -> list[Work]:
    """Find works matching a free-form reference such as "Author. Title. Venue, year"."""
    response = _get(
        "crossref",
        "https://api.crossref.org/works",
        {"query.bibliographic": reference_text[:500], "rows": rows},
    )
    if response.status_code != 200:
        raise ScholarError(f"Crossref returned HTTP {response.status_code}")
    return [_crossref_work(item) for item in response.json()["message"]["items"]]


# --- OpenAlex ----------------------------------------------------------------


def _openalex_params(**params) -> dict:
    if config.OPENALEX_API_KEY:
        params["api_key"] = config.OPENALEX_API_KEY
    if config.SCHOLAR_MAILTO:
        params["mailto"] = config.SCHOLAR_MAILTO
    return params


def _abstract_from_index(index: dict | None) -> str:
    """OpenAlex stores an abstract as {word: [positions]}. Put it back in order."""
    if not index:
        return ""
    positions = [(position, word) for word, places in index.items() for position in places]
    return " ".join(word for _, word in sorted(positions))


def _openalex_work(item: dict) -> Work:
    doi = (item.get("doi") or "").lower().removeprefix("https://doi.org/")
    location = item.get("best_oa_location") or {}
    pdf_url = location.get("pdf_url") or (item.get("open_access") or {}).get("oa_url") or ""
    arxiv = ARXIV_ID.search(doi.removeprefix("10.48550/arxiv.")) if "arxiv" in doi else None
    arxiv = arxiv or (re.search(r"arxiv\.org/(?:abs|pdf)/" + ARXIV_ID.pattern, pdf_url))
    return Work(
        title=" ".join((item.get("title") or "").split()),
        authors=[
            (a.get("author") or {}).get("display_name") or "" for a in item.get("authorships") or []
        ],
        year=item.get("publication_year"),
        doi=doi,
        arxiv_id=arxiv.group(1) if arxiv else "",
        openalex_id=(item.get("id") or "").rsplit("/", 1)[-1],
        abstract=_abstract_from_index(item.get("abstract_inverted_index")),
        pdf_url=pdf_url,
        cited_by_count=item.get("cited_by_count"),
        is_retracted=bool(item.get("is_retracted")),
        source="openalex",
    )


def _openalex_list(**params) -> list[Work]:
    response = _get(
        "openalex",
        "https://api.openalex.org/works",
        _openalex_params(select=_OPENALEX_FIELDS, **params),
    )
    if response.status_code != 200:
        raise ScholarError(f"OpenAlex returned HTTP {response.status_code}")
    return [_openalex_work(item) for item in response.json()["results"]]


def openalex_by_doi(doi: str) -> Work | None:
    """Look up a DOI (1 credit). Returns None when OpenAlex has no record of it."""
    response = _get(
        "openalex",
        f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='/')}",
        _openalex_params(select=_OPENALEX_FIELDS),
    )
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise ScholarError(f"OpenAlex returned HTTP {response.status_code} for {doi}")
    return _openalex_work(response.json())


def openalex_search(query: str, max_results: int = 5) -> list[Work]:
    """Search works by topic (10 credits)."""
    # OpenAlex rejects queries containing ?, quotes and other search operators.
    plain = " ".join(re.sub(r"[^\w\s-]", " ", query).split())
    return _openalex_list(search=plain, **{"per-page": max_results}) if plain else []


def openalex_citing(openalex_id: str, max_results: int = 8) -> list[Work]:
    """The most-cited works that cite the given one (1 credit)."""
    return _openalex_list(
        filter=f"cites:{openalex_id}", sort="cited_by_count:desc", **{"per-page": max_results}
    )


def openalex_references(openalex_id: str, max_results: int = 8) -> list[Work]:
    """The most-cited works that the given one cites (1 credit)."""
    response = _get(
        "openalex",
        f"https://api.openalex.org/works/{openalex_id}",
        _openalex_params(select="referenced_works"),
    )
    if response.status_code != 200:
        raise ScholarError(f"OpenAlex returned HTTP {response.status_code} for {openalex_id}")
    ids = [url.rsplit("/", 1)[-1] for url in response.json().get("referenced_works") or []]
    if not ids:
        return []
    return _openalex_list(
        filter="openalex:" + "|".join(ids[:50]),
        sort="cited_by_count:desc",
        **{"per-page": max_results},
    )


# --- Full text ---------------------------------------------------------------


def download(url: str) -> bytes:
    """Fetch a file, refusing anything over the size limit."""
    response = _get("download", url)
    if response.status_code != 200:
        raise ScholarError(f"Download failed with HTTP {response.status_code}: {url}")
    if len(response.content) > MAX_DOWNLOAD_BYTES:
        raise ScholarError(f"File is larger than {MAX_DOWNLOAD_BYTES // 1_000_000} MB: {url}")
    return response.content


def fetch_pdf_text(url: str) -> str:
    """Download a PDF and return its text. Results are cached on disk by URL."""
    cache = Path(config.CACHE_DIR) / "papers" / (hashlib.sha256(url.encode()).hexdigest() + ".txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    content = download(url)
    if not content.startswith(b"%PDF"):
        raise ScholarError(f"Not a PDF: {url}")
    try:
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:  # noqa: BLE001 - pypdf raises many types on damaged files
        raise ScholarError(f"Could not read the PDF at {url}: {error}") from error
    if not text.strip():
        raise ScholarError(f"The PDF at {url} has no extractable text")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text
