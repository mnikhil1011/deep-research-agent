"""Checks every reference against the scholarly record.

Plain code, no model. Each reference is looked up by the strongest handle it
offers, cheapest service first:

    1. its arXiv id          (arXiv, one batched request for the whole paper)
    2. its DOI               (Crossref)
    3. its text              (Crossref's reference search)
    4. its title             (arXiv title search)
    5. its title             (OpenAlex search, which is metered)

A reference is "verified" when a record's title matches it. "not_found" means
no service had a match. That is a prompt to look, not proof of fabrication:
books, reports and web pages are often missing from these indexes.
"""

from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

from app import config
from app.audit.models import Paper, Reference, Resolution
from app.text import normalize
from app.tools import scholar
from app.tools.scholar import ScholarError, Work

TITLE_MATCH = 0.90  # similarity of normalised titles needed to call two works the same
YEAR_TOLERANCE = 2  # a preprint and its published version are often a year or two apart
WORKERS = 8
# arXiv and DataCite DOIs are real but are not registered with Crossref.
_NOT_IN_CROSSREF = ("10.48550/", "10.5281/")


def _squash(text: str) -> str:
    """Letters and digits only, so "long-term", "long term" and "LongTerm" all agree."""
    return normalize(text).replace(" ", "")


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _squash(a), _squash(b)).ratio()


def _shares_an_author(reference: Reference, work: Work) -> bool:
    """True if a surname from the record appears in the reference as written."""
    written = f" {normalize(reference.raw)} {normalize(' '.join(reference.authors))} "
    surnames = [normalize(author).split()[-1] for author in work.authors if normalize(author)]
    return any(len(name) >= 3 and f" {name} " in written for name in surnames)


def matches(reference: Reference, work: Work) -> bool:
    """True if a scholarly record is the work this reference describes."""
    title = normalize(work.title)
    if not title:
        return False
    if not reference.title:
        # No title could be parsed out of the entry, so look for the record's
        # title anywhere in it. This is only safe when there is no title to
        # compare: a record with a short title can sit inside a longer, different one.
        return len(title.split()) >= 3 and _squash(title) in _squash(reference.raw)
    if title_similarity(reference.title, work.title) >= TITLE_MATCH:
        return True
    # A shortened title ("A decomposable attention model") still counts when it
    # is the start of the record's title and an author agrees.
    cited = normalize(reference.title)
    return len(cited.split()) >= 3 and title.startswith(cited) and _shares_an_author(reference, work)


def _search_text(reference: Reference) -> str:
    """The text sent to Crossref's reference search, which reads whole citations."""
    if reference.raw:
        return reference.raw
    first_author = reference.authors[0] if reference.authors else ""
    return " ".join(part for part in (first_author, reference.title, str(reference.year or "")) if part)


def _enrich(work: Work) -> Work:
    """Fill in the abstract, open-access link and OpenAlex id that Crossref lacks."""
    if work.abstract and work.pdf_url:
        return work
    if not work.doi:
        return work
    try:
        extra = scholar.openalex_by_doi(work.doi)
    except ScholarError:
        return work
    if not extra:
        return work
    return work.model_copy(
        update={
            "abstract": work.abstract or extra.abstract,
            "pdf_url": work.pdf_url or extra.pdf_url,
            "openalex_id": extra.openalex_id,
            "arxiv_id": work.arxiv_id or extra.arxiv_id,
            "is_retracted": work.is_retracted or extra.is_retracted,
        }
    )


def resolve(
    reference: Reference, arxiv_works: dict[str, Work] | None, *, enrich: bool
) -> Resolution:
    """Check one reference.

    `arxiv_works` holds the batched arXiv lookups, or None if that request failed.
    """
    issues: list[str] = []
    failures: list[str] = []  # searches that could not be completed
    exact_id = False  # found through an identifier, not through a search

    def verified(work: Work, found_by: str) -> Resolution:
        if enrich:
            work = _enrich(work)
        if work.is_retracted:
            issues.append("This work has been retracted.")
        if (
            exact_id
            and reference.year
            and work.year
            and abs(reference.year - work.year) > YEAR_TOLERANCE
        ):
            issues.append(f"Cited as {reference.year}, but the record is dated {work.year}.")
        return Resolution(
            key=reference.key, status="verified", work=work, issues=issues, found_by=found_by
        )

    def mismatch(work: Work, identifier: str) -> Resolution:
        issues.append(
            f'{identifier} is registered to "{work.title}", which does not match the title as cited.'
        )
        return Resolution(
            key=reference.key, status="mismatch", work=work, issues=issues, found_by=identifier
        )

    if reference.arxiv_id and arxiv_works is not None:
        work = arxiv_works.get(reference.arxiv_id)
        if work:
            exact_id = True
            label = f"arXiv:{reference.arxiv_id}"
            return verified(work, label) if matches(reference, work) else mismatch(work, label)
        issues.append(f"arXiv has no paper with the id {reference.arxiv_id}.")

    if reference.doi and not reference.doi.startswith(_NOT_IN_CROSSREF):
        try:
            work = scholar.crossref_by_doi(reference.doi)
        except ScholarError:
            work = None  # fall through to the searches
        else:
            if work:
                exact_id = True
                label = f"DOI {reference.doi}"
                return verified(work, label) if matches(reference, work) else mismatch(work, label)
            issues.append(f"Crossref has no record of the DOI {reference.doi}.")

    searches = [("Crossref search", lambda: scholar.crossref_search(_search_text(reference)))]
    if reference.title:
        searches.append(
            ("arXiv title search", lambda: scholar.arxiv_search(scholar.arxiv_title_query(reference.title)))
        )
        if config.OPENALEX_FALLBACK:
            searches.append(("OpenAlex search", lambda: scholar.openalex_search(reference.title)))
    for found_by, search in searches:
        try:
            candidates = search()
        except ScholarError as error:
            failures.append(str(error))
            continue
        match = next((work for work in candidates if matches(reference, work)), None)
        if match:
            return verified(match, found_by)

    if failures:
        # A search that failed might have found it, so "not found" would be a guess.
        return Resolution(key=reference.key, status="lookup_failed", issues=issues + failures[:1])
    issues.append(
        "No matching record on arXiv, Crossref or OpenAlex. It may be a book, a report or a "
        "web page that these indexes do not cover, or the reference may be wrong."
    )
    return Resolution(key=reference.key, status="not_found", issues=issues)


def resolve_all(paper: Paper) -> list[Resolution]:
    """Check every reference in the paper, in bibliography order."""
    references = list(paper.references.values())
    arxiv_ids = [r.arxiv_id for r in references if r.arxiv_id]
    try:
        arxiv_works = scholar.arxiv_by_ids(arxiv_ids) if arxiv_ids else {}
    except ScholarError:
        arxiv_works = None
    cited = {key for sentence in paper.sentences for key in sentence.keys}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(
            pool.map(lambda r: resolve(r, arxiv_works, enrich=r.key in cited), references)
        )
