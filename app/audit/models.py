"""Data models for the paper audit, and the state its graph passes around."""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

from app.graph.state import ToolCall, Usage
from app.tools.scholar import Work

# --- The paper being audited -------------------------------------------------


class Reference(BaseModel):
    """One entry in the paper's bibliography, as the paper gives it."""

    key: str
    raw: str = ""  # the whole entry as plain text
    title: str = ""
    authors: list[str] = []
    year: int | None = None
    doi: str = ""
    arxiv_id: str = ""


class CitedSentence(BaseModel):
    """A sentence in the paper that cites at least one reference."""

    id: int
    text: str  # citations appear as [key] markers
    context: str = ""  # the sentence before it, for pronouns and continuations
    section: str = ""
    keys: list[str]


class Paper(BaseModel):
    title: str
    abstract: str
    opening: str  # the start of the body, where contributions are usually stated
    sentences: list[CitedSentence]
    references: dict[str, Reference]
    missing_keys: list[str] = []  # cited in the text but absent from the bibliography
    uncited_keys: list[str] = []  # in the bibliography but never cited


# --- Reference check ---------------------------------------------------------

ReferenceStatus = Literal["verified", "mismatch", "not_found", "lookup_failed"]


class Resolution(BaseModel):
    """What the scholarly record says about one reference."""

    key: str
    status: ReferenceStatus
    work: Work | None = None  # the matching record, when one was found
    issues: list[str] = []  # human-readable problems, such as a retraction
    found_by: str = ""  # which lookup produced the match


# --- Citation audit ----------------------------------------------------------

Support = Literal["supported", "partially_supported", "not_supported", "cannot_verify"]


class CitationVerdict(BaseModel):
    """Whether one cited work backs one sentence that cites it."""

    sentence_id: int
    key: str
    verdict: Support
    reason: str
    evidence: str = ""  # a quotation from the cited work
    evidence_found: bool = False  # set by code: the quotation really is in the text read
    basis: Literal["full_text", "abstract"] = "abstract"


# --- Related work and novelty ------------------------------------------------


class Contribution(BaseModel):
    id: int
    statement: str  # one claimed contribution, in a single sentence


class Candidate(BaseModel):
    """A paper a scout proposes as close prior work for one contribution."""

    contribution_id: int
    work: Work
    why: str
    already_cited: bool = False  # set by code, by comparing with the reference list


Overlap = Literal["high", "partial", "low"]


class Assessment(BaseModel):
    """How much one candidate overlaps with one claimed contribution."""

    contribution_id: int
    work_ref: str
    overlap: Overlap
    reason: str
    approach: str = ""  # each of these is as stated in the abstract, or "not stated"
    data: str = ""
    result: str = ""


# --- Graph state -------------------------------------------------------------


class AuditState(TypedDict, total=False):
    source: str  # a path to a .tex file or folder, or an arXiv id
    mode: str  # "full", "citations" (no related-work search) or "references" (no model calls)
    paper: Paper
    resolutions: list[Resolution]
    audit_plan: list[dict]  # one entry per cited work to audit: key, work, sentences
    contributions: list[Contribution]
    # These are filled by agents that run in parallel, so they use reducers.
    verdicts: Annotated[list[CitationVerdict], operator.add]
    candidates: Annotated[list[Candidate], operator.add]
    assessments: Annotated[list[Assessment], operator.add]
    notes: Annotated[list[str], operator.add]  # things that went wrong but did not stop the run
    tool_calls: Annotated[list[ToolCall], operator.add]
    usage: Annotated[list[Usage], operator.add]
    report: str
