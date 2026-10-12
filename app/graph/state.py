"""Shared graph state and the data models that flow through it."""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class Decision(BaseModel):
    """The orchestrator's choice of what happens next."""

    reasoning: str = Field(description="One or two sentences explaining the choice")
    action: Literal["research", "write"]
    tasks: list[str] = Field(
        description="Research tasks to run in parallel. Empty when action is 'write'."
    )


class Source(BaseModel):
    url: str
    title: str
    content: str


class Finding(BaseModel):
    """One factual claim and the pages that support it."""

    claim: str
    source_urls: list[str]
    task: str


class TaskReport(BaseModel):
    """How one researcher's run went."""

    task: str
    steps: int
    findings: int
    note: str = ""


class ToolCall(BaseModel):
    """One tool request a researcher made, and what came back. Kept for auditing."""

    task: str
    step: int
    name: str
    input: dict
    output: str
    is_error: bool = False


class Usage(BaseModel):
    """Token counts for one model call, as the API reported them."""

    agent: str
    model: str
    input_tokens: int  # uncached input only
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


def merge_sources(old: dict[str, Source], new: dict[str, Source]) -> dict[str, Source]:
    """Union by URL. If two researchers saw the same page, keep the fuller text."""
    merged = dict(old)
    for url, source in new.items():
        if url not in merged or len(source.content) > len(merged[url].content):
            merged[url] = source
    return merged


class ResearchState(TypedDict, total=False):
    query: str
    decision: Decision
    round: int  # research rounds dispatched so far
    # These use reducers, so parallel researchers can all contribute.
    sources: Annotated[dict[str, Source], merge_sources]
    findings: Annotated[list[Finding], operator.add]
    task_reports: Annotated[list[TaskReport], operator.add]
    tool_calls: Annotated[list[ToolCall], operator.add]
    usage: Annotated[list[Usage], operator.add]
    report: str


class ResearcherInput(TypedDict):
    """What the orchestrator hands to each researcher it dispatches."""

    query: str
    task: str
