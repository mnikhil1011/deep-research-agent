"""Shared graph state and the data models that flow through it."""

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    title: str = Field(description="Working title for the final report")
    sub_questions: list[str] = Field(
        description="Self-contained questions that together answer the user's query"
    )


class Source(BaseModel):
    id: int
    url: str
    title: str
    content: str


class Finding(BaseModel):
    """One factual claim and the sources that support it."""

    claim: str = Field(description="A single factual statement, in your own words")
    source_ids: list[int] = Field(description="Ids of the sources that support it")
    sub_question: str = ""


class Usage(BaseModel):
    agent: str
    model: str
    input_tokens: int
    output_tokens: int


class ResearchState(TypedDict, total=False):
    query: str
    plan: ResearchPlan
    # Lists use an additive reducer so several nodes can contribute.
    sources: Annotated[list[Source], operator.add]
    findings: Annotated[list[Finding], operator.add]
    usage: Annotated[list[Usage], operator.add]
    report: str
