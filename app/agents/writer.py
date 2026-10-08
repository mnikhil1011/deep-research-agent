"""Writer: turns the collected findings into a cited Markdown report."""

import re

from app import config, llm
from app.graph.state import ResearchState

SYSTEM = """You are the writing agent in a research pipeline. You are given the \
user's query, a report title, and findings gathered by a researcher. Each finding \
ends with the ids of the sources that support it, like [3] or [1][4].

Write a Markdown report that answers the query:
- Start with the title as a level-1 heading, then a short summary of the answer.
- Organise the body into sections with level-2 headings. Follow the logic of \
the answer, not the order the findings arrive in.
- Use only the findings provided. After every claim, cite its source ids in \
the same [n] form. Do not cite ids that were not given to you.
- Where findings conflict, say so and present both sides.
- If the findings do not answer part of the query, say what is missing.
- Do not add a list of sources at the end. It is appended automatically."""


def _format_findings(state: ResearchState) -> str:
    sections = []
    for question in state["plan"].sub_questions:
        lines = [
            f"- {f.claim} {''.join(f'[{i}]' for i in f.source_ids)}"
            for f in state.get("findings", [])
            if f.sub_question == question
        ]
        sections.append(f"## {question}\n" + ("\n".join(lines) or "(no findings)"))
    return "\n\n".join(sections)


def _sources_section(report: str, state: ResearchState) -> str:
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", report)}
    lines = [
        f"{s.id}. [{s.title}]({s.url})" for s in state.get("sources", []) if s.id in cited
    ]
    return "## Sources\n\n" + "\n".join(lines) if lines else ""


def write(state: ResearchState) -> dict:
    prompt = (
        f"Query: {state['query']}\n"
        f"Title: {state['plan'].title}\n\n"
        f"Findings, grouped by sub-question:\n\n{_format_findings(state)}"
    )
    report, usage = llm.text(
        agent="writer", model=config.WRITER_MODEL, system=SYSTEM, prompt=prompt
    )
    sources = _sources_section(report, state)
    if sources:
        report = f"{report.rstrip()}\n\n{sources}\n"
    return {"report": report, "usage": [usage]}
