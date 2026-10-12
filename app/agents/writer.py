"""Writer: turns the collected findings into a cited Markdown report."""

import re

from app import config, llm
from app.graph.state import Finding, ResearchState

SYSTEM = """You are the writing agent in a research pipeline. You are given the \
user's query and the findings gathered by researchers, grouped by research task. \
Each finding ends with the ids of the sources that support it, like [3] or [1][4].

Write a Markdown report that answers the query:
- Start with a level-1 heading that titles the report, then a short summary of \
the answer.
- Organise the body into sections with level-2 headings. Follow the logic of \
the answer, not the order the findings arrive in.
- Use only the findings provided. After every claim, cite its source ids in \
the same [n] form. Do not cite ids that were not given to you.
- Where findings conflict, say so and present both sides.
- If the findings do not answer part of the query, say what is missing.
- Do not add a list of sources at the end. It is appended automatically."""


def number_sources(findings: list[Finding]) -> dict[str, int]:
    """Number each cited URL from 1, in order of first appearance."""
    numbers: dict[str, int] = {}
    for finding in findings:
        for url in finding.source_urls:
            numbers.setdefault(url, len(numbers) + 1)
    return numbers


def _format_findings(state: ResearchState, numbers: dict[str, int]) -> str:
    sections = []
    for report in state.get("task_reports", []):
        lines = [
            f"- {f.claim} {''.join(f'[{numbers[url]}]' for url in f.source_urls)}"
            for f in state.get("findings", [])
            if f.task == report.task
        ]
        sections.append(f"## {report.task}\n" + ("\n".join(lines) or "(no findings)"))
    return "\n\n".join(sections)


def _sources_section(report: str, state: ResearchState, numbers: dict[str, int]) -> str:
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", report)}
    sources = state.get("sources", {})
    lines = [
        f"{number}. [{sources[url].title}]({url})"
        for url, number in numbers.items()
        if number in cited
    ]
    return "## Sources\n\n" + "\n".join(lines) if lines else ""


def write(state: ResearchState) -> dict:
    numbers = number_sources(state.get("findings", []))
    prompt = (
        f"Query: {state['query']}\n\n"
        f"Findings, grouped by research task:\n\n{_format_findings(state, numbers)}"
    )
    report, usage = llm.text(
        agent="writer", model=config.WRITER_MODEL, system=SYSTEM, prompt=prompt
    )
    sources = _sources_section(report, state, numbers)
    if sources:
        report = f"{report.rstrip()}\n\n{sources}\n"
    return {"report": report, "usage": [usage]}
