"""Runs the whole graph offline, with the LLM and web search stubbed out."""

import pytest

from app import llm
from app.agents import researcher
from app.agents.researcher import ExtractedFindings
from app.graph.builder import build_graph
from app.graph.state import Finding, ResearchPlan, Usage


def _usage(agent: str) -> Usage:
    return Usage(agent=agent, model="stub", input_tokens=10, output_tokens=5)


@pytest.fixture
def stubs(monkeypatch):
    prompts = {}

    def structured(*, agent, schema, prompt, **_):
        if schema is ResearchPlan:
            return ResearchPlan(title="Stub Report", sub_questions=["Q1", "Q2"]), _usage(agent)
        # 99 is an id the researcher was never shown.
        findings = [
            Finding(claim=f"grounded claim for {prompt[:12]}", source_ids=[1, 99]),
            Finding(claim="ungrounded claim", source_ids=[99]),
        ]
        return ExtractedFindings(findings=findings), _usage(agent)

    def text(*, agent, prompt, **_):
        prompts["writer"] = prompt
        return "# Stub Report\n\nA claim [1]. Another [2].", _usage(agent)

    def web_search(query, max_results):
        # shared.example is returned for both sub-questions.
        return [
            {"url": "https://shared.example", "title": "Shared", "content": "..."},
            {"url": f"https://{query.lower()}.example", "title": query, "content": "..."},
        ]

    monkeypatch.setattr(llm, "structured", structured)
    monkeypatch.setattr(llm, "text", text)
    monkeypatch.setattr(researcher, "web_search", web_search)
    return prompts


def test_graph_runs_end_to_end(stubs):
    result = build_graph().invoke({"query": "test query"})

    assert result["plan"].sub_questions == ["Q1", "Q2"]
    assert [u.agent for u in result["usage"]] == ["planner", "researcher", "researcher", "writer"]
    assert result["report"].startswith("# Stub Report")


def test_sources_are_deduplicated_with_stable_ids(stubs):
    result = build_graph().invoke({"query": "test query"})

    assert [(s.id, s.url) for s in result["sources"]] == [
        (1, "https://shared.example"),
        (2, "https://q1.example"),
        (3, "https://q2.example"),
    ]


def test_findings_citing_unknown_sources_are_dropped(stubs):
    result = build_graph().invoke({"query": "test query"})

    assert [f.source_ids for f in result["findings"]] == [[1], [1]]
    assert [f.sub_question for f in result["findings"]] == ["Q1", "Q2"]
    assert "ungrounded" not in stubs["writer"]


def test_report_lists_only_cited_sources(stubs):
    report = build_graph().invoke({"query": "test query"})["report"]

    sources = report.split("## Sources")[1]
    assert "1. [Shared](https://shared.example)" in sources
    assert "2. [Q1](https://q1.example)" in sources
    assert "q2.example" not in sources
