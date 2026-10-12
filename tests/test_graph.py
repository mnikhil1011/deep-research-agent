"""Runs the whole graph offline, with the LLM and web access replaced by fakes."""

import sys
from types import SimpleNamespace

import pytest
from fakes import fake_tool_turn, fake_web_search, usage

import main
from app import config, llm
from app.agents import researcher
from app.graph.builder import build_graph
from app.graph.state import Decision


@pytest.fixture
def run(monkeypatch):
    """Returns a function that runs the graph with a scripted orchestrator."""
    seen = SimpleNamespace(orchestrator_prompts=[], writer_prompt=None)

    def text(*, agent, prompt, **_):
        seen.writer_prompt = prompt
        return "# Report\n\nA claim [1]. Another [3].", usage(agent)

    monkeypatch.setattr(llm, "text", text)
    monkeypatch.setattr(llm, "tool_turn", fake_tool_turn)
    monkeypatch.setattr(researcher, "web_search", fake_web_search)
    monkeypatch.setattr(researcher, "read_page", lambda url: f"full text of {url}")

    def run_graph(*decisions: tuple[str, list[str]]):
        script = iter(decisions)

        def structured(*, agent, prompt, **_):
            seen.orchestrator_prompts.append(prompt)
            action, tasks = next(script)
            return Decision(reasoning="because", action=action, tasks=tasks), usage(agent)

        monkeypatch.setattr(llm, "structured", structured)
        return build_graph().invoke({"query": "test query"}), seen

    return run_graph


def test_orchestrator_loops_until_it_chooses_to_write(run):
    result, seen = run(("research", ["T1", "T2"]), ("research", ["T3"]), ("write", []))

    assert [r.task for r in result["task_reports"]] == ["T1", "T2", "T3"]
    assert result["round"] == 2
    assert result["report"].startswith("# Report")
    # Each call after the first sees what the researchers found.
    assert "No research has been done yet" in seen.orchestrator_prompts[0]
    assert "claim about T1" in seen.orchestrator_prompts[1]
    assert "claim about T3" in seen.orchestrator_prompts[2]


def test_round_limit_forces_the_write(run, monkeypatch):
    monkeypatch.setattr(config, "MAX_RESEARCH_ROUNDS", 2)

    result, seen = run(("research", ["T1"]), ("research", ["T2"]), ("research", ["T3"]))

    assert [r.task for r in result["task_reports"]] == ["T1", "T2"]
    assert result["decision"].action == "write"
    assert len(seen.orchestrator_prompts) == 2  # the third decision needed no model call
    assert "report" in result


def test_tasks_are_capped_and_repeats_are_dropped(run, monkeypatch):
    monkeypatch.setattr(config, "MAX_TASKS_PER_ROUND", 3)

    result, _ = run(("research", ["T1", "T1", "T2", "T3", "T4"]), ("research", ["T2"]))

    # T4 is over the cap; the second round only repeated T2, so it became a write.
    assert [r.task for r in result["task_reports"]] == ["T1", "T2", "T3"]
    assert result["decision"].action == "write"


def test_first_round_always_researches(run):
    result, _ = run(("write", []), ("write", []))

    assert [r.task for r in result["task_reports"]] == ["test query"]


def test_findings_citing_unknown_sources_are_dropped(run):
    result, seen = run(("research", ["T1"]), ("write", []))

    assert [f.claim for f in result["findings"]] == ["claim about T1"]
    assert result["findings"][0].source_urls == ["https://shared.example", "https://t1.example"]
    assert "ungrounded" not in seen.writer_prompt


def test_sources_are_merged_and_numbered_across_researchers(run):
    result, seen = run(("research", ["T1", "T2"]), ("write", []))

    assert set(result["sources"]) == {
        "https://shared.example",
        "https://t1.example",
        "https://t2.example",
    }
    # The page that was read in full keeps the full text, not the snippet.
    assert result["sources"]["https://shared.example"].content.startswith("full text")
    # The shared page has one number in both tasks' findings.
    assert "claim about T1 [1][2]" in seen.writer_prompt
    assert "claim about T2 [1][3]" in seen.writer_prompt


def test_report_lists_only_cited_sources(run):
    result, _ = run(("research", ["T1", "T2"]), ("write", []))

    sources = result["report"].split("## Sources")[1]
    assert "1. [Shared](https://shared.example)" in sources
    assert "3. [T2](https://t2.example)" in sources
    assert "t1.example" not in sources  # [2] was never cited in the report


def test_researcher_that_never_submits_does_not_sink_the_run(run):
    result, seen = run(("research", ["never submits", "T1"]), ("write", []))

    stuck = result["task_reports"][0]
    assert stuck.steps == config.MAX_RESEARCHER_STEPS
    assert stuck.findings == 0
    assert "ran out of steps" in stuck.note
    assert [f.claim for f in result["findings"]] == ["claim about T1"]
    assert "ran out of steps" in seen.orchestrator_prompts[1]


def test_tool_errors_go_back_to_the_model(run):
    result, _ = run(("research", ["search fails"]), ("write", []))

    # The fake model only searches again if it was shown the error.
    assert result["task_reports"][0].steps == 3
    assert [f.claim for f in result["findings"]] == ["claim after recovering"]
    assert result["findings"][0].source_urls == ["https://recovered.example"]


def test_researcher_is_told_to_finish_before_its_last_step(run, monkeypatch):
    conversation = []

    def tool_turn(*, messages, **kwargs):
        conversation[:] = [messages]
        return fake_tool_turn(messages=messages, **kwargs)

    monkeypatch.setattr(llm, "tool_turn", tool_turn)
    researcher.research({"query": "q", "task": "never submits"})

    # messages alternate user/assistant, so the reply to step k is at index 2k.
    replies = {k: conversation[0][2 * k]["content"] for k in range(1, config.MAX_RESEARCHER_STEPS)}
    last_research_step = config.MAX_RESEARCHER_STEPS - 1
    assert replies[last_research_step][-1] == {"type": "text", "text": researcher.FINISH_NOW}
    assert all(
        block["type"] == "tool_result"
        for k, reply in replies.items()
        if k != last_research_step
        for block in reply
    )


def test_cli_prints_progress_and_saves_the_report(run, monkeypatch, tmp_path, capsys):
    # Reuse the fixture's fakes; the CLI builds and runs its own graph.
    script = iter([("research", ["T1", "T2"]), ("write", [])])

    def structured(*, agent, **_):
        action, tasks = next(script)
        return Decision(reasoning="because", action=action, tasks=tasks), usage(agent)

    monkeypatch.setattr(llm, "structured", structured)
    monkeypatch.setattr(sys, "argv", ["main.py", "What", "is", "X?"])
    monkeypatch.chdir(tmp_path)

    main.main()

    out = capsys.readouterr().out
    assert "[orchestrator] research: because" in out
    assert "[researcher] 1 findings in 3 steps: T1" in out
    assert "[researcher] 1 findings in 3 steps: T2" in out
    assert "[writer] report drafted" in out
    assert (tmp_path / "reports" / "what-is-x.md").read_text(encoding="utf-8").startswith("# Report")
