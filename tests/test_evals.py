"""Tests for the evaluation harness. No model or web calls are made.

The judge is replaced by rule-based fakes, so these tests check the arithmetic
and the bookkeeping. They cannot show that the real judge's verdicts are right:
`python -m evals.check_judge` and `python -m evals.review` are for that.
"""

import json
import time
from pathlib import Path

import pytest
from fakes import fake_tool_turn, fake_web_search, usage

from app import config, llm
from app.agents import researcher
from app.graph.state import Decision, Source, Usage
from evals import check_judge, judge, metrics, review, run, summarize
from evals.cases import CASES_MD_PATH, Case, load_cases, render_markdown
from evals.sentences import citation_markers, split_report

# --- sentences ---------------------------------------------------------------


def test_split_report_reads_sentences_and_their_citations():
    report = (
        "# Title\n\n"
        "Costs fell by 40% in 2023 [1][2]. Prices then rose. [3] The trend is unclear.\n\n"
        "## Section\n\n"
        "- A bullet with a source [2].\n"
        "1. A numbered point.\n\n"
        "## Sources\n\n1. [A](https://a.example)\n"
    )

    sentences = split_report(report)

    assert [(s.text, s.citations) for s in sentences] == [
        ("Costs fell by 40% in 2023.", (1, 2)),
        ("Prices then rose.", (3,)),
        ("The trend is unclear.", ()),
        ("A bullet with a source.", (2,)),
        ("A numbered point.", ()),
    ]
    assert [s.index for s in sentences] == [0, 1, 2, 3, 4]


def test_split_report_does_not_break_on_abbreviations_or_decimals():
    report = "Firms, e.g. Toyota, spent $4.5 billion in the U.S. Department of Energy trials [1]. Dr. Lee agreed [2]."

    assert [s.citations for s in split_report(report)] == [(1,), (2,)]


def test_split_report_treats_a_table_row_as_one_unit_and_skips_rules():
    report = "| Firm | Year |\n| --- | --- |\n| Toyota | 2027 [1] |\n\n---\n\nDone."

    assert [(s.text, s.citations) for s in split_report(report)] == [
        ("Firm | Year", ()),
        ("Toyota | 2027", (1,)),
        ("Done.", ()),
    ]


def test_repeated_markers_count_once_per_sentence_but_every_time_overall():
    report = "One fact [1] and another [1]. A third [2].\n\n## Sources\n\n1. [A](u)\n2. [B](u)"

    assert [s.citations for s in split_report(report)] == [(1,), (2,)]
    assert citation_markers(report) == [1, 1, 2]


# --- cases -------------------------------------------------------------------


def test_case_set_is_balanced_and_valid():
    cases = load_cases()

    groups = [c.tags[0] for c in cases]
    assert {g: groups.count(g) for g in set(groups)} == {
        "technology": 6, "science": 6, "business": 6, "policy": 6, "trap": 6,
    }  # fmt: skip
    assert all(len(c.aspects) >= 3 for c in cases if not c.is_trap)


def test_cases_markdown_is_in_step_with_cases_json():
    assert CASES_MD_PATH.read_text(encoding="utf-8") == render_markdown(load_cases()), (
        "Run `python -m evals.cases` to regenerate evals/CASES.md"
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"id": "a b", "tags": ["x"], "question": "q", "aspects": ["a"]},
        {"id": "ok", "tags": ["x"], "question": "q"},
        {"id": "ok", "tags": ["x"], "question": "q", "aspects": ["a"], "false_premise": "p"},
        {"id": "ok", "tags": ["x"], "question": "q", "false_premise": "p"},
        {"id": "ok", "tags": ["trap"], "question": "q", "aspects": ["a"]},
    ],
)
def test_malformed_cases_are_rejected(fields):
    with pytest.raises(ValueError):
        Case.model_validate(fields)


# --- fake judge --------------------------------------------------------------


def _judge_usage() -> Usage:
    return Usage(agent="judge", model="judge-stub", input_tokens=100, output_tokens=20)


@pytest.fixture
def fake_judge(monkeypatch):
    """Rule-based judge: verdicts follow marker words in the text being graded."""
    calls = {"support": [], "claims": 0}

    def support(sources, sentences, ledger):
        ledger.append(_judge_usage())
        calls["support"].append((tuple(sources), [s.text for s in sentences]))
        return {
            s.index: judge.SupportVerdict(
                index=s.index,
                reason="rule",
                support="full" if "FULL" in s.text else "partial" if "PARTIAL" in s.text else "none",
            )
            for s in sentences
        }

    def claims(sentences, ledger):
        ledger.append(_judge_usage())
        calls["claims"] += 1
        return {
            s.index: judge.ClaimVerdict(
                index=s.index,
                reason="rule",
                kind="claim" if any(ch.isdigit() for ch in s.text) else "not_claim",
            )
            for s in sentences
        }

    def completeness(question, aspects, report, ledger):
        ledger.append(_judge_usage())
        return [
            judge.AspectVerdict(
                index=i, reason="rule", status="answered" if aspect in report else "missing"
            )
            for i, aspect in enumerate(aspects)
        ]

    def trap(question, false_premise, report, ledger):
        ledger.append(_judge_usage())
        return judge.TrapVerdict(reason="rule", handled="no such thing" in report)

    monkeypatch.setattr(judge, "judge_support", support)
    monkeypatch.setattr(judge, "judge_claims", claims)
    monkeypatch.setattr(judge, "judge_completeness", completeness)
    monkeypatch.setattr(judge, "judge_trap", trap)
    monkeypatch.setattr(judge, "JUDGE_MODEL", "judge-stub")
    return calls


SOURCES = {
    1: Source(url="https://a.example", title="A", content="text a"),
    2: Source(url="https://b.example", title="B", content="text b"),
}
ANSWERABLE = Case(id="c1", tags=["technology"], question="Q?", aspects=["alpha", "beta", "gamma"])
TRAP = Case(id="t1", tags=["trap"], question="Trap?", false_premise="It is false.")

# --- metrics -----------------------------------------------------------------


def test_metrics_are_the_documented_ratios(fake_judge):
    report = (
        "# Report\n\n"
        "An opening line with no facts.\n\n"
        "FULL one [1]. FULL four [1]. PARTIAL two [2]. Wrong three [1][2]. "
        "An uncited fact from 2020. FULL but dangling [9].\n\n"
        "It covers alpha and beta.\n\n"
        "## Sources\n\n1. [A](https://a.example)\n"
    )

    grade = metrics.grade_report(ANSWERABLE, report, SOURCES, [])

    # Cited: 5 sentences (2 full, 1 partial, 2 unsupported). Uncited claims: 1.
    assert grade.detail["counts"] == {
        "sentences": 8, "cited": 5, "full": 2, "partial": 1, "unsupported": 2,
        "uncited_claims": 1, "claims": 6, "citation_markers": 6, "valid_citation_markers": 5,
    }  # fmt: skip
    assert grade.scores == pytest.approx(
        {
            "groundedness": 2 / 6,
            "citation_coverage": 5 / 6,
            "citation_accuracy": 2 / 5,
            "unsupported_rate": 2 / 5,
            "valid_citations": 5 / 6,
            "completeness": 2 / 3,
        }
    )
    s = grade.scores
    assert s["groundedness"] == pytest.approx(s["citation_accuracy"] * s["citation_coverage"])


def test_sentences_citing_the_same_sources_share_one_judge_call(fake_judge):
    report = "FULL one [1]. FULL four [1]. PARTIAL two [2]. Wrong three [2][1]. FULL but dangling [9]."

    metrics.grade_report(ANSWERABLE, report, SOURCES, [])

    assert sorted(fake_judge["support"]) == [
        ((1,), ["FULL one.", "FULL four."]),
        ((1, 2), ["Wrong three."]),
        ((2,), ["PARTIAL two."]),
    ]  # the sentence citing only [9] needs no judge: it has no source to check


def test_a_sentence_citing_only_missing_sources_is_unsupported(fake_judge):
    grade = metrics.grade_report(ANSWERABLE, "FULL but dangling [9].", SOURCES, [])

    assert grade.scores["citation_accuracy"] == 0.0
    assert grade.scores["valid_citations"] == 0.0
    assert grade.detail["sentences"][0]["support"] == "none"


def test_a_report_with_no_claims_has_no_groundedness_score(fake_judge):
    grade = metrics.grade_report(TRAP, "There is no such thing.", SOURCES, [])

    assert grade.scores == {"trap_handled": 1.0}


def test_an_empty_report_scores_zero_not_a_pass(fake_judge):
    assert metrics.grade_report(ANSWERABLE, "", SOURCES, []).scores == {"completeness": 0.0}
    assert metrics.grade_report(TRAP, "", SOURCES, []).scores == {"trap_handled": 0.0}


def test_a_perfect_report_scores_one_everywhere(fake_judge):
    report = "FULL alpha [1]. FULL beta [2]. FULL gamma [1][2]."

    assert metrics.grade_report(ANSWERABLE, report, SOURCES, []).scores == {
        "groundedness": 1.0,
        "citation_coverage": 1.0,
        "citation_accuracy": 1.0,
        "unsupported_rate": 0.0,
        "valid_citations": 1.0,
        "completeness": 1.0,
    }


def test_judge_spend_is_kept_when_grading_fails_part_way(fake_judge, monkeypatch):
    def broken(*args):
        raise judge.GraderError("judge stopped with refusal")

    monkeypatch.setattr(judge, "judge_completeness", broken)
    ledger: list[Usage] = []

    with pytest.raises(judge.GraderError):
        metrics.grade_report(ANSWERABLE, "FULL one [1].", SOURCES, ledger)

    assert len(ledger) == 1  # the support call that completed before the failure


# --- judge plumbing ----------------------------------------------------------


def test_judge_rejects_a_reply_that_skips_or_invents_sentences(monkeypatch):
    def ask(system, prompt, schema, ledger):
        return judge.SupportVerdicts(
            verdicts=[judge.SupportVerdict(index=7, reason="r", support="full")]
        )

    monkeypatch.setattr(judge, "_ask", ask)
    sentences = split_report("One [1]. Two [1].")

    with pytest.raises(judge.GraderError, match="expected \\[0, 1\\]"):
        judge.judge_support({1: SOURCES[1]}, sentences, [])


def test_judge_prompts_carry_the_material_but_not_the_citation_numbers_inline(monkeypatch):
    prompts = []

    def ask(system, prompt, schema, ledger):
        prompts.append(prompt)
        return judge.SupportVerdicts(
            verdicts=[judge.SupportVerdict(index=0, reason="r", support="full")]
        )

    monkeypatch.setattr(judge, "_ask", ask)
    judge.judge_support({2: SOURCES[2]}, split_report("Costs fell [2]."), [])

    assert '<source id="2" title="B" url="https://b.example">\ntext b\n</source>' in prompts[0]
    assert '<sentence index="0">Costs fell.</sentence>' in prompts[0]


@pytest.mark.parametrize(
    ("served", "expected"),
    [
        ("claude-haiku-4-5", True),
        ("claude-haiku-4-5-20251001", True),
        ("claude-haiku-4-5@20251001", True),
        ("claude-haiku-4-5-2025-10-01", True),
        ("claude-haiku-4-5-turbo", False),
        ("claude-opus-4-8", False),
    ],
)
def test_served_as_accepts_only_the_requested_model_or_its_snapshot(served, expected):
    assert judge.served_as("claude-haiku-4-5", served) is expected


def test_check_judge_reports_a_pass_only_when_every_probe_matches(monkeypatch, capsys):
    expected_support = {text: verdict for _, text, verdict in check_judge.SUPPORT_PROBES}
    expected_claims = {text: kind for _, text, kind in check_judge.CLAIM_PROBES}
    expected_aspects = {report: statuses for _, report, statuses in check_judge.COMPLETENESS_PROBES}
    expected_trap = {report: handled for _, report, handled in check_judge.TRAP_PROBES}

    monkeypatch.setattr(judge, "judge_support", lambda sources, sentences, ledger: {
        s.index: judge.SupportVerdict(index=s.index, reason="r", support=expected_support[s.text])
        for s in sentences})
    monkeypatch.setattr(judge, "judge_claims", lambda sentences, ledger: {
        s.index: judge.ClaimVerdict(index=s.index, reason="r", kind=expected_claims[s.text])
        for s in sentences})
    monkeypatch.setattr(judge, "judge_completeness", lambda q, aspects, report, ledger: [
        judge.AspectVerdict(index=i, reason="r", status=status)
        for i, status in enumerate(expected_aspects[report])])
    monkeypatch.setattr(judge, "judge_trap", lambda q, premise, report, ledger:
                        judge.TrapVerdict(reason="r", handled=expected_trap[report]))
    assert check_judge.main() == 0

    # A judge that passes an empty report must fail the check.
    monkeypatch.setattr(judge, "judge_trap", lambda q, premise, report, ledger:
                        judge.TrapVerdict(reason="r", handled=True))
    assert check_judge.main() == 1
    assert "FAIL  trap         accepts the false premise" in capsys.readouterr().out


# --- runner ------------------------------------------------------------------

def _unapproved_state() -> dict:
    """The project's metric and price declarations, as they are before any approval."""
    path = Path(run.__file__).parent / "runs" / "_state.json"
    state = json.loads(path.read_text("utf-8"))
    state.pop("harness_sha", None)
    return state


CASES = [
    Case(id="c1", tags=["technology"], question="Question one?", aspects=["FULL claim", "absent"]),
    Case(id="c2", tags=["science"], question="Question two?", aspects=["FULL claim"]),
    Case(id="t1", tags=["trap"], question="Trap question?", false_premise="It is false."),
]


@pytest.fixture
def harness(monkeypatch, tmp_path, fake_judge):
    """The real runner and graph, with the model, the web and the judge faked."""
    env = type("Env", (), {})()
    env.flow = tmp_path / "runs"
    env.flow.mkdir()
    (env.flow / "_state.json").write_text(json.dumps(_unapproved_state()), "utf-8")
    env.writer_calls = []
    env.writer_error = {}  # question -> exception to raise

    def structured(*, agent, prompt, **_):
        first = "No research has been done yet" in prompt
        return Decision(
            reasoning="because", action="research" if first else "write", tasks=["T1"] if first else []
        ), usage(agent)

    def text(*, agent, prompt, **_):
        question = prompt.split("\n")[0].removeprefix("Query: ")
        env.writer_calls.append(question)
        if question in env.writer_error:
            raise env.writer_error[question]
        return "# Report\n\nA FULL claim [1]. Another FULL one [2]. An uncited fact from 2020.", usage(agent)

    monkeypatch.setattr(llm, "structured", structured)
    monkeypatch.setattr(llm, "text", text)
    monkeypatch.setattr(llm, "tool_turn", fake_tool_turn)
    monkeypatch.setattr(researcher, "web_search", fake_web_search)
    monkeypatch.setattr(researcher, "read_page", lambda url: f"full text of {url}")
    for name in ("ORCHESTRATOR_MODEL", "RESEARCHER_MODEL", "WRITER_MODEL"):
        monkeypatch.setattr(config, name, "stub")
    monkeypatch.setattr(config, "MAX_RESEARCH_ROUNDS", config.MAX_RESEARCH_ROUNDS)  # restored after
    monkeypatch.setattr(run, "load_cases", lambda: CASES)

    def go(*extra, approve=True):
        args = ["--flow", str(env.flow), *extra]
        if "--variant" not in extra:
            args += ["--variant", "baseline"]
        return run.main(args + (["--approve-harness"] if approve else []))

    env.run = go
    env.rows = lambda v="baseline": [
        json.loads(line) for line in (env.flow / v / "results.jsonl").read_text("utf-8").splitlines()
    ] if (env.flow / v / "results.jsonl").exists() else []
    env.errors = lambda v="baseline": [
        json.loads(line) for line in (env.flow / v / "errors.jsonl").read_text("utf-8").splitlines()
    ] if (env.flow / v / "errors.jsonl").exists() else []
    return env


def test_run_scores_every_case_and_saves_the_evidence(harness):
    assert harness.run() == 0

    rows = {row["prompt_id"]: row for row in harness.rows()}
    assert set(rows) == {"c1", "c2", "t1"}
    row = rows["c1"]
    # 2 cited sentences, both fully supported, plus 1 uncited claim.
    assert row["grade"] == pytest.approx(
        {
            "groundedness": 2 / 3,
            "citation_coverage": 2 / 3,
            "citation_accuracy": 1.0,
            "unsupported_rate": 0.0,
            "valid_citations": 1.0,
            "completeness": 0.5,
        }
    )
    assert rows["t1"]["grade"]["trap_handled"] == 0.0
    assert "completeness" not in rows["t1"]["grade"]
    assert (row["status"], row["rep"], row["tags"], row["prompt"]) == ("ok", 0, ["technology"], "Question one?")
    # Two orchestrator calls, three researcher steps and one writer call, 10 in / 5 out each.
    assert row["usage"] == {
        "input_tokens": 60, "output_tokens": 30,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
    }  # fmt: skip
    assert row["usage_by_model"] == {"stub": row["usage"]}
    assert (row["model"], row["judge_model"]) == ("stub", "judge-stub")
    assert row["judge_usage"]["input_tokens"] == 400  # 2 support groups, claims, completeness
    assert {k: row[k] for k in ("rounds", "tasks", "tool_calls", "web_searches", "pages_read", "sources_cited")} == {
        "rounds": 1, "tasks": 1, "tool_calls": 2, "web_searches": 1, "pages_read": 1, "sources_cited": 2,
    }  # fmt: skip
    assert row["latency_s"] >= 0 and row["report_words"] > 0

    trace = json.loads((harness.flow / "baseline" / "traces" / "c1_rep0.json").read_text("utf-8"))
    assert {turn["role"] for turn in trace} == {"system", "user", "assistant", "tool_call", "tool_result"}
    assert [t["name"] for t in trace if t["role"] == "tool_call"] == [
        "web_search", "read_page", "submit_findings",
    ]  # fmt: skip
    assert trace[-1]["content"].startswith("# Report")

    verdicts = json.loads((harness.flow / "baseline" / "judge" / "c1_rep0.json").read_text("utf-8"))
    assert [s.get("support") or s.get("kind") for s in verdicts["sentences"]] == ["full", "full", "claim"]
    assert verdicts["sources"]["1"]["url"] == "https://shared.example"
    assert verdicts["sources"]["1"]["content"].startswith("full text")


def test_rerunning_skips_what_is_already_scored(harness):
    harness.run()
    calls = len(harness.writer_calls)

    assert harness.run() == 0

    assert len(harness.writer_calls) == calls
    assert len(harness.rows()) == 3


def test_reps_are_separate_rows_with_their_own_files(harness):
    harness.run("--reps", "2", "--cases", "c1")

    # Attempts run concurrently, so rows land in completion order.
    assert sorted((r["prompt_id"], r["rep"]) for r in harness.rows()) == [("c1", 0), ("c1", 1)]
    assert (harness.flow / "baseline" / "traces" / "c1_rep1.json").exists()


def test_a_refusal_is_recorded_as_an_error_and_retried_next_time(harness):
    harness.writer_error["Question two?"] = llm.LLMError("writer: model refused", "refusal")

    assert harness.run() == 1

    assert {r["prompt_id"] for r in harness.rows()} == {"c1", "t1"}
    (error,) = harness.errors()
    assert (error["prompt_id"], error["failure_class"], error["stage"]) == ("c2", "refusal", "app")
    # The orchestrator and researcher calls made before the refusal were still billed.
    assert error["usage_by_model"]["stub"]["input_tokens"] == 50

    harness.writer_error.clear()
    assert harness.run() == 0
    assert {r["prompt_id"] for r in harness.rows()} == {"c1", "c2", "t1"}


def test_a_judge_failure_is_not_scored_as_a_model_failure(harness, monkeypatch):
    def broken(*args):
        raise judge.GraderError("judge stopped with max_tokens")

    monkeypatch.setattr(judge, "judge_trap", broken)

    assert harness.run() == 1

    assert {r["prompt_id"] for r in harness.rows()} == {"c1", "c2"}
    (error,) = harness.errors()
    assert (error["prompt_id"], error["failure_class"], error["stage"]) == ("t1", "grader_error", "grading")
    assert error["judge_usage"]["input_tokens"] > 0


def test_an_attempt_over_the_time_ceiling_is_a_timeout_not_a_score(harness, monkeypatch):
    slow = llm.text

    def text(**kwargs):
        time.sleep(0.5)
        return slow(**kwargs)

    monkeypatch.setattr(llm, "text", text)

    assert harness.run("--cases", "c1", "--timeout-s", "0.1") == 1

    assert harness.rows() == []
    assert [e["failure_class"] for e in harness.errors()] == ["timeout"]


def test_a_response_from_the_wrong_model_fails_the_attempt(harness, monkeypatch):
    monkeypatch.setattr(config, "WRITER_MODEL", "claude-sonnet-5-5")  # the fakes answer as "stub"

    assert harness.run("--cases", "c1") == 1

    (error,) = harness.errors()
    assert error["failure_class"] == "serving_substitution"
    assert "writer was served by stub" in error["error"]


def test_the_harness_must_be_approved_and_reapproved_after_a_change(harness, monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="has not been approved yet"):
        harness.run(approve=False)
    assert harness.rows() == []

    harness.run("--cases", "c1")
    assert harness.run("--cases", "c1", approve=False) == 0  # unchanged, so no approval needed

    grader = tmp_path / "grader.py"
    grader.write_text("v1")
    monkeypatch.setattr(run, "HARNESS_FILES", [*run.HARNESS_FILES, grader])
    with pytest.raises(SystemExit, match="changed since it was approved"):
        harness.run("--cases", "c1", approve=False)


def test_a_variant_holds_one_configuration(harness):
    assert harness.run("--variant", "v1", "--set", "MAX_RESEARCH_ROUNDS=1", "--note", "One round") == 0

    saved = json.loads((harness.flow / "v1" / "config.json").read_text("utf-8"))
    assert saved["overrides"] == {"MAX_RESEARCH_ROUNDS": "1"}
    assert saved["settings"]["max_research_rounds"] == 1
    assert (harness.flow / "v1" / "change.md").read_text("utf-8").splitlines()[0] == "One round"

    with pytest.raises(SystemExit, match="different settings"):
        harness.run("--variant", "v1", "--set", "MAX_RESEARCH_ROUNDS=2")
    with pytest.raises(SystemExit):
        harness.run("--set", "MAX_RESEARCH_ROUNDS=1")  # the baseline takes no overrides
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        harness.run("--variant", "v2", "--set", "NOT_A_SETTING=1")


# --- summary -----------------------------------------------------------------


def _row(case, group, grade, *, model="claude-sonnet-5-5", latency=10.0, rep=0):
    return {
        "prompt_id": case, "rep": rep, "tags": [group], "status": "ok", "grade": grade,
        "usage_by_model": {model: {"input_tokens": 1_000_000, "output_tokens": 100_000,
                                   "cache_read_input_tokens": 1_000_000,
                                   "cache_creation_input_tokens": 1_000_000}},
        "judge_model": "claude-opus-5-5",
        "judge_usage": {"input_tokens": 500_000, "output_tokens": 50_000},
        "latency_s": latency, "rounds": 2, "tool_calls": 6, "sources_cited": 5, "report_words": 800,
    }  # fmt: skip


def _write_flow(flow, variant_rows, errors=()):
    flow.mkdir()
    (flow / "_state.json").write_text(json.dumps(_unapproved_state()), "utf-8")
    for variant, rows in variant_rows.items():
        (flow / variant).mkdir()
        (flow / variant / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    (flow / "baseline" / "errors.jsonl").write_text("".join(json.dumps(e) + "\n" for e in errors), "utf-8")


def test_summary_numbers_are_recomputable_from_the_rows(tmp_path):
    flow = tmp_path / "runs"
    _write_flow(
        flow,
        {
            "baseline": [
                _row("a", "technology", {"groundedness": 0.5, "completeness": 1.0}),
                _row("a", "technology", {"groundedness": 0.7, "completeness": 1.0}, rep=1),
                _row("b", "science", {"groundedness": 0.9, "completeness": 0.5}, latency=30.0),
                _row("t", "trap", {"trap_handled": 1.0}, latency=20.0),
            ],
            "v1": [
                _row("a", "technology", {"groundedness": 0.8, "completeness": 1.0}),
                _row("b", "science", {"groundedness": 1.0, "completeness": 0.5}),
            ],
        },
        errors=[{"prompt_id": "x", "rep": 0, "failure_class": "timeout", "usage_by_model": {}, "judge_usage": {}}],
    )
    (flow / "v1" / "change.md").write_text("One round only\n", "utf-8")

    text = summarize.summarize(flow)

    # Case a averages its two reps to 0.6, so the mean over cases a and b is 0.75.
    assert "| baseline | 3 | 1 | 0.75 [0.60, 0.90] (n=2) | 0.75 [0.50, 1.00] (n=2) | 1.00 (n=1) |" in text
    # Paired change in groundedness: a +0.2, b +0.1.
    assert "| Groundedness | +0.15 [+0.10, +0.20] | 2 | clear change |" in text
    assert "| Completeness | +0.00 [+0.00, +0.00] | 2 | within noise |" in text
    assert "Change: One round only" in text
    # Sonnet at $2 in / $10 out: 1M input + 0.1M cached read + 1.25M cache write + 0.1M output.
    assert summarize.app_cost(_row("a", "g", {}), {"claude-sonnet-5-5": {"in": 2.0, "out": 10.0}}) == pytest.approx(
        2.0 + 0.2 + 2.5 + 1.0
    )
    # Median latency over cases: a 10, b 30, t 20. App cost $5.70, judge cost $3.00 per attempt.
    assert "| baseline | 20 s | $5.700 | $3.000 | 2.0 | 6.0 | 5.0 | 800.0 | $34.80 |" in text
    assert "| baseline | timeout | 1 |" in text
    assert "| baseline | trap | 1 | n/a | n/a | 1.00 (n=1) |" in text


def test_an_unpriced_model_is_reported_as_not_measured_never_as_zero(tmp_path):
    flow = tmp_path / "runs"
    _write_flow(flow, {"baseline": [_row("a", "technology", {"groundedness": 1.0}, model="mystery-model")]})

    text = summarize.summarize(flow)

    assert "| baseline | 10 s | not measured | $3.000 |" in text
    assert text.rstrip().splitlines()[-1] != "" and "| not measured |" in text


def test_bootstrap_interval_is_reproducible_and_brackets_the_mean():
    values = [0.2, 0.4, 0.6, 0.8, 1.0]

    low, high = summarize.bootstrap_ci(values)

    assert (low, high) == summarize.bootstrap_ci(values)
    assert low < 0.6 < high
    assert summarize.bootstrap_ci([0.5]) is None


def test_summary_runs_on_real_runner_output(harness):
    harness.run()

    text = summarize.summarize(harness.flow)

    assert "| baseline | 3 | 0 | 0.67 [0.67, 0.67] (n=3) |" in text
    assert "not measured" in text  # the stub model has no price


# --- human review of the judge ----------------------------------------------


def test_review_sheet_hides_the_judges_verdict_and_scores_agreement(harness):
    harness.run()
    variant = harness.flow / "baseline"

    sheet = review.write_sheet(variant, n=4)

    text = sheet.read_text("utf-8")
    key = json.loads((variant / "review_key.json").read_text("utf-8"))
    assert text.count("## Item ") == 4 and set(key.values()) == {"full"}
    assert "Judge" not in text.split("## Item 1")[1]
    assert "full text of https://shared.example" in text

    with pytest.raises(SystemExit, match="No verdicts filled in"):
        review.score_sheet(variant)

    blank, answer = "\nYour verdict:\n", "\nYour verdict: {}\n"
    filled = text.replace(blank, answer.format("full"), 3).replace(blank, answer.format("none"))
    sheet.write_text(filled, "utf-8")
    result = review.score_sheet(variant)
    assert result.startswith("Agreement: 3 of 4 labelled items (75%).")
    assert "you said none, the judge said full" in result
