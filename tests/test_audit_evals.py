"""Tests for the two paper-audit benchmarks. No model or network calls are made."""

import json
import random

import pytest
from fakes import tool_calls, usage

from app import config, llm
from app.audit.models import CitedSentence, Paper, Reference
from app.tools import scholar
from app.tools.scholar import Work
from evals.audit import citations, refcheck

REAL = [
    Reference(key="a", raw="A. Author. Deep residual learning for image recognition. 2016.",
              title="Deep residual learning for image recognition", authors=["A. Author"], year=2016, arxiv_id="1512.03385"),
    Reference(key="b", raw="B. Writer. Attention is all you need today. 2017.",
              title="Attention is all you need today", authors=["B. Writer"], year=2017, arxiv_id="1706.03762"),
    Reference(key="c", raw="C. Scribe. Batch normalization accelerates deep network training. 2015.",
              title="Batch normalization accelerates deep network training", authors=["C. Scribe"], year=2015),
]  # fmt: skip

# --- Reference-check benchmark ----------------------------------------------


def test_invented_references_are_spliced_and_match_no_real_title():
    planted = refcheck.invented_references(REAL, 3, random.Random(0))

    assert [p.key for p in planted] == ["invented-1", "invented-2", "invented-3"]
    for reference in planted:
        assert not reference.doi and not reference.arxiv_id
        assert all(refcheck.title_similarity(reference.title, real.title) < 0.8 for real in REAL)
        assert reference.title in reference.raw
    assert planted == refcheck.invented_references(REAL, 3, random.Random(0))  # same seed, same benchmark


def test_wrong_id_references_keep_the_text_but_carry_another_works_id():
    planted = refcheck.wrong_id_references(REAL, 2, random.Random(0))

    assert len(planted) == 2
    for reference in planted:
        victim = next(real for real in REAL if real.title == reference.title)
        assert reference.arxiv_id and reference.arxiv_id != victim.arxiv_id
        assert reference.key.startswith("wrong-id-")


def _row(label, status, paper="p1"):
    return {"paper": paper, "key": f"{label}-{status}", "label": label, "status": status, "as_cited": "t", "found_by": "", "issues": []}


def test_refcheck_scores_are_recomputable_from_the_rows():
    rows = (
        [_row("real", "verified")] * 8 + [_row("real", "not_found")] * 2 + [_row("real", "lookup_failed")]
        + [_row("invented", "not_found")] * 3 + [_row("invented", "verified")]
        + [_row("wrong_id", "mismatch")] * 4
    )  # fmt: skip

    text = refcheck.summarize(rows, "held_out")

    assert "19 references checked, 1 lookups failed and are not scored." in text
    assert "| Real references verified | 0.80 (8/10) |" in text
    assert "| Invented references caught (not verified) | 0.75 (3/4) |" in text
    assert "| Wrong identifiers caught (reported as a mismatch) | 1.00 (4/4) |" in text
    # Flagged: 2 real + 3 invented + 4 wrong_id = 9, of which 7 were planted.
    assert "| Flagged references that were planted | 0.78 (7/9) |" in text
    assert "| real | 8 | 0 | 2 | 1 |" in text
    assert "tuned on these papers" not in text
    assert "tuned on these papers" in refcheck.summarize(rows, "tuning")


def test_refcheck_checks_real_and_planted_references_together(monkeypatch):
    paper = Paper(title="", abstract="", opening="", sentences=[], references={r.key: r for r in REAL})
    monkeypatch.setattr(refcheck, "load_paper", lambda source: paper)
    known = {
        "1512.03385": Work(title="Deep Residual Learning for Image Recognition", arxiv_id="1512.03385"),
        "1706.03762": Work(title="Attention Is All You Need Today", arxiv_id="1706.03762"),
    }
    monkeypatch.setattr(scholar, "arxiv_by_ids", lambda ids: {i: known[i] for i in ids if i in known})
    monkeypatch.setattr(scholar, "crossref_search", lambda text, rows=5: [])
    monkeypatch.setattr(scholar, "arxiv_search", lambda query, max_results=5: [])
    monkeypatch.setattr(scholar, "openalex_search", lambda query, max_results=5: [])

    rows = refcheck.check_paper("0000.00000")

    outcome = {(row["label"], row["status"]) for row in rows}
    assert ("real", "verified") in outcome
    assert ("real", "not_found") in outcome  # reference c is in no fake index
    assert {row["status"] for row in rows if row["label"] == "invented"} == {"not_found"}
    assert {row["status"] for row in rows if row["label"] == "wrong_id"} == {"mismatch"}


# --- Citation-audit benchmark -----------------------------------------------


def _work(key, title, abstract):
    return Work(title=title, abstract=abstract, arxiv_id=f"0000.0000{key}", pdf_url=f"https://arxiv.org/pdf/{key}")


@pytest.fixture
def one_paper(monkeypatch):
    """A paper whose three cited works are all 'on arXiv'."""
    topics = {
        "gears": ("Sparse Gears", "Sparse gears reduce friction in widget transmissions and improve speed."),
        "cogs": ("Dense Cogs", "Dense cogs also reduce friction in widget transmissions but cost more."),
        "whales": ("Whale Song", "Humpback whales sing complex songs during seasonal ocean migration."),
    }
    works = {key: _work(n, *topics[key]) for n, key in enumerate(topics, start=1)}
    sentences = [
        CitedSentence(id=0, section="Intro", keys=["gears"], context="Widgets are slow.",
                      text="Sparse gears were shown to reduce friction in widget transmissions by a wide margin [gears]."),
        CitedSentence(id=1, section="Intro", keys=["gears"],
                      text="A second sentence about the same work is skipped because one per work is enough [gears]."),
        CitedSentence(id=2, section="Intro", keys=["gears", "cogs"],
                      text="Sentences that cite two works at once are left out of the benchmark entirely [gears, cogs]."),
        CitedSentence(id=3, section="Intro", keys=["cogs"], text="Too short to use [cogs]."),
    ]  # fmt: skip
    references = {key: Reference(key=key, title=work.title, arxiv_id=work.arxiv_id) for key, work in works.items()}
    paper = Paper(title="P", abstract="", opening="", sentences=sentences, references=references)
    monkeypatch.setattr(citations, "load_paper", lambda source: paper)
    monkeypatch.setattr(citations, "_arxiv_works", lambda paper, keys: works)
    return works


def test_each_sentence_is_paired_with_its_own_work_a_far_one_and_a_near_one(one_paper):
    items = citations.items_for_paper("0000.00000", "tuning")

    assert [(item["id"], item["key"]) for item in items] == [
        ("0000.00000-s0-original", "gears"),
        ("0000.00000-s0-swapped_far", "whales"),  # shares no vocabulary with the sentence
        ("0000.00000-s0-swapped_near", "cogs"),  # shares the most
    ]
    far = items[1]
    assert far["sentence"]["text"].endswith("by a wide margin [whales].")  # the marker is rewritten
    assert far["sentence"]["keys"] == ["whales"] and far["original_key"] == "gears"
    assert far["work"]["title"] == "Whale Song" and far["split"] == "tuning"
    assert items[0]["sentence"]["context"] == "Widgets are slow."


def test_the_frozen_benchmark_file_is_well_formed():
    items = json.loads(citations.ITEMS_PATH.read_text("utf-8"))
    papers = {split: set(citations.benchmark_papers(split)) for split in ("tuning", "held_out")}

    assert len({item["id"] for item in items}) == len(items)
    by_sentence: dict[str, list[dict]] = {}
    for item in items:
        assert item["paper"] in papers[item["split"]]
        assert item["work"]["abstract"] and item["sentence"]["keys"] == [item["key"]]
        assert f"[{item['key']}]" in item["sentence"]["text"]
        by_sentence.setdefault(item["id"].rsplit("-", 1)[0], []).append(item)
    for group in by_sentence.values():
        # Every sentence appears once with its own work and once with each kind of swap.
        assert sorted(item["label"] for item in group) == sorted(citations.LABELS)
        original = next(item for item in group if item["label"] == "original")
        assert original["key"] == original["original_key"]
        assert all(item["key"] != original["key"] for item in group if item is not original)


def _result(item_id, label, verdict, split="held_out", basis="full_text"):
    return {
        "id": item_id, "paper": "p", "split": split, "label": label, "verdict": verdict, "basis": basis,
        "steps": 2, "model": "claude-sonnet-5-5", "evidence_found": True,
        "usage": {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    }  # fmt: skip


def test_citation_scores_are_recomputable_from_the_rows():
    rows = (
        [_result(f"o{i}", "original", v) for i, v in enumerate(["supported"] * 6 + ["cannot_verify"] * 3 + ["not_supported"])]
        + [_result(f"f{i}", "swapped_far", v) for i, v in enumerate(["not_supported"] * 9 + ["cannot_verify"])]
        + [_result(f"n{i}", "swapped_near", v) for i, v in enumerate(["not_supported"] * 5 + ["partially_supported"] * 3 + ["supported"] * 2)]
        + [_result("t0", "original", "supported", split="tuning")]
    )  # fmt: skip

    text = citations.summarize_rows(rows, errors=2)

    assert "31 items scored, 2 failed attempts on record. Auditor: claude-sonnet-5-5." in text
    held_out = text.split("## Held-out papers")[1].split("## Tuning papers")[0]
    assert "| Original citations judged supported | 0.60 (6/10) |" in held_out
    assert "| Original citations wrongly flagged as not supported | 0.10 (1/10) |" in held_out
    assert "| Far swaps caught (not supported) | 0.90 (9/10) |" in held_out
    assert "| Near swaps caught (not supported) | 0.50 (5/10) |" in held_out
    assert "| Near swaps caught or doubted (not or partly supported) | 0.80 (8/10) |" in held_out
    # 15 "not supported" flags in all: 1 original, 9 far, 5 near.
    assert '| "Not supported" flags that were swaps | 0.93 (14/15) |' in held_out
    assert "| swapped_near | 2 | 3 | 5 | 0 |" in held_out
    assert "| Original citations judged supported | 1.00 (1/1) |" in text.split("## Tuning papers")[1]
    assert "Mean steps per item: 2.0. Tokens: 3100 in, 310 out." in text


@pytest.fixture
def bench(monkeypatch, tmp_path, one_paper):
    """The benchmark's files in a temporary folder, with one paper's items built."""
    monkeypatch.setattr(citations, "ITEMS_PATH", tmp_path / "items.json")
    monkeypatch.setattr(citations, "RESULTS", tmp_path / "runs")
    monkeypatch.setattr(citations, "benchmark_papers", lambda split: ["0000.00000"] if split == "tuning" else [])
    monkeypatch.setattr(config, "AUDITOR_MODEL", "stub")
    monkeypatch.setattr(scholar, "fetch_pdf_text", lambda url: "full text")
    assert citations.main(["build"]) == 0
    return tmp_path


def _auditor_says(monkeypatch, verdict_for):
    def tool_turn(*, agent, messages, **_):
        key = messages[0]["content"].split("[")[1].split("]")[0]
        verdict = {"sentence_id": 0, "verdict": verdict_for(key), "evidence": "", "reason": "r"}
        return tool_calls(("submit_verdicts", {"verdicts": [verdict]})), usage(agent)

    monkeypatch.setattr(llm, "tool_turn", tool_turn)


def test_run_audits_each_item_once_and_resumes(bench, monkeypatch):
    _auditor_says(monkeypatch, lambda key: "cannot_verify" if key == "gears" else "not_supported")

    assert citations.main(["run", "--limit", "2"]) == 0
    assert citations.main(["run"]) == 0  # picks up the one item that was left

    rows = [json.loads(line) for line in (bench / "runs" / "results.jsonl").read_text("utf-8").splitlines()]
    assert sorted((row["label"], row["verdict"]) for row in rows) == [
        ("original", "cannot_verify"), ("swapped_far", "not_supported"), ("swapped_near", "not_supported"),
    ]  # fmt: skip
    assert rows[0]["model"] == "stub" and rows[0]["usage"]["input_tokens"] == 10
    assert citations.main(["summarize"]) == 0
    assert "| Far swaps caught (not supported) | 1.00 (1/1) |" in (bench / "runs" / "summary.md").read_text("utf-8")


def test_an_auditor_that_fails_is_an_error_not_a_verdict(bench, monkeypatch):
    def refuse(**_):
        raise llm.LLMError("auditor: model refused", "refusal")

    monkeypatch.setattr(llm, "tool_turn", refuse)

    assert citations.main(["run"]) == 1

    assert not (bench / "runs" / "results.jsonl").exists()
    errors = [json.loads(line) for line in (bench / "runs" / "errors.jsonl").read_text("utf-8").splitlines()]
    assert len(errors) == 3 and "did not finish" in errors[0]["error"]


def test_a_response_from_the_wrong_model_is_not_scored(bench, monkeypatch):
    _auditor_says(monkeypatch, lambda key: "supported")
    monkeypatch.setattr(config, "AUDITOR_MODEL", "claude-sonnet-5-5")  # the fake answers as "stub"

    assert citations.main(["run", "--limit", "1"]) == 1
    assert "was served by stub" in (bench / "runs" / "errors.jsonl").read_text("utf-8")
