"""Tests for the paper audit. No model or network calls are made.

The scholarly services are replaced by a small in-memory catalogue, and the
models by scripted replies, so these tests check the plumbing and the rules
that code enforces. They cannot show that a real auditor's verdicts are right.
"""

import json
import sys
from types import SimpleNamespace

import pytest
from fakes import tool_calls, usage

import audit
from app import config, llm
from app.audit import analyst, latex, resolver
from app.audit.auditor import audit_citations, quote_found, rank_passages, split_passages
from app.audit.builder import build_audit_graph
from app.audit.models import CitedSentence, Contribution, Reference
from app.audit.scout import scout_related_work
from app.tools import scholar
from app.tools.scholar import ScholarError, Work

# --- A small paper on disk ---------------------------------------------------

MAIN_TEX = r"""
\documentclass{article}
\title{Faster Widgets with Sparse Gears}
\begin{document}
\maketitle
\begin{abstract}
We make widgets faster. % a comment that must not appear
\end{abstract}
\section{Introduction}
Widgets are slow \citep{smith2020}. Prior work, e.g.\ on gears \cite{jones2019,smith2020}, helps.
\input{method}
\begin{table}
A row \cite{tableonly2018} & 1.0
\end{table}
\bibliography{refs}
\end{document}
"""
METHOD_TEX = r"""
\section{Method}
We follow \citet{lee2021} and set $k=3$\footnote{A note \cite{ghost2000}.}.
Short one \cite{jones2019}.
This sentence cites a key that has no entry \cite{nokey}.
"""
REFS_BIB = r"""
@article{smith2020,
  title = {Slow {W}idgets: A Study},
  author = {Smith, Jane and Doe, John},
  journal = {Journal of Widgets},
  year = {2020},
  doi = {10.1000/widgets.1}
}
@inproceedings{jones2019,
  title = "Gears for Widgets",
  author = "Alan Jones and others",
  booktitle = "Proc. of GearConf",
  year = 2019,
  eprint = {1901.00001},
  archivePrefix = {arXiv}
}
@misc{lee2021, title={Sparse Methods}, author={Lee, Kim}, year={2021}, url={https://arxiv.org/abs/2101.00002}}
@book{tableonly2018, title={A Book Cited Only in a Table}, author={Roe, Ann}, year={2018}}
@article{unused1999, title={Never Cited Anywhere}, author={Old, Tim}, year={1999}}
"""


@pytest.fixture
def paper_dir(tmp_path):
    folder = tmp_path / "paper"
    folder.mkdir()
    (folder / "main.tex").write_text(MAIN_TEX, "utf-8")
    (folder / "method.tex").write_text(METHOD_TEX, "utf-8")
    (folder / "refs.bib").write_text(REFS_BIB, "utf-8")
    return folder


# --- Reading LaTeX -----------------------------------------------------------


def test_paper_is_read_into_sentences_with_their_citations(paper_dir):
    paper = latex.load_paper(str(paper_dir))

    assert paper.title == "Faster Widgets with Sparse Gears"
    assert paper.abstract == "We make widgets faster."
    assert [(s.id, s.section, s.keys) for s in paper.sentences] == [
        (0, "Introduction", ["smith2020"]),
        (1, "Introduction", ["jones2019", "smith2020"]),
        (2, "Method", ["lee2021"]),
        (3, "Method", ["nokey"]),
    ]  # "Short one" is under the minimum length, and the footnote and table are not prose
    assert paper.sentences[0].text == "Widgets are slow [smith2020]."
    assert paper.sentences[1].text == "Prior work, e.g. on gears [jones2019, smith2020], helps."
    assert paper.sentences[1].context == "Widgets are slow [smith2020]."
    assert paper.sentences[2].text == "We follow [lee2021] and set $k=3$."


def test_bibliography_fields_are_parsed_from_bibtex(paper_dir):
    references = latex.load_paper(str(paper_dir)).references

    smith = references["smith2020"]
    assert (smith.title, smith.year, smith.doi) == ("Slow Widgets: A Study", 2020, "10.1000/widgets.1")
    assert smith.authors == ["Jane Smith", "John Doe"]
    assert references["jones2019"].authors == ["Alan Jones"]  # "and others" is not an author
    assert (references["jones2019"].year, references["jones2019"].arxiv_id) == (2019, "1901.00001")
    assert references["lee2021"].arxiv_id == "2101.00002"  # taken from the URL


def test_missing_and_uncited_keys_are_reported(paper_dir):
    paper = latex.load_paper(str(paper_dir))

    assert paper.missing_keys == ["ghost2000", "nokey"]
    # A work cited only inside a table still counts as cited. An unused entry in
    # the .bib database is not part of the paper, so it is simply left out.
    assert "tableonly2018" in paper.references and "unused1999" not in paper.references
    assert paper.uncited_keys == []


def test_a_compiled_bibliography_is_parsed_when_there_is_no_bib_file(tmp_path):
    (tmp_path / "main.tex").write_text(
        r"\documentclass{article}\begin{document}"
        r"Attention works well for translation tasks \cite{vaswani2017}."
        r"\bibliography{missing}\end{document}",
        "utf-8",
    )
    (tmp_path / "main.bbl").write_text(
        r"""\begin{thebibliography}{1}
\bibitem[Vaswani et~al.(2017)Vaswani, Shazeer]{vaswani2017}
Ashish Vaswani and Noam Shazeer.
\newblock Attention is all you need.
\newblock In {\em NeurIPS}, 2017. arXiv:1706.03762.
\bibitem{plain2001}
Some Author. A two-block entry with a DOI, 2001. \url{https://doi.org/10.1000/xyz.5}.
\end{thebibliography}""",
        "utf-8",
    )

    paper = latex.load_paper(str(tmp_path / "main.tex"))
    references = paper.references
    assert paper.uncited_keys == ["plain2001"]  # printed in the reference list, never cited

    vaswani = references["vaswani2017"]
    assert (vaswani.title, vaswani.year, vaswani.arxiv_id) == ("Attention is all you need", 2017, "1706.03762")
    assert vaswani.raw.startswith("Ashish Vaswani and Noam Shazeer. Attention is all you need.")
    assert (references["plain2001"].title, references["plain2001"].doi) == ("", "10.1000/xyz.5")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("see https://doi.org/10.1162/tacl_a_00028. Also arXiv:1706.03762", ("10.1162/tacl_a_00028", "1706.03762")),
        ("doi: 10.18653/v1/N19-1423, CoRR abs/1409.0473", ("10.18653/v1/n19-1423", "1409.0473")),
        ("Nature 521, 10.1038/nature14539.", ("10.1038/nature14539", "")),
        # A DOI-like run inside another site's URL path is not a DOI.
        ("https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00028/1567626/x.pdf", ("", "")),
    ],
)
def test_identifiers_are_found_in_reference_text(text, expected):
    assert latex._identifiers(text) == expected


def test_reading_fails_clearly_on_something_that_is_not_a_paper(tmp_path):
    with pytest.raises(latex.PaperError, match="not a file, a folder or an arXiv id"):
        latex.load_paper(str(tmp_path / "nothing.tex"))
    (tmp_path / "notes.tex").write_text("just notes", "utf-8")
    with pytest.raises(latex.PaperError, match="documentclass"):
        latex.load_paper(str(tmp_path))


# --- A fake scholarly record -------------------------------------------------

SMITH = Work(title="Slow Widgets: A Study", authors=["Jane Smith", "John Doe"], year=2020,
             doi="10.1000/widgets.1", source="crossref")  # fmt: skip
JONES = Work(title="Gears for Widgets", authors=["Alan Jones"], year=2019, arxiv_id="1901.00001",
             abstract="We show that gears make widgets 40% faster on the WidgetBench suite.",
             pdf_url="https://arxiv.org/pdf/1901.00001", source="arxiv")  # fmt: skip
LEE = Work(title="Sparse Methods", authors=["Kim Lee"], year=2021, arxiv_id="2101.00002",
           abstract="Sparse methods reduce cost.", pdf_url="https://arxiv.org/pdf/2101.00002", source="arxiv")  # fmt: skip
SMITH_OPENALEX = Work(title="Slow Widgets: A Study", doi="10.1000/widgets.1", openalex_id="W1",
                      abstract="Widgets are slow because of friction.", source="openalex")  # fmt: skip
RIVAL = Work(title="Sparse Gears Make Widgets Fast", authors=["Rival One"], year=2022, openalex_id="W9",
             abstract="We speed up widgets with sparse gears.", cited_by_count=50, source="openalex")  # fmt: skip
PAPER_ITSELF = Work(title="Faster Widgets With Sparse Gears", openalex_id="W7", source="openalex")
JONES_OPENALEX = Work(title="Gears for Widgets", arxiv_id="1901.00001", openalex_id="W2", source="openalex")
FULL_TEXT = (
    "Introduction. Widgets matter. " * 60
    + "In our main experiment the sparse gear reduced friction by 31 percent on steel widgets. "
    + "Related work is discussed elsewhere. " * 60
)


@pytest.fixture
def catalogue(monkeypatch):
    """Replace every scholarly lookup with answers from the records above."""
    calls = SimpleNamespace(log=[], fail=set())

    def service(name, answer):
        def call(*args, **kwargs):
            calls.log.append(name)
            if name in calls.fail:
                raise ScholarError(f"{name} is down")
            return answer(*args, **kwargs)

        monkeypatch.setattr(scholar, name, call)

    by_arxiv = {JONES.arxiv_id: JONES, LEE.arxiv_id: LEE}
    service("arxiv_by_ids", lambda ids: {i: by_arxiv[i] for i in ids if i in by_arxiv})
    service("arxiv_search", lambda query, max_results=5: [])
    service("crossref_by_doi", lambda doi: SMITH if doi == SMITH.doi else None)
    service("crossref_search", lambda text, rows=5: [])
    service("openalex_by_doi", lambda doi: SMITH_OPENALEX if doi == SMITH.doi else None)
    service("openalex_search", lambda query, max_results=5: [RIVAL, PAPER_ITSELF, JONES_OPENALEX])
    service("openalex_citing", lambda openalex_id, max_results=8: [])
    service("openalex_references", lambda openalex_id, max_results=8: [])
    service("fetch_pdf_text", lambda url: FULL_TEXT)
    return calls


# --- Reference check ---------------------------------------------------------


def _reference(**fields) -> Reference:
    return Reference(key="k", **fields)


def test_references_are_resolved_by_their_strongest_identifier(paper_dir, catalogue):
    resolutions = {r.key: r for r in resolver.resolve_all(latex.load_paper(str(paper_dir)))}

    assert {key: r.status for key, r in resolutions.items()} == {
        "smith2020": "verified", "jones2019": "verified", "lee2021": "verified",
        "tableonly2018": "not_found",
    }  # fmt: skip
    assert resolutions["jones2019"].found_by == "arXiv:1901.00001"
    assert resolutions["smith2020"].found_by == "DOI 10.1000/widgets.1"
    # Crossref had no abstract, so the cited work was filled in from OpenAlex.
    assert resolutions["smith2020"].work.abstract == "Widgets are slow because of friction."
    assert resolutions["smith2020"].work.openalex_id == "W1"
    assert "No matching record" in resolutions["tableonly2018"].issues[0]
    assert catalogue.log.count("arxiv_by_ids") == 1  # one batched request, not one per reference


def test_an_identifier_that_points_at_another_work_is_a_mismatch(catalogue):
    wrong = _reference(title="A Completely Different Title", doi=SMITH.doi, raw="Different. 2020.")

    resolution = resolver.resolve(wrong, {}, enrich=False)

    assert resolution.status == "mismatch"
    assert 'is registered to "Slow Widgets: A Study"' in resolution.issues[0]


def test_a_reference_that_no_service_knows_is_not_found(catalogue):
    invented = _reference(
        title="Quantum Widgets Considered Harmful", doi="10.9999/fake", arxiv_id="9999.99999",
        raw="A. Nobody. Quantum Widgets Considered Harmful. 2023.",
    )  # fmt: skip

    resolution = resolver.resolve(invented, {}, enrich=False)

    assert resolution.status == "not_found"
    assert resolution.issues[0] == "arXiv has no paper with the id 9999.99999."
    assert resolution.issues[1] == "Crossref has no record of the DOI 10.9999/fake."


def test_a_failed_search_is_reported_as_failed_not_as_not_found(catalogue):
    catalogue.fail.add("crossref_search")

    resolution = resolver.resolve(_reference(title="Some Real Paper Title", raw="x"), {}, enrich=False)

    assert resolution.status == "lookup_failed"
    assert "crossref_search is down" in resolution.issues[0]


def test_retractions_and_wrong_years_are_flagged(catalogue, monkeypatch):
    retracted = SMITH.model_copy(update={"is_retracted": True, "year": 2012})
    monkeypatch.setattr(scholar, "crossref_by_doi", lambda doi: retracted)

    resolution = resolver.resolve(
        _reference(title="Slow Widgets: A Study", doi=SMITH.doi, year=2020), {}, enrich=False
    )

    assert resolution.status == "verified"
    assert resolution.issues == [
        "This work has been retracted.",
        "Cited as 2020, but the record is dated 2012.",
    ]


@pytest.mark.parametrize(
    ("reference", "work", "expected"),
    [
        # Hyphenation and spacing differences do not matter.
        (_reference(title="Long-term dependencies in nets"), Work(title="LongTerm Dependencies in Nets"), True),
        # A title that was not parsed out is looked for in the whole entry.
        (_reference(raw="J. Smith and J. Doe. Slow widgets: a study. JW, 2020."), SMITH, True),
        # A shortened title counts only when an author agrees.
        (_reference(title="Slow widgets: a", raw="Jane Smith. Slow widgets: a. 2020"), SMITH, True),
        (_reference(title="Slow widgets: a", raw="Someone Else. Slow widgets: a. 2020"), SMITH, False),
        (_reference(title="Deep learning", raw="LeCun. Deep learning."), Work(title="Deep learning for AI", authors=["Y LeCun"]), False),
        (_reference(title="Fast Widgets: A Survey"), SMITH, False),
    ],
)
def test_matching_a_reference_to_a_record(reference, work, expected):
    assert resolver.matches(reference, work) is expected


# --- Citation auditor --------------------------------------------------------


def test_passage_search_finds_the_relevant_part_of_a_paper():
    passages = split_passages(FULL_TEXT)

    best = rank_passages("sparse gear friction steel", passages)

    assert "reduced friction by 31 percent" in passages[best[0]]
    assert rank_passages("zebra", passages) == []


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("gears make widgets 40% faster", True),
        ("Gears  make widgets\n40% FASTER!", True),  # case, spacing and punctuation are ignored
        ("gears make widgets ... on the WidgetBench suite", True),  # each part of an elided quote
        ("gears make widgets 80% faster", False),
        ("gears make", False),  # too short to count as evidence
        ("", False),
    ],
)
def test_quotations_are_checked_against_the_text(quote, expected):
    assert quote_found(quote, JONES.abstract) is expected


def _sentences(*texts: str) -> list[CitedSentence]:
    return [CitedSentence(id=i, text=text, section="Intro", keys=["jones2019"]) for i, text in enumerate(texts)]


def _auditor_model(monkeypatch, *replies):
    """Script the auditor's model: one reply per turn."""
    turns = iter(replies)
    seen = SimpleNamespace(prompts=[], tools=[])

    def tool_turn(*, agent, messages, tools, **_):
        seen.prompts.append(messages[0]["content"])
        seen.tools.append([tool["name"] for tool in tools])
        return next(turns), usage(agent)

    monkeypatch.setattr(llm, "tool_turn", tool_turn)
    return seen


def _verdict(sentence_id, verdict, evidence="", reason="because"):
    return {"sentence_id": sentence_id, "verdict": verdict, "evidence": evidence, "reason": reason}


def test_auditor_verdicts_are_kept_only_when_their_evidence_is_real(catalogue, monkeypatch):
    seen = _auditor_model(
        monkeypatch,
        tool_calls(("search_paper", {"query": "sparse gear friction steel"})),
        tool_calls(("submit_verdicts", {"verdicts": [
            _verdict(0, "supported", "gears make widgets 40% faster"),
            _verdict(1, "supported", "the sparse gear reduced friction by 31 percent on steel widgets"),
            _verdict(2, "supported", "gears make widgets fly to the moon"),
            _verdict(3, "not_supported"),
            _verdict(99, "supported", "gears make widgets 40% faster"),
        ]})),
    )  # fmt: skip
    sentences = _sentences("Gears speed up widgets [jones2019].", "Friction fell by 31% [jones2019].",
                           "Widgets can fly [jones2019].", "Gears are slow [jones2019].",
                           "A sentence the auditor skips [jones2019].")  # fmt: skip

    result = audit_citations({"key": "jones2019", "work": JONES, "sentences": sentences})

    verdicts = {v.sentence_id: v for v in result["verdicts"]}
    assert [verdicts[i].verdict for i in range(5)] == [
        "supported",       # the quotation is in the abstract
        "supported",       # the quotation is in a passage the search returned
        "cannot_verify",   # the quotation is nowhere in the text: downgraded
        "not_supported",   # needs no quotation
        "cannot_verify",   # no verdict was returned for it
    ]  # fmt: skip
    assert "Downgraded" in verdicts[2].reason and not verdicts[2].evidence_found
    assert verdicts[4].reason == "The auditor returned no verdict for this sentence."
    assert 99 not in verdicts
    assert {v.basis for v in result["verdicts"] if v.sentence_id < 4} == {"full_text"}
    assert seen.tools[0] == ["search_paper", "submit_verdicts"]
    assert '<sentence id="1" section="Intro">' in seen.prompts[0]
    assert [c.name for c in result["tool_calls"]] == ["search_paper", "submit_verdicts"]


def test_auditor_without_full_text_gets_no_search_tool(catalogue, monkeypatch):
    abstract_only = JONES.model_copy(update={"pdf_url": ""})
    seen = _auditor_model(
        monkeypatch,
        tool_calls(("submit_verdicts", {"verdicts": [_verdict(0, "cannot_verify")]})),
    )

    result = audit_citations({"key": "jones2019", "work": abstract_only, "sentences": _sentences("X [jones2019].")})

    assert seen.tools[0] == ["submit_verdicts"]
    assert "Only the abstract is available" in seen.prompts[0]
    assert result["verdicts"][0].basis == "abstract"


def test_a_work_with_no_text_at_all_is_not_sent_to_the_model(catalogue, monkeypatch):
    monkeypatch.setattr(llm, "tool_turn", lambda **_: pytest.fail("the model was called"))
    nothing = Work(title="A Paywalled Book")

    result = audit_citations({"key": "k", "work": nothing, "sentences": _sentences("X [k].", "Y [k].")})

    assert [v.verdict for v in result["verdicts"]] == ["cannot_verify", "cannot_verify"]
    assert "Neither an abstract nor an open-access copy" in result["verdicts"][0].reason


def test_an_auditor_that_never_submits_leaves_a_note_not_a_crash(catalogue, monkeypatch):
    monkeypatch.setattr(
        llm, "tool_turn", lambda *, agent, **_: (tool_calls(("search_paper", {"query": "gears"})), usage(agent))
    )

    result = audit_citations({"key": "jones2019", "work": JONES, "sentences": _sentences("X [jones2019].")})

    assert result["verdicts"][0].verdict == "cannot_verify"
    assert "did not finish: ran out of steps" in result["notes"][0]


# --- Related-work scout ------------------------------------------------------

CONTRIBUTION = Contribution(id=1, statement="Sparse gears make widgets faster.")


def _scout_payload(**overrides):
    payload = {
        "contribution": CONTRIBUTION,
        "paper_title": "Faster Widgets with Sparse Gears",
        "paper_abstract": "We make widgets faster.",
        "cited_works": [JONES],
        "cited_titles": ["Gears for Widgets"],
    }
    return {**payload, **overrides}


def test_scout_keeps_only_real_new_candidates(catalogue, monkeypatch):
    turns = iter([
        tool_calls(("search_papers", {"query": "sparse gears widgets"})),
        tool_calls(("get_citing", {"paper_id": "openalex:W9"}), ("get_citing", {"paper_id": "made-up"})),
        tool_calls(("submit_candidates", {"candidates": [
            {"paper_id": "openalex:W9", "why": "Same idea."},
            {"paper_id": "openalex:W404", "why": "An id no tool returned."},
            {"paper_id": "openalex:W7", "why": "This is the paper under review."},
            {"paper_id": "openalex:W2", "why": "Already in the reference list."},
            {"paper_id": "openalex:W9", "why": "A repeat."},
        ]})),
    ])  # fmt: skip
    monkeypatch.setattr(llm, "tool_turn", lambda *, agent, **_: (next(turns), usage(agent)))

    result = scout_related_work(_scout_payload())

    assert [(c.work.ref, c.already_cited) for c in result["candidates"]] == [
        ("openalex:W9", False),
        ("openalex:W2", True),
    ]
    errors = [c.output for c in result["tool_calls"] if c.is_error]
    assert errors == ["Error: Unknown paper id made-up. Use an id from an earlier result."]


def test_scout_falls_back_to_arxiv_when_openalex_is_unavailable(catalogue, monkeypatch):
    catalogue.fail.add("openalex_search")
    monkeypatch.setattr(scholar, "arxiv_search", lambda query, max_results=5: [LEE] if query == "all:sparse AND all:gears" else [])
    turns = iter([
        tool_calls(("search_papers", {"query": "sparse gears"})),
        tool_calls(("submit_candidates", {"candidates": [{"paper_id": "arxiv:2101.00002", "why": "Close."}]})),
    ])  # fmt: skip
    monkeypatch.setattr(llm, "tool_turn", lambda *, agent, **_: (next(turns), usage(agent)))

    result = scout_related_work(_scout_payload())

    assert [c.work.title for c in result["candidates"]] == ["Sparse Methods"]
    assert "these results are from arXiv only" in result["tool_calls"][0].output


# --- Analyst -----------------------------------------------------------------


def test_contributions_are_capped_and_numbered_by_code(paper_dir, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTRIBUTIONS", 2)
    monkeypatch.setattr(
        llm, "structured",
        lambda *, agent, schema, **_: (schema(contributions=["First  claim.", " ", "Second.", "Third."]), usage(agent)),
    )  # fmt: skip

    result = analyst.extract_contributions({"paper": latex.load_paper(str(paper_dir))})

    assert [(c.id, c.statement) for c in result["contributions"]] == [(1, "First claim.")]


def test_assessments_of_papers_that_were_not_offered_are_dropped(monkeypatch):
    def structured(*, agent, schema, **_):
        item = {"reason": "r", "overlap": "high", "approach": "a", "data": "not stated", "result": "x"}
        return schema(assessments=[{"paper_id": "openalex:W9", **item}, {"paper_id": "openalex:W404", **item}]), usage(agent)

    monkeypatch.setattr(llm, "structured", structured)
    candidates = [{"contribution_id": 1, "work": RIVAL, "why": "w"}]

    result = analyst.assess_overlap({"contribution": CONTRIBUTION, "candidates": [analyst.Candidate(**c) for c in candidates]})

    assert [(a.work_ref, a.overlap) for a in result["assessments"]] == [("openalex:W9", "high")]


def test_a_refused_analyst_call_becomes_a_note(paper_dir, monkeypatch):
    def refuse(**_):
        raise llm.LLMError("analyst: model refused", "refusal")

    monkeypatch.setattr(llm, "structured", refuse)

    result = analyst.extract_contributions({"paper": latex.load_paper(str(paper_dir))})

    assert result["contributions"] == []
    assert "could not be extracted" in result["notes"][0]


# --- The whole audit ---------------------------------------------------------


@pytest.fixture
def models(monkeypatch):
    """Script every model in the audit graph."""
    calls = SimpleNamespace(agents=[])

    def tool_turn(*, agent, messages, **_):
        calls.agents.append(agent)
        prompt = messages[0]["content"]
        turn = sum(1 for m in messages if m["role"] == "assistant")
        if agent == "auditor":
            ids = [int(part.split('"')[0]) for part in prompt.split('<sentence id="')[1:]]
            if "[jones2019]" in prompt.split("\n")[0]:
                verdicts = [_verdict(i, "not_supported", reason="The work is about gears, not this.") for i in ids]
            else:
                verdicts = [_verdict(i, "cannot_verify") for i in ids]
            return tool_calls(("submit_verdicts", {"verdicts": verdicts})), usage(agent)
        if turn == 0:
            return tool_calls(("search_papers", {"query": "sparse gears"})), usage(agent)
        candidates = [{"paper_id": "openalex:W9", "why": "Same idea."}, {"paper_id": "openalex:W2", "why": "Cited."}]
        return tool_calls(("submit_candidates", {"candidates": candidates})), usage(agent)

    def structured(*, agent, schema, prompt, **_):
        calls.agents.append(agent)
        if schema is analyst.Claims:
            return schema(contributions=["Sparse gears make widgets faster."]), usage(agent)
        item = {"reason": "Same problem and approach.", "approach": "sparse gears", "data": "not stated", "result": "faster"}
        return schema(assessments=[
            {"paper_id": "openalex:W9", "overlap": "high", **item},
            {"paper_id": "openalex:W2", "overlap": "partial", **item},
        ]), usage(agent)  # fmt: skip

    monkeypatch.setattr(llm, "tool_turn", tool_turn)
    monkeypatch.setattr(llm, "structured", structured)
    return calls


def test_full_audit_runs_every_stage_and_reports_what_it_found(paper_dir, catalogue, models):
    state = build_audit_graph().invoke({"source": str(paper_dir), "mode": "full"})

    # jones2019 is cited in one sentence; smith2020 in two; lee2021 in one.
    assert sorted((v.key, v.sentence_id, v.verdict) for v in state["verdicts"]) == [
        ("jones2019", 1, "not_supported"),
        ("lee2021", 2, "cannot_verify"),
        ("smith2020", 0, "cannot_verify"),
        ("smith2020", 1, "cannot_verify"),
    ]
    assert [c.statement for c in state["contributions"]] == ["Sparse gears make widgets faster."]
    assert [(c.work.ref, c.already_cited) for c in state["candidates"]] == [("openalex:W9", False), ("openalex:W2", True)]
    assert sorted(models.agents) == ["analyst", "analyst", "auditor", "auditor", "auditor", "scout", "scout"]

    report = state["report"]
    assert report.startswith("# Paper audit: Faster Widgets with Sparse Gears")
    first = report.split("## What to look at first")[1].split("##")[0]
    assert "1. 1 citation(s) are not supported by the work they cite." in first
    assert "2. 1 uncited paper(s) appear to make a contribution the paper claims as its own." in first
    assert "3. 1 reference(s) could not be found in any index." in first
    assert "4. 2 citation key(s) have no entry in the bibliography." in first
    assert "- **[jones2019]** in *Introduction*, checked against the abstract only:" in report
    assert "| [Sparse Gears Make Widgets Fast (2022)](https://openalex.org/W9) | High | **No** |" in report
    assert "| [Gears for Widgets](https://arxiv.org/abs/1901.00001) | Partial | Yes |" in report
    assert "### Comparison of the related work" in report
    assert "Cited in the text but missing from the bibliography: `ghost2000`, `nokey`." in report


def test_references_mode_makes_no_model_calls(paper_dir, catalogue, monkeypatch):
    monkeypatch.setattr(llm, "tool_turn", lambda **_: pytest.fail("a model was called"))
    monkeypatch.setattr(llm, "structured", lambda **_: pytest.fail("a model was called"))

    state = build_audit_graph().invoke({"source": str(paper_dir), "mode": "references"})

    assert len(state["resolutions"]) == 4
    assert not state.get("verdicts") and not state.get("contributions")
    assert "## Reference check" in state["report"] and "## Citation audit" not in state["report"]


def test_citations_mode_stops_before_the_related_work_search(paper_dir, catalogue, models):
    state = build_audit_graph().invoke({"source": str(paper_dir), "mode": "citations"})

    assert set(models.agents) == {"auditor"}
    assert "## Citation audit" in state["report"] and "## Related work" not in state["report"]


def test_only_the_most_cited_works_are_audited_when_over_the_limit(paper_dir, catalogue, models, monkeypatch):
    monkeypatch.setattr(config, "MAX_REFERENCES_AUDITED", 1)

    state = build_audit_graph().invoke({"source": str(paper_dir), "mode": "citations"})

    assert {v.key for v in state["verdicts"]} == {"smith2020"}  # cited twice, the others once
    assert "2 cited works were not audited, because the limit is 1." in state["notes"][0]
    assert "## Problems during the audit" in state["report"]


def test_cli_prints_progress_and_saves_the_audit(paper_dir, catalogue, models, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["audit.py", str(paper_dir)])
    monkeypatch.chdir(tmp_path)

    assert audit.main() == 0

    out = capsys.readouterr().out
    assert "[read_paper] Faster Widgets with Sparse Gears: 4 references, 4 citing sentences" in out
    assert "[check_references] 3 verified, 1 not_found" in out
    assert "[auditor] [jones2019] 1 not_supported" in out
    assert "[scout] 2 candidate papers for contribution 1" in out
    folder = tmp_path / "audits" / "faster-widgets-with-sparse-gears"
    assert (folder / "audit.md").read_text("utf-8").startswith("# Paper audit")
    saved = json.loads((folder / "audit.json").read_text("utf-8"))
    assert saved["paper"]["title"] == "Faster Widgets with Sparse Gears"
    assert saved["verdicts"][0]["key"] in ("jones2019", "lee2021", "smith2020")


def test_cli_reports_an_unreadable_paper_without_a_traceback(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["audit.py", str(tmp_path / "missing.tex")])

    assert audit.main() == 1
    assert "The audit could not start" in capsys.readouterr().err
