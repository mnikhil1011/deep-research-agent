"""Turns one report into scores.

Each metric is a plain ratio over counted items, so any score can be recomputed
by hand from the per-sentence verdicts saved beside it.

    groundedness       fully supported claims / all claims          (headline)
    citation_accuracy  fully supported sentences / cited sentences
    citation_coverage  cited sentences / all claims
    unsupported_rate   unsupported sentences / cited sentences      (lower is better)
    valid_citations    citation markers naming a real source / all markers
    completeness       aspects answered / aspects                   (answerable cases)
    trap_handled       1 if the false premise was caught, else 0    (trap cases)

A claim is a sentence that carries a citation, or an uncited sentence the judge
classifies as stating a checkable fact. So
groundedness = citation_accuracy x citation_coverage.

A metric is left out of a case's grade when its denominator is zero. A report
with no claims has no groundedness score; it is not counted as 0 or as 1.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.graph.state import Source, Usage
from evals import judge
from evals.cases import Case
from evals.sentences import Sentence, citation_markers, split_report

JUDGE_CONCURRENCY = 4


@dataclass
class Grade:
    scores: dict[str, float]
    explanation: dict[str, str]
    detail: dict = field(default_factory=dict)  # every verdict, for auditing


def _support_verdicts(
    cited: list[Sentence], sources: dict[int, Source], ledger: list[Usage]
) -> dict[int, dict]:
    """Return {sentence index: {"support", "reason"}} for every cited sentence."""
    verdicts: dict[int, dict] = {}
    groups: dict[tuple[int, ...], list[Sentence]] = defaultdict(list)
    for sentence in cited:
        real = tuple(sorted(n for n in sentence.citations if n in sources))
        if real:
            groups[real].append(sentence)
        else:
            verdicts[sentence.index] = {
                "support": "none",
                "reason": "Every source number it cites is missing from the source list.",
            }

    # Sentences citing the same sources are judged together, in one call.
    def judge_group(item: tuple[tuple[int, ...], list[Sentence]]):
        numbers, sentences = item
        return judge.judge_support({n: sources[n] for n in numbers}, sentences, ledger)

    with ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as pool:
        for result in pool.map(judge_group, groups.items()):
            for index, verdict in result.items():
                verdicts[index] = {"support": verdict.support, "reason": verdict.reason}
    return verdicts


def grade_report(case: Case, report: str, sources: dict[int, Source], ledger: list[Usage]) -> Grade:
    """Score one report. `sources` maps the report's citation numbers to sources.

    Judge token usage is appended to `ledger` as calls complete, so it is still
    accounted for if grading fails part-way.
    """
    sentences = split_report(report)
    cited = [s for s in sentences if s.citations]
    uncited = [s for s in sentences if not s.citations]

    support = _support_verdicts(cited, sources, ledger)
    kinds = judge.judge_claims(uncited, ledger) if uncited else {}

    full = sum(1 for v in support.values() if v["support"] == "full")
    partial = sum(1 for v in support.values() if v["support"] == "partial")
    unsupported = sum(1 for v in support.values() if v["support"] == "none")
    uncited_claims = sum(1 for v in kinds.values() if v.kind == "claim")
    claims = len(cited) + uncited_claims
    markers = citation_markers(report)
    valid_markers = sum(1 for n in markers if n in sources)

    scores: dict[str, float] = {}
    explanation: dict[str, str] = {}
    if claims:
        scores["groundedness"] = full / claims
        scores["citation_coverage"] = len(cited) / claims
        explanation["groundedness"] = (
            f"{full} of {claims} claims fully supported "
            f"({partial} partly, {unsupported} unsupported, {uncited_claims} uncited)."
        )
    if cited:
        scores["citation_accuracy"] = full / len(cited)
        scores["unsupported_rate"] = unsupported / len(cited)
    if markers:
        scores["valid_citations"] = valid_markers / len(markers)

    detail: dict = {
        "counts": {
            "sentences": len(sentences),
            "cited": len(cited),
            "full": full,
            "partial": partial,
            "unsupported": unsupported,
            "uncited_claims": uncited_claims,
            "claims": claims,
            "citation_markers": len(markers),
            "valid_citation_markers": valid_markers,
        },
        "sentences": [
            {
                "index": s.index,
                "text": s.text,
                "citations": list(s.citations),
                **(support[s.index] if s.citations else {}),
                **(
                    {"kind": kinds[s.index].kind, "reason": kinds[s.index].reason}
                    if not s.citations
                    else {}
                ),
            }
            for s in sentences
        ],
    }

    if case.is_trap:
        verdict = judge.judge_trap(case.question, case.false_premise, report, ledger)
        scores["trap_handled"] = 1.0 if verdict.handled else 0.0
        explanation["trap_handled"] = verdict.reason
        detail["trap"] = {"handled": verdict.handled, "reason": verdict.reason}
    else:
        aspects = judge.judge_completeness(case.question, case.aspects, report, ledger)
        answered = sum(1 for v in aspects if v.status == "answered")
        scores["completeness"] = answered / len(case.aspects)
        explanation["completeness"] = " ".join(
            f"[{v.status}] {aspect}: {v.reason}" for aspect, v in zip(case.aspects, aspects)
        )
        detail["aspects"] = [
            {"aspect": aspect, "status": v.status, "reason": v.reason}
            for aspect, v in zip(case.aspects, aspects)
        ]
    return Grade(scores=scores, explanation=explanation, detail=detail)
