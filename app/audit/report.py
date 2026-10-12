"""Renders the audit as Markdown.

No model writes this report. Every line is assembled by code from the verdicts
and records in the state, so the report cannot say something the audit did not
find.
"""

from collections import Counter

from app.audit.models import AuditState, CitationVerdict, Resolution
from app.tools.scholar import Work

_OVERLAP_ORDER = {"high": 0, "partial": 1, "low": 2}
_STATUS_LABEL = {
    "verified": "Verified",
    "mismatch": "Identifier mismatch",
    "not_found": "Not found",
    "lookup_failed": "Lookup failed",
}


def _cell(text: str) -> str:
    """Make text safe inside a Markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


def work_url(work: Work) -> str:
    if work.doi:
        return f"https://doi.org/{work.doi}"
    if work.arxiv_id:
        return f"https://arxiv.org/abs/{work.arxiv_id}"
    if work.openalex_id:
        return f"https://openalex.org/{work.openalex_id}"
    return ""


def _work_link(work: Work) -> str:
    label = _cell(f"{work.title} ({work.year})" if work.year else work.title)
    url = work_url(work)
    return f"[{label}]({url})" if url else label


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _reference_section(state: AuditState) -> list[str]:
    paper, resolutions = state["paper"], state.get("resolutions", [])
    counts = Counter(r.status for r in resolutions)
    lines = [
        "## Reference check",
        "",
        f"{len(resolutions)} references were checked against arXiv, Crossref and OpenAlex: "
        f"{counts['verified']} verified, {counts['mismatch']} with an identifier that points "
        f"to a different work, {counts['not_found']} not found, and "
        f"{counts['lookup_failed']} that could not be looked up.",
    ]
    flagged = [r for r in resolutions if r.status != "verified" or r.issues]
    if flagged:
        rows = []
        for r in flagged:
            reference = paper.references[r.key]
            rows.append(
                [
                    f"`{_cell(r.key)}`",
                    _cell(reference.title or reference.raw[:120]),
                    _STATUS_LABEL[r.status],
                    _cell(" ".join(r.issues)),
                ]
            )
        lines += ["", *_table(["Key", "As cited", "Status", "What was found"], rows)]
        lines += [
            "",
            '"Not found" is a prompt to check, not proof that a reference is invented. Books, '
            "reports, software and web pages are often absent from these indexes.",
        ]
    if paper.missing_keys:
        keys = ", ".join(f"`{key}`" for key in paper.missing_keys)
        lines += ["", f"Cited in the text but missing from the bibliography: {keys}."]
    if paper.uncited_keys:
        keys = ", ".join(f"`{key}`" for key in paper.uncited_keys)
        lines += ["", f"In the bibliography but never cited: {keys}."]
    return lines


def _citation_block(verdict: CitationVerdict, state: AuditState, resolved: dict[str, Resolution]) -> list[str]:
    sentence = state["paper"].sentences[verdict.sentence_id]
    work = resolved[verdict.key].work
    basis = "full text" if verdict.basis == "full_text" else "abstract only"
    lines = [
        f"- **[{verdict.key}]** in *{sentence.section or 'the paper'}*, checked against the {basis}:",
        f"  - Sentence: {' '.join(sentence.text.split())}",
        f"  - Cited work: {_work_link(work)}",
        f"  - Finding: {' '.join(verdict.reason.split())}",
    ]
    if verdict.evidence and verdict.evidence_found:
        lines.append(f'  - The cited work says: "{" ".join(verdict.evidence.split())}"')
    return lines


def _citation_section(state: AuditState) -> list[str]:
    verdicts = state.get("verdicts", [])
    if not verdicts:
        return []
    resolved = {r.key: r for r in state.get("resolutions", [])}
    counts = Counter(v.verdict for v in verdicts)
    full_text = sum(1 for v in verdicts if v.basis == "full_text")
    lines = [
        "## Citation audit",
        "",
        f"{len(verdicts)} citations were checked, each a sentence paired with one work it "
        f"cites. {full_text} were checked against the cited work's full text and "
        f"{len(verdicts) - full_text} against its abstract only.",
        "",
        *_table(
            ["Verdict", "Citations"],
            [
                ["Supported", str(counts["supported"])],
                ["Partly supported", str(counts["partially_supported"])],
                ["Not supported", str(counts["not_supported"])],
                ["Could not be verified", str(counts["cannot_verify"])],
            ],
        ),
    ]
    for verdict_name, heading, intro in (
        ("not_supported", "Not supported", "The cited work does not appear to say what the sentence attributes to it."),
        ("partially_supported", "Partly supported", "The cited work backs only part of what the sentence attributes to it."),
    ):
        chosen = sorted((v for v in verdicts if v.verdict == verdict_name), key=lambda v: v.sentence_id)
        if chosen:
            lines += ["", f"### {heading}", "", intro, ""]
            for verdict in chosen:
                lines += _citation_block(verdict, state, resolved)
    unverified = sorted({v.key for v in verdicts if v.verdict == "cannot_verify"})
    if unverified:
        keys = ", ".join(f"`{key}`" for key in unverified)
        lines += [
            "",
            "### Could not be verified",
            "",
            f"The text available did not settle these citations either way: {keys}. This is "
            "common when only an abstract could be read. It is not a finding against them.",
        ]
    return lines


def _related_section(state: AuditState) -> list[str]:
    contributions = state.get("contributions", [])
    if not contributions:
        return []
    assessed = {(a.contribution_id, a.work_ref): a for a in state.get("assessments", [])}
    lines = ["## Related work and novelty", "", "Overlap is judged from abstracts, so treat it as a lead to follow up."]
    comparison: list[list[str]] = []
    for contribution in contributions:
        lines += ["", f"### Contribution {contribution.id}", "", contribution.statement]
        candidates = [c for c in state.get("candidates", []) if c.contribution_id == contribution.id]
        paired = [(assessed.get((contribution.id, c.work.ref)), c) for c in candidates]
        rated = [(a, c) for a, c in paired if a and a.overlap != "low"]
        rated.sort(key=lambda pair: (_OVERLAP_ORDER[pair[0].overlap], pair[1].already_cited))
        if not rated:
            if not candidates:
                outcome = "The search found no close prior work for this contribution."
            elif any(a is None for a, _ in paired):
                outcome = "Candidate papers were found, but their overlap could not be assessed."
            else:
                outcome = "The papers found share only a topic with this contribution."
            lines += ["", outcome]
            continue
        rows = []
        for assessment, candidate in rated:
            rows.append(
                [
                    _work_link(candidate.work),
                    assessment.overlap.capitalize(),
                    "Yes" if candidate.already_cited else "**No**",
                    _cell(assessment.reason),
                ]
            )
            comparison.append(
                [
                    _work_link(candidate.work),
                    _cell(assessment.approach),
                    _cell(assessment.data),
                    _cell(assessment.result),
                ]
            )
        lines += ["", *_table(["Prior work", "Overlap", "Already cited", "Why"], rows)]
    if comparison:
        lines += [
            "",
            "### Comparison of the related work",
            "",
            "Taken from each paper's abstract. \"Not stated\" means the abstract does not say.",
            "",
            *_table(["Paper", "Approach", "Data or setting", "Main result"], comparison),
        ]
    return lines


def _priorities(state: AuditState) -> list[str]:
    """The findings a reader should act on first, most serious first."""
    resolutions = state.get("resolutions", [])
    verdicts = state.get("verdicts", [])
    items = []
    retracted = [r.key for r in resolutions if r.work and r.work.is_retracted]
    if retracted:
        items.append(f"{len(retracted)} cited work(s) have been retracted: " + ", ".join(f"`{k}`" for k in retracted) + ".")
    mismatched = [r.key for r in resolutions if r.status == "mismatch"]
    if mismatched:
        items.append(f"{len(mismatched)} reference(s) carry a DOI or arXiv id that belongs to a different work: " + ", ".join(f"`{k}`" for k in mismatched) + ".")
    unsupported = sum(1 for v in verdicts if v.verdict == "not_supported")
    if unsupported:
        items.append(f"{unsupported} citation(s) are not supported by the work they cite.")
    assessed = {(a.contribution_id, a.work_ref): a for a in state.get("assessments", [])}
    uncited_high = [
        c for c in state.get("candidates", [])
        if not c.already_cited and (a := assessed.get((c.contribution_id, c.work.ref))) and a.overlap == "high"
    ]  # fmt: skip
    if uncited_high:
        items.append(f"{len(uncited_high)} uncited paper(s) appear to make a contribution the paper claims as its own.")
    not_found = sum(1 for r in resolutions if r.status == "not_found")
    if not_found:
        items.append(f"{not_found} reference(s) could not be found in any index.")
    partial = sum(1 for v in verdicts if v.verdict == "partially_supported")
    if partial:
        items.append(f"{partial} citation(s) are only partly supported by the work they cite.")
    if state["paper"].missing_keys:
        items.append(f"{len(state['paper'].missing_keys)} citation key(s) have no entry in the bibliography.")
    return items


def render_report(state: AuditState) -> str:
    paper = state["paper"]
    lines = [f"# Paper audit: {paper.title or 'untitled paper'}", ""]
    lines += [
        f"The paper has {len(paper.references)} references and {len(paper.sentences)} "
        "sentences that cite at least one of them.",
        "",
        "## What to look at first",
        "",
    ]
    priorities = _priorities(state)
    lines += [f"{n}. {item}" for n, item in enumerate(priorities, start=1)] or ["Nothing was flagged."]
    for section in (_reference_section(state), _citation_section(state), _related_section(state)):
        if section:
            lines += ["", *section]
    if state.get("notes"):
        lines += ["", "## Problems during the audit", ""]
        lines += [f"- {' '.join(note.split())}" for note in state["notes"]]
    lines += [
        "",
        "## How to read this audit",
        "",
        "- A verdict is a lead for you to check, not a ruling. The auditor can be wrong.",
        "- A citation checked against an abstract only is checked against much less than the "
        "paper says. Many of those end as \"could not be verified\".",
        "- Every \"supported\" verdict quotes the cited work, and the quotation was confirmed to "
        "be in the text that was read. A verdict whose quotation could not be found was "
        "downgraded to \"could not be verified\".",
        "- The scholarly indexes have gaps and errors of their own, so a clean result does not "
        "prove the reference list is right.",
    ]
    return "\n".join(lines) + "\n"
