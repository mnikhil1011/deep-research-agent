"""Assembles the paper-audit graph.

    read the paper -> check references -> plan --> auditor (one per cited work)
        --> contributions --> scout (one per contribution) --> assess (one per
        contribution) --> report

Unlike the research engine, the order of the stages is fixed: an audit always
needs the same checks. The judgment is inside the stages. Auditors and scouts
are agents that decide for themselves what to search for and when to stop.

`mode` shortens the path: "references" stops after the reference check and
makes no model calls; "citations" stops after the citation audit.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app import config
from app.audit.analyst import assess_overlap, extract_contributions
from app.audit.auditor import audit_citations
from app.audit.latex import load_paper
from app.audit.models import AuditState, CitedSentence
from app.audit.report import render_report
from app.audit.resolver import resolve_all
from app.audit.scout import scout_related_work


def read_paper(state: AuditState) -> dict:
    return {"paper": load_paper(state["source"])}


def check_references(state: AuditState) -> dict:
    return {"resolutions": resolve_all(state["paper"])}


def plan_citation_audit(state: AuditState) -> dict:
    """Group the citing sentences by the work they cite, most-cited works first."""
    verified = {r.key: r.work for r in state["resolutions"] if r.status == "verified"}
    by_key: dict[str, list[CitedSentence]] = {}
    for sentence in state["paper"].sentences:
        for key in sentence.keys:
            if key in verified:
                by_key.setdefault(key, []).append(sentence)
    ranked = sorted(by_key.items(), key=lambda item: -len(item[1]))
    chosen = ranked[: config.MAX_REFERENCES_AUDITED]
    notes = []
    if len(ranked) > len(chosen):
        notes.append(
            f"{len(ranked) - len(chosen)} cited works were not audited, because the limit is "
            f"{config.MAX_REFERENCES_AUDITED}. The works cited most often were audited first."
        )
    plan = [{"key": key, "work": verified[key], "sentences": sentences} for key, sentences in chosen]
    return {"audit_plan": plan, "notes": notes}


def join(state: AuditState) -> dict:
    """A point where parallel branches meet. It changes nothing."""
    return {}


def write_report(state: AuditState) -> dict:
    return {"report": render_report(state)}


# --- Routing -----------------------------------------------------------------


def after_references(state: AuditState) -> str:
    return "report" if state.get("mode") == "references" else "plan_citation_audit"


def dispatch_auditors(state: AuditState) -> list[Send] | str:
    return [Send("auditor", task) for task in state["audit_plan"]] or "citations_done"


def after_citations(state: AuditState) -> str:
    return "report" if state.get("mode") == "citations" else "contributions"


def dispatch_scouts(state: AuditState) -> list[Send] | str:
    paper = state["paper"]
    shared = {
        "paper_title": paper.title,
        "paper_abstract": paper.abstract,
        "cited_works": [r.work for r in state["resolutions"] if r.work],
        "cited_titles": [reference.title for reference in paper.references.values()],
    }
    return [
        Send("scout", {"contribution": contribution, **shared})
        for contribution in state.get("contributions", [])
    ] or "report"


def dispatch_assessments(state: AuditState) -> list[Send] | str:
    sends = []
    for contribution in state.get("contributions", []):
        candidates = [c for c in state.get("candidates", []) if c.contribution_id == contribution.id]
        if candidates:
            sends.append(Send("assess", {"contribution": contribution, "candidates": candidates}))
    return sends or "report"


def build_audit_graph(checkpointer=None):
    graph = StateGraph(AuditState)
    graph.add_node("read_paper", read_paper)
    graph.add_node("check_references", check_references)
    graph.add_node("plan_citation_audit", plan_citation_audit)
    graph.add_node("auditor", audit_citations)
    graph.add_node("citations_done", join)
    graph.add_node("contributions", extract_contributions)
    graph.add_node("scout", scout_related_work)
    graph.add_node("scouting_done", join)
    graph.add_node("assess", assess_overlap)
    graph.add_node("report", write_report)

    graph.add_edge(START, "read_paper")
    graph.add_edge("read_paper", "check_references")
    graph.add_conditional_edges("check_references", after_references, ["plan_citation_audit", "report"])
    graph.add_conditional_edges("plan_citation_audit", dispatch_auditors, ["auditor", "citations_done"])
    graph.add_edge("auditor", "citations_done")
    graph.add_conditional_edges("citations_done", after_citations, ["contributions", "report"])
    graph.add_conditional_edges("contributions", dispatch_scouts, ["scout", "report"])
    graph.add_edge("scout", "scouting_done")
    graph.add_conditional_edges("scouting_done", dispatch_assessments, ["assess", "report"])
    graph.add_edge("assess", "report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)
