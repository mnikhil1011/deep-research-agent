"""Orchestrator: decides, round by round, whether to research more or to write."""

from app import config, llm
from app.graph.state import Decision, ResearchState

SYSTEM = """You are the orchestrator of a research pipeline. You do not research \
or write yourself. You decide what happens next, and other agents carry it out:
- Researcher agents each take one task. They search the web, read pages, and \
report findings with sources. Tasks in the same round run in parallel, and each \
researcher sees only its own task and the user's question.
- A writer agent turns all the findings so far into the final report.

Each time you are called, you see the user's question and everything the \
researchers have found so far. Choose one action:
- "research": dispatch a new round of tasks. On the first call, break the \
question into tasks that together cover it without overlapping. On later calls, \
send only targeted follow-ups for what is still missing: a part of the question \
with no findings, a key figure that is absent, or a conflict between findings \
that more evidence could settle. Do not repeat a task that has already run. If \
one came back empty, rephrase it or approach it from another angle.
- "write": hand over to the writer. Choose this when the findings answer the \
question well enough that another round would add little.

Write each task as a self-contained instruction that names its subject \
explicitly, because the researcher does not see the other tasks. Every round \
costs time and money, so research further only when the gap matters to the answer."""


def _format_progress(state: ResearchState) -> str:
    reports = state.get("task_reports", [])
    if not reports:
        return "No research has been done yet."
    sections = []
    for report in reports:
        claims = [f"- {f.claim}" for f in state.get("findings", []) if f.task == report.task]
        body = "\n".join(claims) or f"(no findings{': ' + report.note if report.note else ''})"
        sections.append(f"## {report.task}\n{body}")
    return "Findings so far, by task:\n\n" + "\n\n".join(sections)


def orchestrate(state: ResearchState) -> dict:
    rounds = state.get("round", 0)
    if rounds >= config.MAX_RESEARCH_ROUNDS:
        stop = Decision(reasoning="The research round limit was reached.", action="write", tasks=[])
        return {"decision": stop}

    decision, usage = llm.structured(
        agent="orchestrator",
        model=config.ORCHESTRATOR_MODEL,
        system=SYSTEM,
        prompt=(
            f"Question: {state['query']}\n\n"
            f"Research rounds used: {rounds} of {config.MAX_RESEARCH_ROUNDS}. "
            f"You may dispatch up to {config.MAX_TASKS_PER_ROUND} tasks per round.\n\n"
            f"{_format_progress(state)}"
        ),
        schema=Decision,
    )

    # The model proposes; these rules keep the proposal within bounds.
    tasks: list[str] = []
    if decision.action == "research":
        done = {r.task for r in state.get("task_reports", [])}
        fresh = [t for t in dict.fromkeys(decision.tasks) if t not in done]
        tasks = fresh[: config.MAX_TASKS_PER_ROUND]
    if rounds == 0 and not tasks:
        # Nothing has been researched yet, so there is nothing to write from.
        tasks = [state["query"]]

    decision = Decision(
        reasoning=decision.reasoning, action="research" if tasks else "write", tasks=tasks
    )
    return {"decision": decision, "round": rounds + 1 if tasks else rounds, "usage": [usage]}
