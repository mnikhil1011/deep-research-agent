"""Planner: breaks the user's query into researchable sub-questions."""

from app import config, llm
from app.graph.state import ResearchPlan, ResearchState

SYSTEM = f"""You are the planning agent in a research pipeline. Downstream, a \
researcher runs one web search per sub-question and a writer turns the evidence \
into a report.

Break the user's query into at most {config.MAX_SUB_QUESTIONS} sub-questions. \
Each one must be answerable by a web search on its own, so name the subject \
explicitly instead of referring back to the query. Together they should cover \
the query without overlapping. A narrow query may need only two or three."""


def plan(state: ResearchState) -> dict:
    result, usage = llm.structured(
        agent="planner",
        model=config.PLANNER_MODEL,
        system=SYSTEM,
        prompt=state["query"],
        schema=ResearchPlan,
    )
    result.sub_questions = result.sub_questions[: config.MAX_SUB_QUESTIONS]
    return {"plan": result, "usage": [usage]}
