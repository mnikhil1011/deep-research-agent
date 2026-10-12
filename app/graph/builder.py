"""Assembles the research graph.

    START -> orchestrator -> researcher (one per task, in parallel) -> orchestrator ...
                          -> writer -> END

The orchestrator's decision picks the branch, so the path through the graph is
chosen by the model at run time, within the limits in app/config.py.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agents.orchestrator import orchestrate
from app.agents.researcher import research
from app.agents.writer import write
from app.graph.state import ResearchState


def dispatch(state: ResearchState) -> list[Send] | str:
    """Route on the orchestrator's decision: fan out researchers, or go to the writer."""
    decision = state["decision"]
    if decision.action == "write":
        return "writer"
    return [Send("researcher", {"query": state["query"], "task": task}) for task in decision.tasks]


def build_graph(checkpointer=None):
    graph = StateGraph(ResearchState)
    graph.add_node("orchestrator", orchestrate)
    graph.add_node("researcher", research)
    graph.add_node("writer", write)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", dispatch, ["researcher", "writer"])
    graph.add_edge("researcher", "orchestrator")
    graph.add_edge("writer", END)
    return graph.compile(checkpointer=checkpointer)
