"""Assembles the research graph: planner -> researcher -> writer."""

from langgraph.graph import END, START, StateGraph

from app.agents.planner import plan
from app.agents.researcher import research
from app.agents.writer import write
from app.graph.state import ResearchState


def build_graph(checkpointer=None):
    graph = StateGraph(ResearchState)
    graph.add_node("planner", plan)
    graph.add_node("researcher", research)
    graph.add_node("writer", write)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", END)
    return graph.compile(checkpointer=checkpointer)
