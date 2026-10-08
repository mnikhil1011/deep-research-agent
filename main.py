"""CLI entry point: python main.py "your research question" """

import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # before app imports, which read settings from the environment

from app.graph.builder import build_graph  # noqa: E402

REPORTS_DIR = Path("reports")


def _describe(node: str, update: dict) -> str:
    if node == "planner":
        questions = "\n".join(f"    - {q}" for q in update["plan"].sub_questions)
        return f"planned {len(update['plan'].sub_questions)} sub-questions\n{questions}"
    if node == "researcher":
        return f"{len(update['findings'])} findings from {len(update['sources'])} sources"
    return "report drafted"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: python main.py "your research question"')
    query = " ".join(sys.argv[1:])

    state: dict = {"usage": []}
    for step in build_graph().stream({"query": query}, stream_mode="updates"):
        for node, update in step.items():
            print(f"[{node}] {_describe(node, update)}")
            state["usage"] += update.pop("usage", [])
            state.update(update)

    REPORTS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:60]
    path = REPORTS_DIR / f"{slug}.md"
    path.write_text(state["report"], encoding="utf-8")

    print("\nTokens by agent (input / output):")
    for agent in dict.fromkeys(u.agent for u in state["usage"]):
        calls = [u for u in state["usage"] if u.agent == agent]
        print(
            f"  {agent:<11}{sum(u.input_tokens for u in calls):>8} /"
            f"{sum(u.output_tokens for u in calls):>7}  ({len(calls)} calls, {calls[0].model})"
        )
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()
