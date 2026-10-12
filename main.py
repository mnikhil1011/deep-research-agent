"""CLI entry point: python main.py "your research question" """

import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # before app imports, which read settings from the environment

from app.graph.builder import build_graph  # noqa: E402

REPORTS_DIR = Path("reports")


def _describe(node: str, update: dict) -> str:
    if node == "orchestrator":
        decision = update["decision"]
        tasks = "".join(f"\n    - {task}" for task in decision.tasks)
        return f"{decision.action}: {decision.reasoning}{tasks}"
    if node == "researcher":
        report = update["task_reports"][0]
        note = f" ({report.note})" if report.note else ""
        return f"{report.findings} findings in {report.steps} steps{note}: {report.task}"
    return "report drafted"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: python main.py "your research question"')
    query = " ".join(sys.argv[1:])
    # Model-written text can hold characters the Windows console cannot print.
    sys.stdout.reconfigure(errors="replace")

    state: dict = {}
    stream = build_graph().stream({"query": query}, stream_mode=["updates", "values"])
    for mode, chunk in stream:
        if mode == "values":
            state = chunk  # the full state after each step; the last one is final
            continue
        for node, update in chunk.items():
            print(f"[{node}] {_describe(node, update)}")

    REPORTS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:60]
    path = REPORTS_DIR / f"{slug}.md"
    path.write_text(state["report"], encoding="utf-8")

    print("\nTokens by agent (input / output):")
    for agent in dict.fromkeys(u.agent for u in state["usage"]):
        calls = [u for u in state["usage"] if u.agent == agent]
        print(
            f"  {agent:<13}{sum(u.total_input_tokens for u in calls):>8} /"
            f"{sum(u.output_tokens for u in calls):>7}  ({len(calls)} calls, {calls[0].model})"
        )
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()
