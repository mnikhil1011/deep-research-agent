"""Stand-ins for the model and the web, shared by the test modules.

The fake researcher model follows a fixed script per task: search, read the
first result, then submit. Task names switch on the special behaviours.
"""

from types import SimpleNamespace

from app.graph.state import Usage


def usage(agent: str) -> Usage:
    return Usage(agent=agent, model="stub", input_tokens=10, output_tokens=5)


def tool_calls(*calls: tuple[str, dict]) -> SimpleNamespace:
    blocks = [
        SimpleNamespace(type="tool_use", id=f"call_{i}", name=name, input=tool_input)
        for i, (name, tool_input) in enumerate(calls)
    ]
    return SimpleNamespace(content=blocks)


def fake_tool_turn(*, agent, messages, **_):
    task = messages[0]["content"].split("\n")[0].removeprefix("Task: ")
    turn = sum(1 for m in messages if m["role"] == "assistant")

    if "never submits" in task:
        return tool_calls(("web_search", {"query": task})), usage(agent)
    if "search fails" in task:
        if turn == 0:
            return tool_calls(("web_search", {"query": "fails"})), usage(agent)
        saw_error = messages[2]["content"][0].get("is_error", False)
        if turn == 1 and saw_error:
            return tool_calls(("web_search", {"query": "recovered"})), usage(agent)
        findings = [{"claim": "claim after recovering", "source_ids": [2]}] if saw_error else []
        return tool_calls(("submit_findings", {"findings": findings})), usage(agent)

    if turn == 0:
        return tool_calls(("web_search", {"query": task})), usage(agent)
    if turn == 1:
        return tool_calls(("read_page", {"source_id": 1})), usage(agent)
    findings = [
        # 99 is an id the researcher was never shown.
        {"claim": f"claim about {task}", "source_ids": [1, 2, 99]},
        {"claim": "ungrounded claim", "source_ids": [99]},
    ]
    return tool_calls(("submit_findings", {"findings": findings})), usage(agent)


def fake_web_search(query, max_results):
    if query == "fails":
        raise RuntimeError("search is down")
    # shared.example is returned for every query.
    return [
        {"url": "https://shared.example", "title": "Shared", "content": "snippet"},
        {"url": f"https://{query.lower()}.example", "title": query, "content": "snippet"},
    ]
