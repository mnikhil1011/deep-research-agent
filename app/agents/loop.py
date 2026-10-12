"""The tool loop that every tool-using agent shares.

The model drives it: each step it either asks for tools, which are run and
answered, or calls its "submit" tool, which ends the loop. The loop enforces the
step limit, turns tool failures into messages the model can react to, and keeps
a log of every call.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from app import llm
from app.graph.state import ToolCall, Usage


@dataclass
class LoopResult:
    submission: dict | None  # the arguments of the submit call, or None if it never came
    steps: int
    usage: list[Usage] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    note: str = ""  # why there is no submission, when there is none


def finish_message(submit_tool: str) -> str:
    return f"Finish now: call {submit_tool} with what you have found."


def run_tool_loop(
    *,
    agent: str,
    model: str,
    system: str,
    prompt: str,
    tools: list[dict],
    run_tool: Callable[[str, dict], str],
    submit_tool: str,
    max_steps: int,
    task: str,
) -> LoopResult:
    """Run one agent until it submits, runs out of steps, or the model call fails.

    `run_tool(name, arguments)` returns the tool's output as text, or raises.
    Anything it raises goes back to the model as an error result.
    """
    result = LoopResult(submission=None, steps=0)
    messages: list[dict] = [{"role": "user", "content": prompt}]
    try:
        for step in range(1, max_steps + 1):
            result.steps = step
            response, usage = llm.tool_turn(
                agent=agent, model=model, system=system, messages=messages, tools=tools
            )
            result.usage.append(usage)
            messages.append({"role": "assistant", "content": response.content})

            calls = [block for block in response.content if block.type == "tool_use"]
            submit = next((call for call in calls if call.name == submit_tool), None)
            if submit:
                result.tool_calls.append(
                    ToolCall(task=task, step=step, name=submit.name, input=submit.input, output="")
                )
                result.submission = submit.input
                return result

            reply = []
            for call in calls:
                try:
                    output, is_error = run_tool(call.name, call.input), False
                except Exception as error:  # noqa: BLE001 - the model gets to react to it
                    output, is_error = f"Error: {error}", True
                block = {"type": "tool_result", "tool_use_id": call.id, "content": output}
                reply.append({**block, "is_error": True} if is_error else block)
                result.tool_calls.append(
                    ToolCall(
                        task=task, step=step, name=call.name, input=call.input,
                        output=output, is_error=is_error,
                    )  # fmt: skip
                )
            if not calls or step == max_steps - 1:
                reply.append({"type": "text", "text": finish_message(submit_tool)})
            messages.append({"role": "user", "content": reply})
        result.note = f"ran out of steps without calling {submit_tool}"
    except llm.LLMError as error:
        # One failed agent should not sink the run; the caller sees the note.
        result.note = str(error)
    return result
