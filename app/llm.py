"""Thin wrapper over the Anthropic SDK shared by every agent.

Each call returns the result together with a Usage record, so the graph can
account for tokens per agent.
"""

from functools import lru_cache
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from app.graph.state import Usage

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """The model returned something the pipeline cannot use.

    `kind` is "refusal", "max_tokens" or "bad_output".
    """

    def __init__(self, message: str, kind: str):
        super().__init__(message)
        self.kind = kind


@lru_cache
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def _model_kwargs(model: str) -> dict:
    # Haiku 4.5 rejects `effort` and is not covered by refusal fallbacks.
    if model.startswith("claude-haiku"):
        return {}
    return {
        "output_config": {"effort": "medium"},
        # If a safety classifier declines the request, the API re-runs it on
        # Anthropic's recommended fallback model inside the same call.
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
    }


def _check(response, agent: str) -> Usage:
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise LLMError(f"{agent}: model refused the request (category={category})", "refusal")
    if response.stop_reason == "max_tokens":
        raise LLMError(f"{agent}: output was cut off at max_tokens", "max_tokens")
    usage = response.usage
    return Usage(
        agent=agent,
        model=response.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        # Cached input is billed at different rates, so it is kept separate.
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
    )


def structured(
    *, agent: str, model: str, system: str, prompt: str, schema: type[T]
) -> tuple[T, Usage]:
    """Ask for a response validated against a Pydantic schema."""
    response = _client().beta.messages.parse(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,
        **_model_kwargs(model),
    )
    usage = _check(response, agent)
    if response.parsed_output is None:
        raise LLMError(f"{agent}: response did not match {schema.__name__}", "bad_output")
    return response.parsed_output, usage


def text(*, agent: str, model: str, system: str, prompt: str) -> tuple[str, Usage]:
    """Ask for free-form text. Streams, since reports can be long."""
    with _client().beta.messages.stream(
        model=model,
        max_tokens=32000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        **_model_kwargs(model),
    ) as stream:
        response = stream.get_final_message()
    usage = _check(response, agent)
    return "".join(b.text for b in response.content if b.type == "text"), usage


def tool_turn(*, agent: str, model: str, system: str, messages: list, tools: list):
    """Run one turn of a tool-using conversation. The caller owns the loop.

    Returns the raw response (its content may hold tool_use blocks) and usage.
    """
    response = _client().beta.messages.create(
        model=model,
        max_tokens=16000,
        system=system,
        messages=messages,
        tools=tools,
        # The conversation is resent every turn; cache the part already seen.
        cache_control={"type": "ephemeral"},
        **_model_kwargs(model),
    )
    return response, _check(response, agent)
