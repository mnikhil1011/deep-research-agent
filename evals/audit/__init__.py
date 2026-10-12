"""Benchmarks for the paper audit. See evals/README.md."""

import json
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "runs"


def benchmark_papers(split: str) -> list[str]:
    """The arXiv ids in one split of the benchmark: "held_out" or "tuning"."""
    return json.loads((HERE / "papers.json").read_text(encoding="utf-8"))[split]


def rate(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.2f} ({numerator}/{denominator})" if denominator else "n/a (0)"
