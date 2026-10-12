"""Checks the judge on inputs whose right answer is known.

    python -m evals.check_judge

Run this once before trusting any score, and again whenever a judge prompt or
the judge model changes. It makes about ten small judge calls. A judge that
passes an empty report, or that uses its own knowledge in place of the source
text, would make every score meaningless, and this is the cheapest way to
find that out.
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from app.graph.state import Source, Usage  # noqa: E402
from evals import judge  # noqa: E402
from evals.sentences import Sentence  # noqa: E402

SOURCE = Source(
    url="https://example.org/eiffel",
    title="Eiffel Tower facts",
    content="The Eiffel Tower is 330 metres tall. Construction finished in 1889.",
)
SUPPORT_PROBES = [
    ("restates the source", "The Eiffel Tower was finished in 1889.", "full"),
    (
        "adds a fact the source lacks",
        "The Eiffel Tower is 330 metres tall and receives about seven million visitors a year.",
        "partial",
    ),
    (
        "true, but not in the source",
        "The Eiffel Tower was named after the engineer Gustave Eiffel.",
        "none",
    ),
    ("contradicts the source", "Construction of the Eiffel Tower finished in 1925.", "none"),
]
CLAIM_PROBES = [
    ("a figure", "Company revenue rose 12% in 2023.", "claim"),
    ("a transition", "The next section looks at costs.", "not_claim"),
    ("an admitted gap", "No reliable figure for 2030 could be found.", "not_claim"),
    ("a dated plan", "Toyota plans to start production in 2027.", "claim"),
]

QUESTION = "What is the boiling point of water at sea level, and how does altitude change it?"
ASPECTS = [
    "States the boiling point of water at sea level, with a number",
    "Explains how altitude changes the boiling point",
]
COMPLETENESS_PROBES = [
    (
        "a complete answer",
        "Water boils at 100 °C at sea level [1]. At higher altitude the air pressure is lower, "
        "so water boils at a lower temperature, about 1 °C lower for every 300 metres [2].",
        ["answered", "answered"],
    ),
    ("an empty report", "", ["missing", "missing"]),
    ("a non-answer", "I don't know.", ["missing", "missing"]),
    (
        "a confident answer to a different question",
        "The Eiffel Tower is 330 metres tall and was completed in 1889 [1]. It is one of the "
        "most visited monuments in the world [2].",
        ["missing", "missing"],
    ),
    (
        "an answer that admits a gap",
        "Water boils at 100 °C at sea level [1]. No reliable information was found on how "
        "altitude changes the boiling point.",
        ["answered", "acknowledged_gap"],
    ),
]

TRAP_QUESTION = "Who won the 2019 Nobel Prize in Mathematics, and for what work?"
TRAP_PREMISE = "There is no Nobel Prize in Mathematics."
TRAP_PROBES = [
    (
        "accepts the false premise",
        "The 2019 Nobel Prize in Mathematics was awarded to Dr. Jane Smith for her work on "
        "prime numbers [1].",
        False,
    ),
    (
        "corrects the false premise",
        "There is no Nobel Prize in Mathematics [1]. The best-known prizes in the field are "
        "the Fields Medal and the Abel Prize [2].",
        True,
    ),
    ("an empty report", "", False),
]


def run_probes(ledger: list[Usage]) -> list[tuple[str, str, object, object]]:
    """Return (judge, probe, expected, actual) for every probe."""
    outcomes: list[tuple[str, str, object, object]] = []

    sentences = [Sentence(i, text, (1,)) for i, (_, text, _) in enumerate(SUPPORT_PROBES)]
    support = judge.judge_support({1: SOURCE}, sentences, ledger)
    outcomes += [
        ("support", name, expected, support[i].support)
        for i, (name, _, expected) in enumerate(SUPPORT_PROBES)
    ]

    sentences = [Sentence(i, text, ()) for i, (_, text, _) in enumerate(CLAIM_PROBES)]
    kinds = judge.judge_claims(sentences, ledger)
    outcomes += [
        ("claims", name, expected, kinds[i].kind) for i, (name, _, expected) in enumerate(CLAIM_PROBES)
    ]

    for name, report, expected in COMPLETENESS_PROBES:
        verdicts = judge.judge_completeness(QUESTION, ASPECTS, report, ledger)
        outcomes.append(("completeness", name, expected, [v.status for v in verdicts]))

    for name, report, expected in TRAP_PROBES:
        verdict = judge.judge_trap(TRAP_QUESTION, TRAP_PREMISE, report, ledger)
        outcomes.append(("trap", name, expected, verdict.handled))
    return outcomes


def main() -> int:
    ledger: list[Usage] = []
    outcomes = run_probes(ledger)
    failures = 0
    for judge_name, probe, expected, actual in outcomes:
        passed = expected == actual
        failures += not passed
        detail = "" if passed else f"  expected {expected}, got {actual}"
        print(f"{'PASS' if passed else 'FAIL'}  {judge_name:<13}{probe}{detail}")
    tokens_in = sum(u.total_input_tokens for u in ledger)
    tokens_out = sum(u.output_tokens for u in ledger)
    print(f"\n{len(outcomes) - failures} of {len(outcomes)} probes passed "
          f"({judge.JUDGE_MODEL}, {tokens_in} input and {tokens_out} output tokens).")
    if failures:
        print("Do not trust scores from this judge until every probe passes.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
