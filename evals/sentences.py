"""Splits a report into gradeable sentences and reads the citations on each.

This is deliberately plain code, not a model call: the unit being graded has to
be the same on every run, or scores would move for reasons unrelated to the
system under test.
"""

import re
from dataclasses import dataclass

from app.text import split_sentences

CITATION = re.compile(r"\[(\d+)\]")
SOURCES_HEADING = "\n## Sources"

_LIST_MARKER = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_TABLE_RULE = re.compile(r"^\|?[\s:|-]*-[\s:|-]*\|?$")


@dataclass(frozen=True)
class Sentence:
    index: int
    text: str  # citation markers removed
    citations: tuple[int, ...]  # source numbers, in order of first appearance


def report_body(report: str) -> str:
    """The report without the Sources list that the writer's code appends."""
    return report.rsplit(SOURCES_HEADING, 1)[0] if SOURCES_HEADING in report else report


def _units(body: str) -> list[str]:
    units: list[str] = []
    in_code = False
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line or line.startswith("#"):
            continue
        if _TABLE_RULE.fullmatch(line) or set(line) <= set("-*_ "):
            continue  # a table's header rule, or a horizontal rule
        if line.startswith("|"):
            units.append(" | ".join(cell.strip() for cell in line.strip("|").split("|")))
            continue  # one table row is one unit
        line = _LIST_MARKER.sub("", line.lstrip("> ").strip())
        units += split_sentences(line)
    return units


def split_report(report: str) -> list[Sentence]:
    """Return the report's sentences in order, skipping headings and the Sources list."""
    sentences = []
    for unit in _units(report_body(report)):
        citations = tuple(dict.fromkeys(int(n) for n in CITATION.findall(unit)))
        text = re.sub(r"\s+", " ", CITATION.sub("", unit)).strip()
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        if not any(ch.isalnum() for ch in text):
            continue
        sentences.append(Sentence(index=len(sentences), text=text, citations=citations))
    return sentences


def citation_markers(report: str) -> list[int]:
    """Every citation marker in the body, repeats included."""
    return [int(n) for n in CITATION.findall(report_body(report))]
