"""The benchmark cases: loading, validation, and a readable Markdown view.

A case is either answerable (it has `aspects`, the things a complete answer
must address) or a trap (it has `false_premise`, the reason the question
cannot be answered as asked). Only `question` is ever shown to the system
under test; the other fields are for the graders.

    python -m evals.cases        # regenerate evals/CASES.md from cases.json
"""

import json
import re
from pathlib import Path

from pydantic import BaseModel, model_validator

CASES_PATH = Path(__file__).parent / "cases.json"
CASES_MD_PATH = Path(__file__).parent / "CASES.md"

# Ids become file names and report links, so they are kept to a safe alphabet.
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class Case(BaseModel):
    id: str
    tags: list[str]
    question: str
    aspects: list[str] = []
    false_premise: str = ""

    @property
    def is_trap(self) -> bool:
        return bool(self.false_premise)

    @model_validator(mode="after")
    def _check(self):
        if not SAFE_ID.match(self.id):
            raise ValueError(f"case id {self.id!r} may only contain letters, digits, _ and -")
        if not self.tags:
            raise ValueError(f"{self.id}: needs at least one tag")
        if bool(self.aspects) == bool(self.false_premise):
            raise ValueError(f"{self.id}: needs exactly one of 'aspects' or 'false_premise'")
        if self.is_trap != (self.tags[0] == "trap"):
            raise ValueError(f"{self.id}: the first tag must be 'trap' for trap cases only")
        return self


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    cases = [Case.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]
    # Windows and macOS file systems ignore case, so ids must differ beyond it.
    seen: dict[str, str] = {}
    for case in cases:
        key = case.id.lower()
        if key in seen:
            raise ValueError(f"duplicate case id: {case.id!r} and {seen[key]!r}")
        seen[key] = case.id
    questions = [c.question.strip().lower() for c in cases]
    if len(set(questions)) != len(questions):
        raise ValueError("two cases have the same question")
    return cases


def render_markdown(cases: list[Case]) -> str:
    answerable = [c for c in cases if not c.is_trap]
    traps = [c for c in cases if c.is_trap]
    lines = [
        "# Benchmark cases",
        "",
        "Generated from `cases.json` by `python -m evals.cases`. Do not edit by hand.",
        "",
        f"{len(cases)} cases: {len(answerable)} answerable and {len(traps)} traps. The system "
        "under test sees only the question. Aspects and false premises are for the graders.",
        "",
        "| Id | Tags | Question |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {c.id} | {', '.join(c.tags)} | {c.question} |" for c in cases]
    lines += [
        "",
        "## Answerable cases",
        "",
        "A complete report addresses every aspect listed. The aspects say what must be "
        "covered, not what the answer is, so they do not go stale when the facts change.",
    ]
    for case in answerable:
        lines += ["", f"### {case.id}", "", case.question, ""]
        lines += [f"- {aspect}" for aspect in case.aspects]
    lines += [
        "",
        "## Trap cases",
        "",
        "Each question rests on a false premise. A good report says so, or says that it "
        "could not verify the premise, and does not invent details.",
    ]
    for case in traps:
        lines += ["", f"### {case.id}", "", case.question, "", f"False premise: {case.false_premise}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    CASES_MD_PATH.write_text(render_markdown(load_cases()), encoding="utf-8")
    print(f"Wrote {CASES_MD_PATH}")
