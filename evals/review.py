"""Measures how often the judge agrees with a person.

    python -m evals.review --variant baseline --n 24    # write a labelling sheet
    python -m evals.review --variant baseline --score   # compare your labels with the judge's

The sheet shows sentences and the source text they cite, without the judge's
verdict, so the judge cannot sway you. You write full, partial or none after
each "Your verdict:" line. The judge's verdicts are kept in review_key.json and
compared when you run --score.
"""

import argparse
import json
import random
import re
from pathlib import Path

from evals import DEFAULT_FLOW

VERDICTS = ("full", "partial", "none")


def _fenced(text: str) -> str:
    """Wrap text in a code fence longer than any run of backticks inside it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(4, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def collect(variant_dir: Path) -> list[dict]:
    """Every cited sentence the judge graded, with the source text it was shown."""
    items = []
    for path in sorted((variant_dir / "judge").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for sentence in data["sentences"]:
            sources = [data["sources"][str(n)] | {"number": n} for n in sentence["citations"]
                       if str(n) in data["sources"]]
            if "support" in sentence and sources:
                items.append(
                    {
                        "case": data["prompt_id"],
                        "rep": data["rep"],
                        "sentence": sentence["text"],
                        "sources": sources,
                        "judge": sentence["support"],
                    }
                )
    return items


def sample(items: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Pick n items, drawing evenly from each verdict so rare ones are represented."""
    rng = random.Random(seed)
    pools = {v: [i for i in items if i["judge"] == v] for v in VERDICTS}
    for pool in pools.values():
        rng.shuffle(pool)
    picked = []
    while len(picked) < n and any(pools.values()):
        for verdict in VERDICTS:
            if pools[verdict] and len(picked) < n:
                picked.append(pools[verdict].pop())
    rng.shuffle(picked)  # so position gives no hint of the judge's verdict
    return picked


def write_sheet(variant_dir: Path, n: int) -> Path:
    picked = sample(collect(variant_dir), n)
    if not picked:
        raise SystemExit(f"No judged sentences under {variant_dir / 'judge'}. Run the eval first.")
    lines = [
        "# Judge review sheet",
        "",
        "For each item, read the sentence and the source text. After `Your verdict:` write one of:",
        "",
        "- `full`: the source text states every factual claim in the sentence.",
        "- `partial`: it supports some of the claims, but at least one is missing or different.",
        "- `none`: it supports none of the claims, or contradicts them.",
        "",
        "Judge only against the source text shown, not against what you know to be true.",
        "Then run `python -m evals.review --score`.",
    ]
    for number, item in enumerate(picked, start=1):
        lines += ["", f"## Item {number}", "", f"Case {item['case']}, repetition {item['rep']}.",
                  "", "Sentence:", "", _fenced(item["sentence"])]
        for source in item["sources"]:
            lines += ["", f"Source [{source['number']}]: {source['title']} ({source['url']})", "",
                      _fenced(source["content"])]
        lines += ["", "Your verdict:"]
    sheet = variant_dir / "review.md"
    sheet.write_text("\n".join(lines) + "\n", encoding="utf-8")
    key = {str(number): item["judge"] for number, item in enumerate(picked, start=1)}
    (variant_dir / "review_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    return sheet


def score_sheet(variant_dir: Path) -> str:
    key = json.loads((variant_dir / "review_key.json").read_text(encoding="utf-8"))
    text = (variant_dir / "review.md").read_text(encoding="utf-8")
    labels = {}
    for block in re.split(r"^## Item ", text, flags=re.MULTILINE)[1:]:
        number = block.split("\n", 1)[0].strip()
        # The last match is the answer line; the sentence or source may contain the phrase too.
        answers = re.findall(r"^Your verdict:[ \t]*(\S*)[ \t]*$", block, flags=re.MULTILINE)
        if answers and answers[-1].lower() in VERDICTS:
            labels[number] = answers[-1].lower()
    if not labels:
        raise SystemExit("No verdicts filled in yet. Write full, partial or none after each 'Your verdict:'.")
    agree = sum(1 for number, label in labels.items() if key.get(number) == label)
    lines = [
        f"Agreement: {agree} of {len(labels)} labelled items ({agree / len(labels):.0%}).",
        f"{len(key) - len(labels)} items not labelled.",
    ]
    lines += [
        f"  Item {number}: you said {label}, the judge said {key.get(number)}"
        for number, label in sorted(labels.items(), key=lambda kv: int(kv[0]))
        if key.get(number) != label
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the judge's verdicts with your own.")
    parser.add_argument("--flow", type=Path, default=DEFAULT_FLOW)
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--n", type=int, default=24, help="number of items on the sheet")
    parser.add_argument("--score", action="store_true", help="score a filled-in sheet")
    args = parser.parse_args(argv)
    variant_dir = args.flow / args.variant
    if args.score:
        print(score_sheet(variant_dir))
    else:
        print(f"Wrote {write_sheet(variant_dir, args.n)}. Fill it in, then run with --score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
