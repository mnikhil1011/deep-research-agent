"""Measures the citation auditor on citations with a known answer.

    python -m evals.audit.citations build       make the benchmark (no model, no key)
    python -m evals.audit.citations run         audit every item (model calls)
    python -m evals.audit.citations summarize   scores, from the saved results

Each benchmark sentence comes from a real paper and cites exactly one work. It
is tested three times, each time paired with a different work:

    original       the work the authors cited. Presumed to support the sentence.
    swapped_far    another work from the same reference list, chosen to be on a
                   different subject. It should not support the sentence.
    swapped_near   another work from the same list, chosen to be on the closest
                   subject. This is the hard case, and its label is the least
                   certain: a closely related work sometimes does support it.

The auditor should mark originals "supported" and swapped pairs "not_supported".
Only works found on arXiv are used, so every item has an abstract and full text.
"""

import argparse
import json
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()

from app import config, llm  # noqa: E402
from app.audit.auditor import audit_citations  # noqa: E402
from app.audit.latex import load_paper  # noqa: E402
from app.audit.models import CitedSentence, Paper  # noqa: E402
from app.audit.resolver import matches  # noqa: E402
from app.tools import scholar  # noqa: E402
from app.tools.scholar import ScholarError, Work  # noqa: E402
from evals.audit import HERE, RUNS, benchmark_papers, rate  # noqa: E402
from evals.judge import served_as  # noqa: E402

ITEMS_PATH = HERE / "citation_items.json"
RESULTS = RUNS / "citations"
SENTENCES_PER_PAPER = 10
MIN_WORDS = 12  # short sentences rarely make a claim specific enough to check
LABELS = ("original", "swapped_far", "swapped_near")
VERDICTS = ("supported", "partially_supported", "not_supported", "cannot_verify")
_lock = threading.Lock()


# --- Building the benchmark --------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]{4,}", text.lower())}


def _overlap(a: str, b: str) -> float:
    """Share of distinct words two texts have in common (Jaccard similarity)."""
    first, second = _tokens(a), _tokens(b)
    return len(first & second) / len(first | second) if first | second else 0.0


def _arxiv_works(paper: Paper, keys: list[str]) -> dict[str, Work]:
    """Find the given references on arXiv: by id in one request, then by title."""
    references = [paper.references[key] for key in keys if key in paper.references]
    by_id = scholar.arxiv_by_ids([r.arxiv_id for r in references if r.arxiv_id])
    found: dict[str, Work] = {}
    for reference in references:
        work = by_id.get(reference.arxiv_id) if reference.arxiv_id else None
        if work is None and reference.title:
            try:
                results = scholar.arxiv_search(scholar.arxiv_title_query(reference.title), 3)
            except ScholarError:
                results = []
            work = next((w for w in results if matches(reference, w)), None)
        if work and work.abstract and matches(reference, work):
            found[reference.key] = work
    return found


def _eligible(paper: Paper) -> list[CitedSentence]:
    return [s for s in paper.sentences if len(s.keys) == 1 and len(s.text.split()) >= MIN_WORDS]


def items_for_paper(arxiv_id: str, split: str) -> list[dict]:
    """Build the benchmark items for one paper."""
    paper = load_paper(f"arxiv:{arxiv_id}")
    works = _arxiv_works(paper, list(dict.fromkeys(s.keys[0] for s in _eligible(paper))))
    sentences = [s for s in _eligible(paper) if s.keys[0] in works]
    # One sentence per cited work, so that a single well-cited paper cannot dominate.
    chosen: dict[str, CitedSentence] = {}
    for sentence in sentences:
        chosen.setdefault(sentence.keys[0], sentence)
    items = []
    for sentence in list(chosen.values())[:SENTENCES_PER_PAPER]:
        true_key = sentence.keys[0]
        true_work = works[true_key]
        anchor = f"{sentence.text} {true_work.title} {true_work.abstract}"
        others = sorted(
            (key for key in works if key != true_key and works[key].ref != true_work.ref),
            key=lambda key: (_overlap(anchor, f"{works[key].title} {works[key].abstract}"), key),
        )
        if len(others) < 2:
            continue
        for label, key in (("original", true_key), ("swapped_far", others[0]), ("swapped_near", others[-1])):
            shown = sentence.model_copy(
                update={
                    "text": sentence.text.replace(f"[{true_key}]", f"[{key}]"),
                    "context": sentence.context,
                    "keys": [key],
                }
            )
            items.append(
                {
                    "id": f"{arxiv_id}-s{sentence.id}-{label}",
                    "paper": arxiv_id,
                    "split": split,
                    "label": label,
                    "key": key,
                    "original_key": true_key,
                    "sentence": shown.model_dump(),
                    "work": works[key].model_dump(),
                }
            )
    return items


def build() -> int:
    items = []
    for split in ("tuning", "held_out"):
        for arxiv_id in benchmark_papers(split):
            print(f"Building items from arXiv:{arxiv_id} ({split}) ...", file=sys.stderr)
            items += items_for_paper(arxiv_id, split)
    ITEMS_PATH.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    counts = Counter((item["split"], item["label"]) for item in items)
    print(f"Wrote {len(items)} items to {ITEMS_PATH}: {dict(counts)}")
    return 0


def load_items() -> list[dict]:
    if not ITEMS_PATH.exists():
        raise SystemExit(f"{ITEMS_PATH} does not exist. Run `python -m evals.audit.citations build` first.")
    return json.loads(ITEMS_PATH.read_text(encoding="utf-8"))


# --- Running the auditor -----------------------------------------------------


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _append(path, record: dict) -> None:
    with _lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def audit_item(item: dict) -> dict:
    """Run the real auditor on one item and return a results row. Raises on failure."""
    result = audit_citations(
        {
            "key": item["key"],
            "work": Work.model_validate(item["work"]),
            "sentences": [CitedSentence.model_validate(item["sentence"])],
        }
    )
    usage = result.get("usage", [])
    if result.get("notes") or not usage:
        # The auditor did not finish. That is a failed attempt, not a "cannot_verify".
        raise llm.LLMError("; ".join(result.get("notes", [])) or "no model call was made", "bad_output")
    for record in usage:
        if not served_as(config.AUDITOR_MODEL, record.model):
            raise RuntimeError(f"the auditor was served by {record.model}, not {config.AUDITOR_MODEL}")
    verdict = result["verdicts"][0]
    return {
        "id": item["id"],
        "paper": item["paper"],
        "split": item["split"],
        "label": item["label"],
        "verdict": verdict.verdict,
        "reason": verdict.reason,
        "evidence": verdict.evidence,
        "evidence_found": verdict.evidence_found,
        "basis": verdict.basis,
        "steps": len(usage),
        "model": usage[-1].model,
        "usage": {
            "input_tokens": sum(u.input_tokens for u in usage),
            "output_tokens": sum(u.output_tokens for u in usage),
            "cache_read_input_tokens": sum(u.cache_read_tokens for u in usage),
            "cache_creation_input_tokens": sum(u.cache_write_tokens for u in usage),
        },
    }


def run(limit: int, concurrency: int) -> int:
    items = load_items()
    RESULTS.mkdir(parents=True, exist_ok=True)
    results_path, errors_path = RESULTS / "results.jsonl", RESULTS / "errors.jsonl"
    done = {row["id"] for row in _read_jsonl(results_path)}
    todo = [item for item in items if item["id"] not in done][: limit or None]
    print(f"{len(todo)} of {len(items)} items to audit", file=sys.stderr)
    failed = 0

    def work(item: dict) -> None:
        nonlocal failed
        try:
            row = audit_item(item)
        except Exception as error:  # noqa: BLE001 - recorded, never scored
            failed += 1
            _append(errors_path, {"id": item["id"], "label": item["label"], "error": f"{type(error).__name__}: {error}"})
            print(f"  {item['id']} FAILED: {error}", file=sys.stderr)
            return
        _append(results_path, row)
        print(f"  {item['id']} -> {row['verdict']} ({row['basis']})", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(work, todo))
    print(f"Done: {len(todo) - failed} audited, {failed} failed -> {results_path}", file=sys.stderr)
    return 1 if failed else 0


# --- Scoring -----------------------------------------------------------------


def summarize_rows(rows: list[dict], errors: int) -> str:
    """Scores from the result rows. `errors` is the number of failed attempts on record."""
    lines = [
        "# Citation-audit benchmark",
        "",
        "Generated by `python -m evals.audit.citations summarize`. Do not edit by hand.",
        "",
        f"{len(rows)} items scored, {errors} failed attempts on record. Auditor: "
        f"{', '.join(sorted({row['model'] for row in rows})) or 'none'}.",
    ]
    for split in ("held_out", "tuning"):
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            continue
        by = {label: [row for row in subset if row["label"] == label] for label in LABELS}
        count = lambda label, *verdicts: sum(row["verdict"] in verdicts for row in by[label])  # noqa: E731
        flagged = [row for row in subset if row["verdict"] == "not_supported"]
        lines += [
            "",
            f"## {split.replace('_', '-').capitalize()} papers",
            "",
            "| Measure | Result |",
            "| --- | --- |",
            f"| Original citations judged supported | {rate(count('original', 'supported'), len(by['original']))} |",
            f"| Original citations wrongly flagged as not supported | {rate(count('original', 'not_supported'), len(by['original']))} |",
            f"| Far swaps caught (not supported) | {rate(count('swapped_far', 'not_supported'), len(by['swapped_far']))} |",
            f"| Near swaps caught (not supported) | {rate(count('swapped_near', 'not_supported'), len(by['swapped_near']))} |",
            f"| Near swaps caught or doubted (not or partly supported) | {rate(count('swapped_near', 'not_supported', 'partially_supported'), len(by['swapped_near']))} |",
            f"| \"Not supported\" flags that were swaps | {rate(sum(row['label'] != 'original' for row in flagged), len(flagged))} |",
            "",
            "| Pairing | Supported | Partly | Not supported | Could not verify |",
            "| --- | --- | --- | --- | --- |",
        ]
        for label in LABELS:
            verdicts = Counter(row["verdict"] for row in by[label])
            lines.append(f"| {label} | " + " | ".join(str(verdicts[v]) for v in VERDICTS) + " |")
    full_text = sum(row["basis"] == "full_text" for row in rows)
    tokens_in = sum(sum(v for k, v in row["usage"].items() if k != "output_tokens") for row in rows)
    tokens_out = sum(row["usage"]["output_tokens"] for row in rows)
    lines += [
        "",
        "## Effort",
        "",
        f"{full_text} of {len(rows)} items were checked against full text. Mean steps per item: "
        f"{sum(row['steps'] for row in rows) / len(rows):.1f}. Tokens: {tokens_in} in, {tokens_out} out.",
        "",
        "## Limits",
        "",
        "- \"Original\" is presumed correct, but authors do miscite. Some flags on originals may be right.",
        "- A near swap can support the sentence by coincidence, so its catch rate understates the auditor.",
        "- Swapping in a different paper is one kind of miscitation. Overstated or distorted claims "
        "about the right paper are another, and this benchmark does not contain them.",
        "- The papers are well known, and the auditor's model may remember them. Scores on obscure "
        "papers could be lower.",
    ]
    return "\n".join(lines) + "\n"


def summarize() -> int:
    rows = _read_jsonl(RESULTS / "results.jsonl")
    if not rows:
        raise SystemExit("No results yet. Run `python -m evals.audit.citations run` first.")
    scored = {row["id"] for row in rows}
    errors = [e for e in _read_jsonl(RESULTS / "errors.jsonl") if e["id"] not in scored]
    text = summarize_rows(rows, len(errors))
    (RESULTS / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the citation auditor.")
    parser.add_argument("command", choices=["build", "run", "summarize"])
    parser.add_argument("--limit", type=int, default=0, help="run: audit at most this many items")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "build":
        return build()
    if args.command == "run":
        return run(args.limit, args.concurrency)
    return summarize()


if __name__ == "__main__":
    raise SystemExit(main())
