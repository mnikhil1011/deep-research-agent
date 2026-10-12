"""Measures the reference checker on real references and on planted wrong ones.

    python -m evals.audit.refcheck                   the held-out papers
    python -m evals.audit.refcheck --split tuning    the papers the rules were tuned on

No model is involved, so this needs no API key. It does call arXiv, Crossref
and OpenAlex, and each planted "invented" reference costs about 10 OpenAlex
credits.

For each benchmark paper, three kinds of reference are checked:

    real       the paper's own references. The checker should verify them.
    invented   a title spliced from two real titles, with one's authors and the
               other's year. No such work exists, so it must not be verified.
    wrong_id   a real reference carrying another reference's arXiv id. The
               checker should report that the id belongs to a different work.

The planted references are generated from a fixed seed, so every run builds the
same benchmark.
"""

import argparse
import json
import random
import re
from collections import Counter

from app.audit.latex import load_paper
from app.audit.models import Paper, Reference
from app.audit.resolver import resolve_all, title_similarity
from app.tools import scholar
from evals.audit import RUNS, benchmark_papers, rate

PLANTED_PER_KIND = 6
_IDENTIFIER = re.compile(r"arXiv[:\s]*\S+|https?://\S+|\bdoi:\s*\S+|abs/\d{4}\.\d{4,5}", re.IGNORECASE)


def invented_references(real: list[Reference], count: int, rng: random.Random) -> list[Reference]:
    """Build references to works that do not exist, from pieces of ones that do."""
    usable = [r for r in real if len(r.title.split()) >= 4]
    planted: list[Reference] = []
    attempts = 0
    while len(planted) < count and len(usable) >= 2 and attempts < 200:
        attempts += 1
        first, second = rng.sample(usable, 2)
        head, tail = first.title.split(), second.title.split()
        title = " ".join(head[: len(head) // 2] + tail[len(tail) // 2 :])
        # The splice must not happen to reproduce a real title.
        if any(title_similarity(title, r.title) >= 0.8 for r in real):
            continue
        authors = ", ".join(first.authors) or "Anonymous"
        planted.append(
            Reference(
                key=f"invented-{len(planted) + 1}",
                raw=f"{authors}. {title}. {second.year or ''}".strip(),
                title=title,
                authors=first.authors,
                year=second.year,
            )
        )
    return planted


def wrong_id_references(real: list[Reference], count: int, rng: random.Random) -> list[Reference]:
    """Give real references the arXiv id of a different work."""
    donors = [r for r in real if r.arxiv_id]
    planted: list[Reference] = []
    attempts = 0
    while len(planted) < count and donors and attempts < 200:
        attempts += 1
        victim, donor = rng.choice([r for r in real if r.title]), rng.choice(donors)
        if victim.key == donor.key or title_similarity(victim.title, donor.title) >= 0.5:
            continue
        if any(p.title == victim.title for p in planted):
            continue
        planted.append(
            victim.model_copy(
                update={
                    "key": f"wrong-id-{len(planted) + 1}",
                    "raw": " ".join(_IDENTIFIER.sub("", victim.raw).split()),
                    "doi": "",
                    "arxiv_id": donor.arxiv_id,
                }
            )
        )
    return planted


def check_paper(arxiv_id: str) -> list[dict]:
    """Check one paper's real references and the references planted from them."""
    real = list(load_paper(f"arxiv:{arxiv_id}").references.values())
    rng = random.Random(f"refcheck-{arxiv_id}")
    labelled = (
        [("real", r) for r in real]
        + [("invented", r) for r in invented_references(real, PLANTED_PER_KIND, rng)]
        + [("wrong_id", r) for r in wrong_id_references(real, PLANTED_PER_KIND, rng)]
    )
    # No sentences, so nothing is "cited" and no OpenAlex credits go on abstracts.
    paper = Paper(
        title="", abstract="", opening="", sentences=[], references={r.key: r for _, r in labelled}
    )
    labels = {r.key: label for label, r in labelled}
    return [
        {
            "paper": arxiv_id,
            "key": resolution.key,
            "label": labels[resolution.key],
            "as_cited": paper.references[resolution.key].title or paper.references[resolution.key].raw[:120],
            "status": resolution.status,
            "found_by": resolution.found_by,
            "issues": resolution.issues,
        }
        for resolution in resolve_all(paper)
    ]


def summarize(rows: list[dict], split: str) -> str:
    """Scores, computed from the rows. A lookup that failed is not counted either way."""
    answered = [row for row in rows if row["status"] != "lookup_failed"]
    by_label = {label: [r for r in answered if r["label"] == label] for label in ("real", "invented", "wrong_id")}
    flagged = [r for r in answered if r["status"] in ("not_found", "mismatch")]
    planted_flagged = [r for r in flagged if r["label"] != "real"]
    lines = [
        f"# Reference-check benchmark: {split.replace('_', '-')} papers",
        "",
        "Generated by `python -m evals.audit.refcheck`. Do not edit by hand.",
        "",
        *(
            ["The matching rules were tuned on these papers, so these scores flatter the checker.", ""]
            if split == "tuning"
            else []
        ),
        f"Papers: {', '.join(sorted({row['paper'] for row in rows}))}. "
        f"{len(rows)} references checked, {len(rows) - len(answered)} lookups failed and are not scored.",
        "",
        "| Measure | Result |",
        "| --- | --- |",
        f"| Real references verified | {rate(sum(r['status'] == 'verified' for r in by_label['real']), len(by_label['real']))} |",
        f"| Invented references caught (not verified) | {rate(sum(r['status'] != 'verified' for r in by_label['invented']), len(by_label['invented']))} |",
        f"| Wrong identifiers caught (reported as a mismatch) | {rate(sum(r['status'] == 'mismatch' for r in by_label['wrong_id']), len(by_label['wrong_id']))} |",
        f"| Flagged references that were planted | {rate(len(planted_flagged), len(flagged))} |",
        "",
        "## Outcomes by kind of reference",
        "",
        "| Kind | Verified | Mismatch | Not found | Lookup failed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label in ("real", "invented", "wrong_id"):
        counts = Counter(r["status"] for r in rows if r["label"] == label)
        lines.append(
            f"| {label} | {counts['verified']} | {counts['mismatch']} | "
            f"{counts['not_found']} | {counts['lookup_failed']} |"
        )
    unverified_real = [r for r in by_label["real"] if r["status"] != "verified"]
    if unverified_real:
        lines += [
            "",
            "## Real references that were not verified",
            "",
            "These count against the checker in the first row. Some are its misses and some are "
            "works the indexes do not cover, such as books and reports.",
            "",
        ]
        lines += [f"- `{r['paper']}` `{r['key']}` ({r['status']}): {r['as_cited']}" for r in unverified_real]
    lines += [
        "",
        "## Limits",
        "",
        "- The planted references are synthetic. Real fabricated references may be harder or "
        "easier to catch than a spliced title.",
        "- The benchmark papers are well known, so their references are better indexed than "
        "those of a typical draft.",
        "- A handful of papers is a small sample. Treat these rates as rough.",
        "- The services change, so the same benchmark can score slightly differently on "
        "another day.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the reference checker.")
    parser.add_argument("--split", choices=["held_out", "tuning"], default="held_out")
    parser.add_argument("--fresh", action="store_true", help="ignore saved results and check again")
    args = parser.parse_args(argv)
    papers = benchmark_papers(args.split)

    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"refcheck-{args.split}.jsonl"
    rows = [] if args.fresh or not path.exists() else [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]  # fmt: skip
    done = {row["paper"] for row in rows}
    for arxiv_id in papers:
        if arxiv_id in done:
            continue
        print(f"Checking the references of arXiv:{arxiv_id} ...")
        rows += check_paper(arxiv_id)
        # Saved after each paper, so an interrupted run keeps what it has.
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    rows = [row for row in rows if row["paper"] in papers]
    text = summarize(rows, args.split)
    (RUNS / f"refcheck-{args.split}.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"OpenAlex credits used in this run: {scholar.openalex_credits['used']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
