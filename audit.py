"""CLI entry point for the paper audit.

    python audit.py draft/main.tex                    a LaTeX file or folder
    python audit.py arxiv:1706.03762                  a paper on arXiv
    python audit.py draft/ --mode references          reference check only: no model calls
    python audit.py draft/ --mode citations           skip the related-work search
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # before app imports, which read settings from the environment

from app.audit.builder import build_audit_graph  # noqa: E402
from app.audit.latex import PaperError  # noqa: E402
from app.tools import scholar  # noqa: E402

AUDITS_DIR = Path("audits")
MAX_CONCURRENCY = 8


def _describe(node: str, update: dict) -> str | None:
    """One progress line for a finished step, or None for steps not worth a line."""
    if node == "read_paper":
        paper = update["paper"]
        return (
            f"{paper.title or 'untitled paper'}: {len(paper.references)} references, "
            f"{len(paper.sentences)} citing sentences"
        )
    if node == "check_references":
        counts = Counter(r.status for r in update["resolutions"])
        return ", ".join(f"{count} {status}" for status, count in counts.most_common())
    if node == "plan_citation_audit":
        return f"{len(update['audit_plan'])} cited works to audit"
    if node == "auditor":
        verdicts = update["verdicts"]
        counts = Counter(v.verdict for v in verdicts)
        summary = ", ".join(f"{count} {verdict}" for verdict, count in counts.most_common())
        return f"[{verdicts[0].key}] {summary}"
    if node == "contributions":
        claims = "".join(f"\n    {c.id}. {c.statement}" for c in update["contributions"])
        return f"{len(update['contributions'])} claimed contributions{claims}"
    if node == "scout":
        candidates = update["candidates"]
        which = candidates[0].contribution_id if candidates else "?"
        return f"{len(candidates)} candidate papers for contribution {which}"
    if node == "assess":
        counts = Counter(a.overlap for a in update.get("assessments", []))
        return ", ".join(f"{count} {overlap}" for overlap, count in counts.most_common()) or "no result"
    if node == "report":
        return "written"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the references and citations of a LaTeX paper.")
    parser.add_argument("paper", help="a .tex file, a folder, or an arXiv id such as arxiv:1706.03762")
    parser.add_argument(
        "--mode", choices=["full", "citations", "references"], default="full",
        help="'references' makes no model calls; 'citations' skips the related-work search",
    )  # fmt: skip
    args = parser.parse_args()
    # Paper text can hold characters the Windows console cannot print.
    sys.stdout.reconfigure(errors="replace")

    state: dict = {}
    stream = build_audit_graph().stream(
        {"source": args.paper, "mode": args.mode},
        config={"max_concurrency": MAX_CONCURRENCY},
        stream_mode=["updates", "values"],
    )
    try:
        for mode, chunk in stream:
            if mode == "values":
                state = chunk
                continue
            for node, update in chunk.items():
                line = _describe(node, update or {})
                if line:
                    print(f"[{node}] {line}")
    except (PaperError, scholar.ScholarError) as error:
        print(f"The audit could not start: {error}", file=sys.stderr)
        return 1

    slug = re.sub(r"[^a-z0-9]+", "-", (state["paper"].title or args.paper).lower()).strip("-")[:60]
    folder = AUDITS_DIR / (slug or "paper")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "audit.md").write_text(state["report"], encoding="utf-8")
    (folder / "audit.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=lambda item: item.model_dump()),
        encoding="utf-8",
    )

    usage = state.get("usage", [])
    if usage:
        print("\nTokens by agent (input / output):")
        for agent in dict.fromkeys(u.agent for u in usage):
            calls = [u for u in usage if u.agent == agent]
            print(
                f"  {agent:<9}{sum(u.total_input_tokens for u in calls):>9} /"
                f"{sum(u.output_tokens for u in calls):>8}  ({len(calls)} calls, {calls[0].model})"
            )
    credits = scholar.openalex_credits
    if credits["used"]:
        print(f"\nOpenAlex credits used: {credits['used']} (remaining today: {credits['remaining']})")
    print(f"\nAudit saved to {folder / 'audit.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
