"""Runs the benchmark: each case through the real graph, then through the graders.

    python -m evals.run --variant baseline --pilot
    python -m evals.run --variant baseline
    python -m evals.run --variant v1 --set MAX_RESEARCH_ROUNDS=1 --note "One round only"

Output goes to <flow>/<variant>/:

    results.jsonl          one row per scored (case, rep), appended as each finishes
    errors.jsonl           one row per failed attempt, with a failure class
    traces/<id>_rep<k>.json  everything the agents did for that case
    judge/<id>_rep<k>.json   every verdict the judge gave, with the source text it saw
    config.json            the settings this variant ran with

A failed attempt never becomes a score. Re-running the same command resumes:
(case, rep) pairs that already have a row are skipped, and failed ones are retried.
"""

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # before app imports, which read settings from the environment

from app import config, llm  # noqa: E402
from app.agents import orchestrator, researcher, writer  # noqa: E402
from app.graph.builder import build_graph  # noqa: E402
from app.graph.state import Usage  # noqa: E402
from evals import DEFAULT_FLOW, VARIANT, judge, metrics  # noqa: E402
from evals.cases import CASES_PATH, Case, load_cases  # noqa: E402

PILOT_IDS = ("tech-01", "sci-01", "biz-01", "pol-01", "trap-01")

# Files whose contents define the measurement. If any of them changes, scores
# from before and after are not comparable, so the runner asks for a fresh
# approval. The application under test is deliberately not on this list.
HARNESS_FILES = [
    Path(__file__),
    Path(metrics.__file__),
    Path(judge.__file__),
    Path(__file__).parent / "sentences.py",
    Path(__file__).parent.parent / "app" / "text.py",  # the sentence splitter itself
    Path(__file__).parent / "cases.py",
    CASES_PATH,
]

_write_lock = threading.Lock()
_count_lock = threading.Lock()


class AttemptFailed(Exception):
    """An attempt that must not be scored. `failure_class` says why."""

    def __init__(self, failure_class: str, message: str):
        super().__init__(message)
        self.failure_class = failure_class


# --- small helpers -----------------------------------------------------------


def _usage_dict(records: list[Usage]) -> dict[str, int]:
    return {
        "input_tokens": sum(u.input_tokens for u in records),
        "output_tokens": sum(u.output_tokens for u in records),
        "cache_read_input_tokens": sum(u.cache_read_tokens for u in records),
        "cache_creation_input_tokens": sum(u.cache_write_tokens for u in records),
    }


def _usage_by_model(records: list[Usage]) -> dict[str, dict[str, int]]:
    models = dict.fromkeys(u.model for u in records)
    return {m: _usage_dict([u for u in records if u.model == m]) for m in models}


def _append_line(path: Path, record: dict) -> None:
    with _write_lock:
        # A crash can leave a torn last line; keep it from merging with this one.
        torn = path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(("\n" if torn else "") + json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def harness_sha(extra_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in sorted({p.resolve() for p in HARNESS_FILES} | {Path(p).resolve() for p in extra_paths}):
        digest.update(path.name.encode())
        digest.update(b"\0")
        # Line endings differ between checkouts; the measurement does not.
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def effective_config() -> dict:
    return {
        "orchestrator_model": config.ORCHESTRATOR_MODEL,
        "researcher_model": config.RESEARCHER_MODEL,
        "writer_model": config.WRITER_MODEL,
        "max_research_rounds": config.MAX_RESEARCH_ROUNDS,
        "max_tasks_per_round": config.MAX_TASKS_PER_ROUND,
        "max_researcher_steps": config.MAX_RESEARCHER_STEPS,
        "search_results_per_query": config.SEARCH_RESULTS_PER_QUERY,
        "page_char_limit": config.PAGE_CHAR_LIMIT,
        "judge_model": judge.JUDGE_MODEL,
        "judge_effort": judge.JUDGE_EFFORT,
    }


def apply_overrides(pairs: list[str]) -> dict[str, str]:
    """Apply KEY=VALUE settings to app.config, keeping each setting's type."""
    applied = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not hasattr(config, key) or not key.isupper():
            raise SystemExit(f"--set expects KEY=VALUE with a setting from app/config.py, got {pair!r}")
        setattr(config, key, type(getattr(config, key))(value))
        applied[key] = value
    return applied


# --- running one case --------------------------------------------------------


def _turns(node: str, update: dict) -> list[dict]:
    """Convert one agent's state update into transcript turns."""
    if node == "orchestrator":
        decision = update["decision"]
        tasks = "".join(f"\n- {task}" for task in decision.tasks)
        return [
            {
                "role": "assistant",
                "content": f"[orchestrator] {decision.action}: {decision.reasoning}{tasks}",
            }
        ]
    if node == "researcher":
        report = update["task_reports"][0]
        turns: list[dict] = [{"role": "assistant", "content": f"[researcher] task: {report.task}"}]
        for call in update["tool_calls"]:
            turns.append(
                {"role": "tool_call", "name": call.name, "content": json.dumps(call.input, indent=2)}
            )
            if call.name != "submit_findings":
                prefix = "(error) " if call.is_error else ""
                turns.append({"role": "tool_result", "name": call.name, "content": prefix + call.output})
        kept = "".join(
            f"\n- {f.claim} ({', '.join(f.source_urls)})" for f in update["findings"]
        )
        note = f" Note: {report.note}." if report.note else ""
        turns.append(
            {
                "role": "assistant",
                "content": (
                    f"[researcher] {report.findings} grounded findings kept after "
                    f"{report.steps} steps.{note}{kept}"
                ),
            }
        )
        return turns
    return [{"role": "assistant", "content": update["report"]}]


def run_case(case: Case, progress: dict) -> dict:
    """Run the application on one question and grade the report.

    `progress` is filled in as work completes, so that a failure or timeout can
    still report what had been spent up to that point.
    """
    transcript = [
        {"role": "system", "content": f"[orchestrator system prompt]\n{orchestrator.SYSTEM}"},
        {"role": "system", "content": f"[researcher system prompt]\n{researcher.SYSTEM}"},
        {"role": "system", "content": f"[writer system prompt]\n{writer.SYSTEM}"},
        {"role": "user", "content": case.question},
    ]
    progress["stage"] = "app"
    state: dict = {}
    started = time.monotonic()
    stream = build_graph().stream({"query": case.question}, stream_mode=["updates", "values"])
    for mode, chunk in stream:
        if mode == "values":
            state = chunk
            progress["usage"] = state.get("usage", [])
            continue
        for node, update in chunk.items():
            transcript += _turns(node, update)
    latency_s = time.monotonic() - started

    report = state.get("report", "")
    if not report.strip():
        raise AttemptFailed("app_error", "the writer returned an empty report")

    requested = {
        "orchestrator": config.ORCHESTRATOR_MODEL,
        "researcher": config.RESEARCHER_MODEL,
        "writer": config.WRITER_MODEL,
    }
    for usage in state["usage"]:
        if not judge.served_as(requested[usage.agent], usage.model):
            raise AttemptFailed(
                "serving_substitution",
                f"{usage.agent} was served by {usage.model}, not {requested[usage.agent]}",
            )

    progress["stage"] = "grading"
    numbers = writer.number_sources(state.get("findings", []))
    sources = {number: state["sources"][url] for url, number in numbers.items()}
    grade = metrics.grade_report(case, report, sources, progress["judge_usage"])

    tool_calls = [c for c in state.get("tool_calls", []) if c.name != "submit_findings"]
    cited = {n for s in grade.detail["sentences"] for n in s["citations"] if n in sources}
    return {
        "state": state,
        "transcript": transcript,
        "grade": grade,
        "sources": sources,
        "perf": {
            "latency_s": round(latency_s, 1),
            "rounds": state.get("round", 0),
            "tasks": len(state.get("task_reports", [])),
            "tool_calls": len(tool_calls),
            "web_searches": sum(1 for c in tool_calls if c.name == "web_search"),
            "pages_read": sum(1 for c in tool_calls if c.name == "read_page" and not c.is_error),
            "sources_cited": len(cited),
            "report_words": len(report.split()),
        },
    }


def _with_ceiling(work, seconds: float):
    """Run `work()` with a hard wall-clock limit.

    Python cannot stop a running thread, so a timed-out attempt may finish in
    the background. Its result is discarded and its slot is freed.
    """
    box: dict = {}

    def target():
        try:
            box["value"] = work()
        except BaseException as error:  # noqa: BLE001 - re-raised by the caller below
            box["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds if seconds > 0 else None)
    if thread.is_alive():
        raise AttemptFailed("timeout", f"exceeded the {seconds:g}s wall-clock ceiling")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _failure_class(error: BaseException, stage: str) -> str:
    if isinstance(error, AttemptFailed):
        return error.failure_class
    if isinstance(error, llm.LLMError):
        return {"refusal": "refusal", "max_tokens": "truncated"}.get(error.kind, "app_error")
    return "grader_error" if stage == "grading" else "app_error"


def attempt(case: Case, rep: int, variant_dir: Path, timeout_s: float) -> dict | None:
    """Run one (case, rep). Write a results row and return it, or record the failure."""
    progress: dict = {"stage": "app", "usage": [], "judge_usage": []}
    started = time.monotonic()
    stem = f"{case.id}_rep{rep}"
    try:
        result = _with_ceiling(lambda: run_case(case, progress), timeout_s)
    except Exception as error:  # noqa: BLE001 - every failure is classified and recorded
        _append_line(
            variant_dir / "errors.jsonl",
            {
                "prompt_id": case.id,
                "rep": rep,
                "failure_class": _failure_class(error, progress["stage"]),
                "stage": progress["stage"],
                "error": f"{type(error).__name__}: {error}",
                # What was billed before the failure still counts as spend.
                "usage_by_model": _usage_by_model(list(progress["usage"])),
                "judge_model": judge.JUDGE_MODEL,
                "judge_usage": _usage_dict(list(progress["judge_usage"])),
                "latency_s": round(time.monotonic() - started, 1),
            },
        )
        return None

    state, grade = result["state"], result["grade"]
    app_usage, judge_usage = state["usage"], list(progress["judge_usage"])
    if not app_usage or not judge_usage:
        raise RuntimeError(f"{stem}: a scored attempt recorded no token usage; refusing to write it")

    writer_models = [u.model for u in app_usage if u.agent == "writer"]
    row = {
        "prompt_id": case.id,
        "rep": rep,
        "prompt": case.question,
        "tags": case.tags,
        "status": "ok",
        "stop_reason": "end_turn",
        "model": writer_models[-1],
        "usage": _usage_dict(app_usage),
        "usage_by_model": _usage_by_model(app_usage),
        "judge_model": judge_usage[-1].model,
        "judge_usage": _usage_dict(judge_usage),
        **result["perf"],
        "grade": grade.scores,
        "explanation": grade.explanation,
        "meta": {"kind": "trap" if case.is_trap else "answerable", **grade.detail["counts"]},
    }
    _write_json(variant_dir / "traces" / f"{stem}.json", result["transcript"])
    _write_json(
        variant_dir / "judge" / f"{stem}.json",
        {
            "prompt_id": case.id,
            "rep": rep,
            "question": case.question,
            "report": state["report"],
            "sources": {n: s.model_dump() for n, s in sorted(result["sources"].items())},
            **grade.detail,
        },
    )
    # The row is written last: its presence is what marks the attempt as done.
    _append_line(variant_dir / "results.jsonl", row)
    return row


# --- command line ------------------------------------------------------------


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the research-report benchmark.")
    parser.add_argument("--flow", type=Path, default=DEFAULT_FLOW, help="results directory")
    parser.add_argument("--variant", default="baseline", help="'baseline' or 'v<N>'")
    parser.add_argument("--reps", type=int, default=1, help="repetitions per case")
    parser.add_argument("--concurrency", type=int, default=2, help="cases run at once")
    parser.add_argument("--timeout-s", type=float, default=1800, help="per-attempt ceiling; 0 = none")
    parser.add_argument("--cases", default="", help="comma-separated case ids (default: all)")
    parser.add_argument("--pilot", action="store_true", help="run one case from each group")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                        help="override a setting from app/config.py for this variant")
    parser.add_argument("--note", default="", help="one line describing what this variant changes")
    parser.add_argument("--approve-harness", action="store_true",
                        help="record the current cases and graders as the approved measurement")
    args = parser.parse_args(argv)
    if not VARIANT.match(args.variant):
        parser.error(f"--variant must be 'baseline' or 'v<N>', got {args.variant!r}")
    if args.reps < 1 or args.concurrency < 1 or args.timeout_s < 0:
        parser.error("--reps and --concurrency must be at least 1, and --timeout-s at least 0")
    if args.variant == "baseline" and args.overrides:
        parser.error("the baseline runs the default settings; use --variant v<N> with --set")
    return args


def _check_harness(flow: Path, approve: bool) -> None:
    state_path = flow / "_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    sha = harness_sha(state.get("harness_paths", []))
    if state.get("harness_sha") == sha:
        return
    if approve:
        state["harness_sha"] = sha
        _write_json(state_path, state)
        print(f"Harness approved: sha256 {sha[:12]} recorded in {state_path}", file=sys.stderr)
        return
    reason = "has not been approved yet" if not state.get("harness_sha") else "changed since it was approved"
    raise SystemExit(
        f"The eval harness (cases and graders) {reason}.\n"
        "Scores from different harness versions are not comparable. Review the change, "
        "then re-run with --approve-harness."
    )


def _check_config(variant_dir: Path, overrides: dict[str, str]) -> None:
    """A variant is one configuration. Refuse to mix two in the same directory."""
    path = variant_dir / "config.json"
    current = {"overrides": overrides, "settings": effective_config()}
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != current:
            raise SystemExit(
                f"{path} records different settings from this run. Results from two "
                "configurations must not share a variant; use a new --variant."
            )
    else:
        _write_json(path, current)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    overrides = apply_overrides(args.overrides)

    cases = load_cases()
    wanted = set(PILOT_IDS) if args.pilot else {c for c in args.cases.split(",") if c}
    if wanted:
        unknown = wanted - {c.id for c in cases}
        if unknown:
            raise SystemExit(f"Unknown case ids: {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c.id in wanted]

    args.flow.mkdir(parents=True, exist_ok=True)
    _check_harness(args.flow, args.approve_harness)
    variant_dir = args.flow / args.variant
    for directory in (variant_dir / "traces", variant_dir / "judge"):
        directory.mkdir(parents=True, exist_ok=True)
    _check_config(variant_dir, overrides)
    if args.variant != "baseline" and (args.note or not (variant_dir / "change.md").exists()):
        settings = ", ".join(f"{k}={v}" for k, v in overrides.items()) or "no setting overrides"
        (variant_dir / "change.md").write_text(
            f"{args.note or settings}\n\nSettings: {settings}\n", encoding="utf-8"
        )

    done = set()
    results_path = variant_dir / "results.jsonl"
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                done.add((row["prompt_id"], row["rep"]))
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [(c, rep) for c in cases for rep in range(args.reps) if (c.id, rep) not in done]
    total = len(cases) * args.reps
    print(f"[{args.variant}] {len(todo)} of {total} (case, rep) to run", file=sys.stderr)

    counts = {"ok": 0, "failed": 0}

    def work(item: tuple[Case, int]) -> None:
        case, rep = item
        started = time.monotonic()
        row = attempt(case, rep, variant_dir, args.timeout_s)
        with _count_lock:
            counts["ok" if row else "failed"] += 1
            finished = counts["ok"] + counts["failed"]
        if row:
            headline = " ".join(f"{k}={v:.2f}" for k, v in row["grade"].items())
            outcome = f"ok  {headline}"
        else:
            outcome = "FAILED (see errors.jsonl)"
        print(
            f"[{args.variant}] {finished}/{len(todo)} {case.id} rep{rep} "
            f"{outcome}  ({time.monotonic() - started:.0f}s)",
            file=sys.stderr,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(work, todo))

    print(
        f"[{args.variant}] done: {counts['ok']} scored, {counts['failed']} failed -> {results_path}",
        file=sys.stderr,
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
