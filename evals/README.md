# Evaluation

This folder holds three benchmarks: two for the paper audit, in `audit/`, and
one for the research engine, described in the rest of this file.

## Paper-audit benchmarks

Both are built from real arXiv papers listed in `audit/papers.json`. The papers
are split in two. The `tuning` papers were used while writing the matching
rules, so scores on them flatter the system. The `held_out` papers were not,
and theirs are the scores to report.

### Reference check (has been run)

```powershell
.\.venv\Scripts\python -m evals.audit.refcheck                  # held-out papers
.\.venv\Scripts\python -m evals.audit.refcheck --split tuning
```

No model is involved, so no API key is needed. For each paper it checks the
real references and two kinds of planted error: invented references, spliced
from two real titles, and real references given another work's arXiv id.
Results are in `audit/runs/refcheck-held_out.md`. An invented reference costs
about 10 OpenAlex credits, so a full run uses much of the free daily allowance.

### Citation audit (built, not yet run)

```powershell
.\.venv\Scripts\python -m evals.audit.citations build       # no model; already done
.\.venv\Scripts\python -m evals.audit.citations run --limit 9   # a small pilot first
.\.venv\Scripts\python -m evals.audit.citations run
.\.venv\Scripts\python -m evals.audit.citations summarize
```

Each benchmark sentence cites exactly one work. It is audited three times:
with the work its authors cited (`original`), with an unrelated work from the
same reference list (`swapped_far`), and with the most closely related other
work (`swapped_near`). The auditor should support the first and reject the
other two. The items are frozen in `audit/citation_items.json`. The summary
reports the share of swaps caught, the share of originals wrongly flagged, and
how often the auditor declined to judge.

`run` calls the model once or more per item, so it costs money. A failed
attempt is written to `errors.jsonl` and never scored, and re-running resumes.

## Research-engine benchmark

This measures how good the research engine's reports are. It runs 30 benchmark
questions through the real graph, grades each report, and reports the scores
with confidence intervals, cost and latency.

No scores exist yet. The harness is built and tested offline, but it has not
been run against the live APIs.

## What is measured

Every metric is a ratio over counted items, so any score can be recomputed by
hand from the verdicts saved beside it.

| Metric | Definition | Graded by |
|---|---|---|
| **Groundedness** (headline) | Fully supported claims / all claims | Judge |
| **Completeness** | Aspects of the question answered / aspects | Judge |
| **Trap handled** | 1 if the report caught the question's false premise, else 0 | Judge |
| Citation accuracy | Fully supported sentences / cited sentences | Judge |
| Citation coverage | Cited sentences / all claims | Judge |
| Unsupported rate | Unsupported sentences / cited sentences (lower is better) | Judge |
| Valid citations | Citation markers that name a real source / all markers | Code |

A claim is a sentence that carries a citation, or an uncited sentence that
states a checkable fact. So groundedness = citation accuracy x citation coverage.

Groundedness and completeness are read together. A report can score high on
groundedness by saying very little, and completeness is what catches that.

Also recorded per case: latency, research rounds, tasks, tool calls, searches,
pages read, sources cited, report length, and token usage per model, from which
cost is derived.

**What is not measured:** whether the sources themselves are right. A report
that faithfully repeats a wrong web page scores well on groundedness.

## The cases

`cases.json` holds 30 questions, listed readably in [CASES.md](CASES.md):

- 24 answerable questions, six each on technology, science, business and
  policy. Each lists the aspects a complete answer must address. Aspects say
  what must be covered, not what the answer is, so they do not go stale.
- 6 trap questions built on a false premise, such as a Nobel Prize in
  Mathematics. A good report says the premise is false and invents nothing.

The system under test sees only the question.

## The judge

The judge is Claude Opus 5.5, which no agent in the pipeline uses, so nothing
grades its own work. It answers four narrow questions, each with its own prompt
in `judge.py`: does this source text support this sentence, does this uncited
sentence state a fact, does the report address this aspect, and did the report
catch this false premise. It is told to judge against the source text only and
not to use its own knowledge.

## How to run it

Each step that calls a model costs money. Run them in this order.

```powershell
# 1. Check the judge on inputs with known answers (about ten small calls).
.\.venv\Scripts\python -m evals.check_judge

# 2. Pilot: one case from each group. The first run must approve the harness.
.\.venv\Scripts\python -m evals.run --pilot --approve-harness

# 3. Read the pilot's numbers, including the measured cost per case.
.\.venv\Scripts\python -m evals.summarize

# 4. Check the judge against your own judgment on a sample of its verdicts.
.\.venv\Scripts\python -m evals.review --n 24
#    ...fill in evals/runs/baseline/review.md, then:
.\.venv\Scripts\python -m evals.review --score

# 5. Full run. It resumes, so the pilot cases are not repeated.
.\.venv\Scripts\python -m evals.run
.\.venv\Scripts\python -m evals.summarize
```

To compare a change against the baseline, run it as a numbered variant:

```powershell
.\.venv\Scripts\python -m evals.run --variant v1 --set MAX_RESEARCH_ROUNDS=1 --note "One research round"
.\.venv\Scripts\python -m evals.summarize
```

The summary then shows the paired change per metric and says whether it is
larger than the noise.

Useful options: `--reps N` repeats each case, `--cases tech-01,sci-02` runs a
subset, `--concurrency N` sets how many cases run at once, and `--timeout-s N`
sets the time limit per attempt.

## What the runner guarantees

- **A failure is never a score.** A refusal, a timeout, an API error or a judge
  failure goes to `errors.jsonl` with a failure class. Only completed, graded
  attempts reach `results.jsonl`.
- **It resumes.** Re-running a command skips the (case, repetition) pairs that
  already have a row and retries the ones that failed.
- **The model is checked.** If a response comes from a model other than the one
  configured, the attempt fails instead of being scored.
- **A variant is one configuration.** The settings are saved in `config.json`,
  and the runner refuses to add results from different settings to it.
- **The measurement is versioned.** The cases and graders are hashed. If any of
  them changes, the runner stops until you pass `--approve-harness`, because
  scores from before and after are not comparable.

## Files

```
evals/
  cases.json        the 30 cases
  CASES.md          the same cases, readable (generated)
  cases.py          loads and validates cases
  sentences.py      splits a report into sentences and reads citations
  judge.py          the four judge prompts and calls
  metrics.py        turns verdicts into scores
  run.py            the runner
  summarize.py      scores with intervals, cost and latency
  check_judge.py    known-answer probes for the judge
  review.py         human agreement check for the judge
  runs/
    _state.json     metric declarations, prices, approved harness hash
    summary.md      the latest summary (generated)
    baseline/       results.jsonl, errors.jsonl, config.json
                    traces/ and judge/ hold the evidence and are not committed
```

## Known limits

- Web search results change over time, so two runs of the same case can differ
  for reasons unrelated to the code. Repetitions and intervals account for this.
- Latency includes any automatic retries the API client makes.
- Cost covers model tokens only, not web search fees.
- A timed-out attempt cannot be cancelled, so it may keep spending in the
  background until it finishes. Its result is discarded.
- With 30 cases, small differences between variants will be within noise. The
  summary says so when they are.
