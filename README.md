# Paper Audit

An agentic system that checks a research paper's references and citations
before a reviewer does. Give it a LaTeX draft; it reports references that do
not exist, citations that the cited work does not support, and close prior work
the draft never mentions.

Most research agents write text. This one checks it. Every claim of support has
to quote the cited work, and the quotation is confirmed in code.

```powershell
python audit.py draft/main.tex                 # a LaTeX file or folder
python audit.py arxiv:1706.03762               # a paper on arXiv
python audit.py draft/ --mode references       # reference check only: no API key needed
```

## What the audit does

| Stage | What it checks | How |
|---|---|---|
| Reference check | Does each reference exist? Does its DOI or arXiv id belong to it? Has it been retracted? | Code. Lookups on arXiv, Crossref and OpenAlex. |
| Citation audit | Does the cited work support the sentence that cites it? | One auditor agent per cited work. It reads the abstract and can search the full text. |
| Related work | What close prior work is not cited? | One scout agent per claimed contribution. It searches by topic and follows the citation graph. |
| Novelty overlap | Does an existing paper already make a claimed contribution? | A model compares each contribution with the scouts' candidates. |

The report is assembled by code from the verdicts, so it cannot say something
the audit did not find.

## How it is built

```
read paper -> check references -> auditor x N -> contributions -> scout x N -> assess x N -> report
   code            code             agents          model          agents        model       code
```

The order of the stages is fixed, because an audit always needs the same
checks. The judgment is inside the stages: auditors and scouts are agents that
choose their own searches and decide when they have seen enough. Built with
LangGraph and the Anthropic SDK.

| Agent | Job | Default model |
|---|---|---|
| Auditor | Checks one cited work against the sentences citing it | `claude-sonnet-5-5` |
| Scout | Finds the closest prior work for one contribution | `claude-haiku-4-5` |
| Analyst | Extracts contributions; rates overlap | `claude-sonnet-5-5` |

## Measured results

The reference checker needs no model, so it has been run. On two held-out
papers (138 references, of which 24 were planted errors):

| Measure | Result |
|---|---|
| Real references verified | 0.94 (107/114) |
| Invented references caught | 1.00 (12/12) |
| Wrong identifiers caught | 1.00 (12/12) |

Two papers is a small sample, and the planted errors are synthetic. See
[evals/audit/runs/refcheck-held_out.md](evals/audit/runs/refcheck-held_out.md)
for every miss.

The citation auditor has a benchmark too, built from real papers with citations
swapped to the wrong work. It has not been run yet, because it needs an API key.

## The research engine

The repository also contains the system this grew from: a general research
agent that answers a question with a cited report (`python main.py "..."`). An
orchestrator decides each round whether to dispatch more researcher agents or
to write. It shares the tool loop, the LLM wrapper and the evaluation ideas
with the audit. Its own 30-question benchmark is in `evals/`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # add ANTHROPIC_API_KEY; TAVILY_API_KEY only for main.py
```

The reference check (`--mode references`) and its benchmark run without any
key. arXiv, Crossref and OpenAlex are free, but OpenAlex meters free use by the
day, and a full audit uses a good share of the allowance.

## Test

```powershell
.\.venv\Scripts\python -m pytest
```

104 tests. They run offline: models, the judge and every web service are
replaced by fakes, and a test that tries to reach a scholarly service fails.

## Layout

```
audit.py                 paper-audit CLI
main.py                  research-engine CLI
app/
  config.py              models and limits
  llm.py                 Anthropic SDK wrapper
  text.py                sentence splitting, shared with the evals
  agents/loop.py         the tool loop every agent shares
  agents/                research engine: orchestrator, researcher, writer
  graph/                 research engine: state and graph
  audit/latex.py         reads a LaTeX paper into sentences and references
  audit/resolver.py      reference check
  audit/auditor.py       citation-auditor agent
  audit/scout.py         related-work scout agent
  audit/analyst.py       contributions and overlap
  audit/report.py        the report, rendered by code
  audit/builder.py       the audit graph
  tools/scholar.py       arXiv, Crossref, OpenAlex, PDF text
  tools/search.py        Tavily web search
evals/
  audit/refcheck.py      reference-check benchmark (run; results in audit/runs/)
  audit/citations.py     citation-audit benchmark (built; not yet run)
  ...                    research-engine benchmark and judge
tests/                   offline tests
```

## Limits

- LaTeX only. PDF and Word drafts are not read yet.
- A citation can only be checked against text that is free to read. With an
  abstract alone, many verdicts are "could not be verified".
- "Not found" is a prompt to check a reference, not proof that it is invented.
- Nothing that calls a model has been run live yet.
