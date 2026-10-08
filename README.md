# Deep Research Engine

A multi-agent research system built on LangGraph. Give it a question; it plans the
research, gathers evidence from the web, and writes a Markdown report in which
every claim cites its source.

## Status

Phase 1 of 6: a working end-to-end skeleton.

```
query -> [planner] -> [researcher] -> [writer] -> cited report
```

| Agent | Job | Default model |
|---|---|---|
| Planner | Breaks the query into sub-questions | `claude-sonnet-5-5` |
| Researcher | One web search per sub-question, extracts cited findings | `claude-haiku-4-5` |
| Writer | Turns the findings into a cited report | `claude-sonnet-5-5` |

Planned next: parallel researchers, a claim verifier and reviewer loop,
human approval of the plan, budgets and guardrails, an eval suite, and an API
and UI.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # then add your ANTHROPIC_API_KEY and TAVILY_API_KEY
```

## Run

```powershell
.\.venv\Scripts\python main.py "How will solid-state batteries affect EV pricing by 2030?"
```

The report is saved to `reports/`, and token usage per agent is printed.

## Test

```powershell
.\.venv\Scripts\python -m pytest
```

The tests run the whole graph offline with the LLM and web search stubbed, so
they need no API keys.

## Layout

```
app/
  config.py          settings and model routing
  llm.py             Anthropic SDK wrapper (structured output, usage tracking)
  graph/state.py     shared state and data models
  graph/builder.py   the LangGraph graph
  agents/            planner, researcher, writer
  tools/search.py    Tavily web search
main.py              CLI
tests/               offline graph tests
```
