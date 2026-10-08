"""Runtime settings, read from the environment (see .env.example)."""

import os

# Model routing: a stronger model plans and writes, a cheap one does the
# high-volume evidence extraction.
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "claude-sonnet-5-5")
RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "claude-haiku-4-5")
WRITER_MODEL = os.getenv("WRITER_MODEL", "claude-sonnet-5-5")

MAX_SUB_QUESTIONS = int(os.getenv("MAX_SUB_QUESTIONS", "5"))
SEARCH_RESULTS_PER_QUESTION = int(os.getenv("SEARCH_RESULTS_PER_QUESTION", "5"))
