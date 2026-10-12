"""Runtime settings, read from the environment (see .env.example)."""

import os

# Model routing: a stronger model orchestrates and writes, a cheap one does the
# high-volume research.
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "claude-sonnet-5-5")
RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "claude-haiku-4-5")
WRITER_MODEL = os.getenv("WRITER_MODEL", "claude-sonnet-5-5")

# Limits on the agents' autonomy. Together they bound the cost of one run.
MAX_RESEARCH_ROUNDS = int(os.getenv("MAX_RESEARCH_ROUNDS", "3"))
MAX_TASKS_PER_ROUND = int(os.getenv("MAX_TASKS_PER_ROUND", "4"))
MAX_RESEARCHER_STEPS = int(os.getenv("MAX_RESEARCHER_STEPS", "6"))

SEARCH_RESULTS_PER_QUERY = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "5"))
PAGE_CHAR_LIMIT = int(os.getenv("PAGE_CHAR_LIMIT", "12000"))

# --- Paper audit -------------------------------------------------------------

# The auditor's verdicts are the product, so it gets the stronger model. Scouts
# make many cheap search calls.
AUDITOR_MODEL = os.getenv("AUDITOR_MODEL", "claude-sonnet-5-5")
SCOUT_MODEL = os.getenv("SCOUT_MODEL", "claude-haiku-4-5")
ANALYST_MODEL = os.getenv("ANALYST_MODEL", "claude-sonnet-5-5")

MAX_AUDITOR_STEPS = int(os.getenv("MAX_AUDITOR_STEPS", "5"))
MAX_SCOUT_STEPS = int(os.getenv("MAX_SCOUT_STEPS", "6"))
MAX_REFERENCES_AUDITED = int(os.getenv("MAX_REFERENCES_AUDITED", "60"))
MAX_CONTRIBUTIONS = int(os.getenv("MAX_CONTRIBUTIONS", "4"))
MAX_CANDIDATES_PER_CONTRIBUTION = int(os.getenv("MAX_CANDIDATES_PER_CONTRIBUTION", "6"))

# Scholarly APIs. Both settings are optional. A contact address is good manners
# towards Crossref and OpenAlex. OpenAlex meters free use by the day; if you
# have an OpenAlex API key with its own allowance, it is sent with each request.
SCHOLAR_MAILTO = os.getenv("SCHOLAR_MAILTO", "")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
# OpenAlex searches are metered, so they are the last resort when checking a reference.
OPENALEX_FALLBACK = os.getenv("OPENALEX_FALLBACK", "true").lower() == "true"
CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
