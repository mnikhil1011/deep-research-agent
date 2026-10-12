"""Evaluation harness for the research engine. See README.md in this folder."""

import re
from pathlib import Path

DEFAULT_FLOW = Path("evals") / "runs"
# Result directories are named this way so the report builder recognises them.
VARIANT = re.compile(r"^(baseline|v[1-9]\d*)$")
