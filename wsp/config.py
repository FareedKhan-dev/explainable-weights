"""wsp/config.py: every model name, path and limit the system reads, in one place."""
import os
from pathlib import Path

# the agent that writes every weight, reached through an OpenAI-compatible chat endpoint
AGENT_MODEL = os.environ.get("WSP_MODEL", "Qwen3.7-Max")
ENDPOINT = os.environ.get("WSP_ENDPOINT", "")            # the runner names the endpoint, nothing is called without it
KEY_FILE = Path(os.environ.get("WSP_KEY_FILE", "api_key.txt"))

# the paper's repository cloned next to this project, the Python that runs its evaluator, and our run folders
PAPER = Path(os.environ.get("WSP_PAPER", Path(__file__).resolve().parents[1].parent / "weight-space-programming-autoresearch"))
EVOLVE = PAPER / "experiments" / "evolve"
PYTHON = Path(os.environ.get("WSP_PYTHON", PAPER / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")))
RUNS = Path(os.environ.get("WSP_RUNS", "runs"))
HF_HOME = Path(os.environ.get("WSP_HF_HOME", ".hfh"))     # kept short, Windows caps a full path at 260 characters

# the study's first five problems, from a toy to a real dataset, and two runs of each (the runners name any of the
# paper's 17 problems on their command line)
PROBLEMS = ["word-reversal-3x3", "addition-five-digits", "gcd-three-digits", "date-normalization", "trec-coarse"]
SEEDS = [1, 2]

# the budget unit the run caps count: input tokens at 5.2 credits per million (cached tokens included) plus output tokens
# counted four times. Later balance readings put real output at 168 to 189 credits per million, so a cap bounds tokens
PRICE_PER_M_INPUT, OUTPUT_WEIGHT = 5.2, 4.0

# limits: budget units for the whole study and for one run (the runners raise the study cap), and calls, hours and
# prompt size for one run (a fresh session starts past COMPACT_AT tokens, because every call re-reads the whole
# conversation and pays for it)
CREDIT_CAP, RUN_CREDIT_CAP = 400.0, 28.0
MAX_CALLS, MAX_HOURS, COMPACT_AT = 400, 3.0, 60_000
