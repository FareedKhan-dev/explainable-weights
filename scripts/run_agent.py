"""scripts/run_agent.py: run the weight-writing loop on one problem, or send one probe call.

    export WSP_ENDPOINT=https://<provider>/v1/chat/completions    # any OpenAI-compatible chat completions endpoint
    export WSP_MODEL=Qwen3.7-Max                                  # the model the endpoint serves
    export WSP_KEY_FILE=api_key.txt                               # a file that holds the API key
    export WSP_PAPER=../weight-space-programming-autoresearch     # the paper's code, at commit 4f8eb615
    python scripts/run_agent.py --probe
    python scripts/run_agent.py --task word-reversal-3x3 --seed 1 --study main

Each study keeps its runs in runs/<study>/. The study cap counts every call logged under runs/, and a file named STOP in
the repository root stops every run before its next call. A run whose folder exists resumes from it.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--task", default="word-reversal-3x3")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--study", default="main")
ap.add_argument("--run-cap", type=float, default=28.0, help="budget one run may spend, counted from its logged tokens")
ap.add_argument("--study-cap", type=float, default=2000.0, help="budget for every call under runs/ together")
ap.add_argument("--probe", action="store_true", help="send one call that should use the bash tool, then stop")
a = ap.parse_args()
if not os.environ.get("WSP_ENDPOINT"):
    sys.exit("set WSP_ENDPOINT to an OpenAI-compatible chat completions endpoint (and WSP_MODEL, WSP_KEY_FILE)")
if not a.probe and (ROOT / "STOP").exists():
    sys.exit(f"STOP file present: {a.task} seed {a.seed} not started")
os.environ.setdefault("WSP_PAPER", str(ROOT.parent / "weight-space-programming-autoresearch"))
os.environ.setdefault("WSP_HF_HOME", str(ROOT / ".hfh"))
os.environ["WSP_RUNS"] = str(ROOT / "runs" / a.study)
sys.path.insert(0, str(ROOT))
from wsp import agent, config  # noqa: E402

client, key = agent.connect()
if a.probe:
    t0 = time.time()
    r = client.chat([{"role": "user", "content": "Call the bash tool once with `uv run interpretable_transformer.py --task demo`."}],
                    max_completion_tokens=2000, extra={"tools": agent.TOOLS})
    print(json.dumps({"model": r.model, "status": r.status, "stop": r.stop_reason, "seconds": round(time.time() - t0, 1),
                      "in": r.input_tokens, "out": r.output_tokens, "reasoning": r.reasoning_tokens,
                      "tool_calls": [(t["name"], t["arguments"][:160]) for t in r.tool_calls], "error": r.error}, indent=1))
else:
    limits = agent.Limits(config.RUNS, credit_cap=a.study_cap, run_credit_cap=a.run_cap, stop_file=ROOT / "STOP",
                          budget_root=ROOT / "runs")
    folder = agent.run(a.task, a.seed, client, root=config.RUNS, limits=limits, key=key)
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    print(json.dumps({k: meta.get(k) for k in ("run", "attempts", "calls", "credits", "started", "finished")}, indent=1))
