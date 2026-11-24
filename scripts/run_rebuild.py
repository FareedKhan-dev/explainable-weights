"""scripts/run_rebuild.py: the rebuild test (wsp/rebuild.py) on a problem's pick, with a live agent model.

For each problem, the agent describes its pick in at most 200 words with the read tool only, two fresh agents rebuild
it from that description, and one control agent rebuilds it from the task alone. The pick comes from
results/qwen_runs.json. Everything goes to runs/rebuild/<task>/{describe,r1,r2,control}/.

    python scripts/run_rebuild.py --tasks word-reversal-3x3 --conversation-cap 6
"""
import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--tasks", nargs="+", required=True)
ap.add_argument("--parallel", type=int, default=3)
ap.add_argument("--conversation-cap", type=float, default=6.0, help="budget for one describe or rebuild conversation")
ap.add_argument("--credit-cap", type=float, default=2000.0, help="budget for every call under runs/ together")
a = ap.parse_args()
if not os.environ.get("WSP_ENDPOINT"):
    sys.exit("set WSP_ENDPOINT to an OpenAI-compatible chat completions endpoint (and WSP_MODEL, WSP_KEY_FILE)")
os.environ.setdefault("WSP_PAPER", str(ROOT.parent / "weight-space-programming-autoresearch"))
os.environ.setdefault("WSP_HF_HOME", str(ROOT / ".hfh"))
os.environ["WSP_RUNS"] = str(ROOT / "runs" / "rebuild")
sys.path.insert(0, str(ROOT))
from wsp import agent, config, rebuild  # noqa: E402


def limits(max_calls):
    return agent.Limits(config.RUNS, max_calls=max_calls, credit_cap=a.credit_cap, run_credit_cap=a.conversation_cap,
                        compact_at=rebuild.CONTEXT_BOUND, stop_file=ROOT / "STOP", budget_root=ROOT / "runs")


def pick(task):
    """The problem's pick: its network file, visible score and exact weights, from the analysis of the runs."""
    picks = {p["task"]: p["pick"] for p in json.loads((ROOT / "results" / "qwen_runs.json").read_text(encoding="utf-8"))["picks"]}
    p = picks[task]
    return next((ROOT / "runs" / p["run"] / "attempts").glob(f"{p['idx']:03d}_*.py")), p["visible"], p["weights"]


def one_problem(task):
    client, key = agent.connect()
    network, visible, weights = pick(task)
    folder = config.RUNS / task
    text, calls = rebuild.describe(client, network, task, folder / "describe", key, limits(20))
    if not text:
        return f"{task}: no description ({calls} calls)"
    jobs = [("r1", False), ("r2", False), ("control", True)]
    with ThreadPoolExecutor(3) as ex:
        done = list(ex.map(lambda j: rebuild.rebuild(agent.connect()[0], task, text, visible, weights, folder / j[0], key,
                                                     control=j[1], limits=limits(80)), jobs))
    return f"{task}: described in {len(text.split())} words; " + "; ".join(
        f"{name} -> {best['name'] if best else None} ({calls} calls)" for (name, _), (best, calls) in zip(jobs, done))


if __name__ == "__main__":
    stamp = lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"launched {stamp()}: {' '.join(a.tasks)}", flush=True)
    with ThreadPoolExecutor(max(1, a.parallel // 3)) as ex:
        for line in ex.map(one_problem, a.tasks):
            print(line, flush=True)
    print(f"finished {stamp()}", flush=True)
