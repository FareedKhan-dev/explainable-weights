"""tests/test_agent_loop.py: drive the agent loop (wsp/agent.py) with a scripted client instead of the API, and check it.

No model is called. The script reads program.md, runs the baseline, tries three forbidden things (a src/ read, `ls`,
a redirected run), writes the paper's published 332-weight GCD network (FinalRadix4FixedPositions) and scores it, ends its
turn, then trips the compaction threshold, which is not a stall, so two more sessions run before the stall rule stops
it. Then the credit caps (the study's and one run's, both counted from logged tokens) and the STOP file are tested.
Every invocation writes under its own .test_runs/wsp-<time>/.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPER = Path(os.environ.get("WSP_PAPER", ROOT.parent / "weight-space-programming-autoresearch")).resolve()
os.environ.update(WSP_PAPER=str(PAPER),
                  WSP_PYTHON=os.environ.get("WSP_PYTHON", str(PAPER / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))),
                  WSP_RUNS=str(ROOT / ".test_runs"), WSP_HF_HOME=os.environ.get("WSP_HF_HOME", str(ROOT / ".hfh")))
sys.path.insert(0, str(ROOT))
from wsp import agent, sandbox  # noqa: E402
from wsp.client import Reply  # noqa: E402

GCD_NET = PAPER / "experiments" / "nl-bottleneck" / "originals" / "gcd-three-digits.py"   # the paper's published 332-weight GCD network
TASK, KEY = "gcd-three-digits", "test-key-never-logged"


def reply(tools=(), text="", tokens_in=1000):
    calls = {i: {"id": f"c{i}", "name": n, "arguments": json.dumps(a)} for i, (n, a) in enumerate(tools)}
    return Reply(model="scripted", text=text, stop_reason="tool_calls" if calls else "stop", input_tokens=tokens_in,
                 output_tokens=50, tool_calls_by_index=calls, seconds=0.01)


class Scripted:
    model = "Test/Scripted"

    def __init__(self, steps):
        self.steps, self.seen = list(steps), []

    def chat(self, messages, effort=None, max_completion_tokens=None, extra=None):
        self.seen.append([dict(m) for m in messages])
        return self.steps.pop(0) if self.steps else reply(text="nothing left")


def main():
    seed = int(time.time())
    root = ROOT / ".test_runs" / f"wsp-{seed}"
    root.mkdir(parents=True)
    stop = root / "STOP"
    gcd = GCD_NET.read_text(encoding="utf-8")
    cmd = f"uv run interpretable_transformer.py --task {TASK}"
    steps = [reply([("read", {"file_path": "program.md"})]), reply([("bash", {"command": cmd})]),
             reply([("read", {"file_path": "src/task.py"})]),
             reply([("write", {"file_path": "interpretable_transformer.py", "content": gcd})]),
             reply([("bash", {"command": cmd})]),
             reply([("write", {"file_path": "interpretable_transformers_lib/FinalRadix4FixedPositions.py", "content": gcd})]),
             reply([("bash", {"command": "ls -la"})]), reply([("bash", {"command": cmd + " > run.log 2>&1"})]),
             reply(text=f"Both attempts are logged. {KEY}"),
             reply([("read", {"file_path": "results/overall_results.csv"})], tokens_in=250_000),
             reply(text="Stopping here.")]
    client = Scripted(steps)
    folder = agent.run(TASK, seed, client, root=root, limits=agent.Limits(root, stop_file=stop), key=KEY)

    checks = []
    att = [json.loads(l) for l in open(folder / "attempts" / "attempts.jsonl", encoding="utf-8")]
    checks.append(("two attempts in order", [x["name"] for x in att] == ["RandomInitInterpretable", "FinalRadix4FixedPositions"]))
    # the untrained baseline can score an answer or two by chance: its near-tied outputs flip with float rounding
    checks.append(("baseline near 0, GCD network 1", att[0]["accuracy"] < 0.05 and att[1]["accuracy"] == 1.0))
    checks.append(("weights read from the CSV", att[1]["n_params"] == 332.0))
    snap = folder / "attempts" / "002_FinalRadix4FixedPositions.py"
    checks.append(("snapshot is the exact file", snap.exists() and hashlib.sha256(snap.read_bytes()).hexdigest()
                   == att[1]["sha256"] == hashlib.sha256(gcd.encode("utf-8")).hexdigest()))
    checks.append(("CSV has two rows", sandbox.attempts(folder) == 2))
    meta = json.loads((folder / "metadata.json").read_text())
    checks.append(("a compacted session is not a stall", [s["end"] for s in meta["sessions"]]
                   == ["end of turn", "compacted", "end of turn", "end of turn"]))
    checks.append(("three refusals", meta["sessions"][-1]["refused_calls"] == 3))
    calls = sum(1 for _ in open(folder / "agent_logs" / "calls.jsonl", encoding="utf-8"))
    checks.append(("twelve calls logged", calls == 12 and meta["calls"] == 12))
    checks.append(("session 2 resumed the conversation", "Continue the autoresearch loop" in client.seen[9][-1]["content"]
                   and len(client.seen[9]) > 10))
    checks.append(("session 3 started fresh", len(client.seen[10]) == 2 and "Continue" in client.seen[10][1]["content"]))
    checks.append(("session 4 resumed session 3", len(client.seen[11]) == 4))
    logs = "".join(p.read_text(encoding="utf-8") for p in (folder / "agent_logs").glob("*.jsonl"))
    checks.append(("key scrubbed from every log", KEY not in logs and "<key>" in logs))

    # 11 calls of 1,000 prompt tokens and one of 250,000, each with 50 output tokens counted four times
    first = agent.credits(folder / "agent_logs" / "calls.jsonl")
    checks.append(("credits counted from tokens", abs(first - (261_000 + 4 * 600) * 5.2 / 1e6) < 1e-9
                   and meta["credits"] == round(first, 2)))
    c2 = Scripted([reply(text="never asked")])
    r2 = agent.run(TASK, seed + 1, c2, root=root, limits=agent.Limits(root, credit_cap=1.0, stop_file=stop))
    checks.append(("study credit cap stops before any call", not c2.seen
                   and json.loads((r2 / "metadata.json").read_text())["sessions"][0]["end"] == "credit cap"))
    # 50,000-token prompts cost about 0.26 credits each, so a 0.6 cap allows three calls and stops the fourth
    c4 = Scripted([reply([("read", {"file_path": "program.md"})], tokens_in=50_000) for _ in range(6)])
    r4 = agent.run(TASK, seed + 3, c4, root=root, limits=agent.Limits(root, credit_cap=1e9, run_credit_cap=0.6, stop_file=stop))
    m4 = json.loads((r4 / "metadata.json").read_text())
    checks.append(("run credit cap stops the run after three calls", len(c4.seen) == 3
                   and m4["sessions"][0]["end"] == "run credit cap" and m4["credits"] == 0.78))
    # replies cut off at the output limit before any tool call: a note, and the session goes on
    cut = lambda: Reply(model="scripted", status="truncated", stop_reason="length", input_tokens=1000, output_tokens=32000)
    c5 = Scripted([cut(), cut(), reply([("read", {"file_path": "program.md"})]), reply(text="done")])
    m5 = json.loads((agent.run(TASK, seed + 4, c5, root=root, limits=agent.Limits(root, stop_file=stop)) / "metadata.json").read_text())
    note = "Your last reply reached the output limit"
    checks.append(("a cut-off reply gets a note and the session goes on", c5.seen[1][-1]["content"].startswith(note)
                   and c5.seen[2][-1]["content"].startswith(note) and len(c5.seen[2]) == len(c5.seen[1]) + 1
                   and [s["cut_offs"] for s in m5["sessions"]] == [2, 0]
                   and [s["end"] for s in m5["sessions"]] == ["end of turn", "end of turn"]))
    c6 = Scripted([cut() for _ in range(4)])
    m6 = json.loads((agent.run(TASK, seed + 5, c6, root=root, limits=agent.Limits(root, stop_file=stop)) / "metadata.json").read_text())
    checks.append(("four cut-offs in a row start a fresh session, not a stall", len(c6.seen[4]) == 2
                   and [s["end"] for s in m6["sessions"]] == ["cut off", "end of turn", "end of turn"]))
    # an API failure: the same conversation again after a wait, and a long outage stops the run
    fail = lambda: Reply(model="scripted", status="infra", error="HTTP 500: busy")
    slept, real_sleep = [], agent.time.sleep
    agent.time.sleep = slept.append
    try:
        c7 = Scripted([fail(), reply([("read", {"file_path": "program.md"})]), reply(text="done")])
        m7 = json.loads((agent.run(TASK, seed + 6, c7, root=root, limits=agent.Limits(root, stop_file=stop)) / "metadata.json").read_text())
        checks.append(("an API failure retries the same conversation after a wait", c7.seen[1] == c7.seen[0]
                       and slept == [60] and [s["end"] for s in m7["sessions"]] == ["infra: HTTP 500: busy", "end of turn", "end of turn"]))
        slept.clear()
        c8 = Scripted([fail() for _ in range(5)])
        m8 = json.loads((agent.run(TASK, seed + 7, c8, root=root, limits=agent.Limits(root, stop_file=stop)) / "metadata.json").read_text())
        checks.append(("five API failures in a row stop the run", len(c8.seen) == 5 and slept == [60, 120, 240, 480]
                       and all(s["end"].startswith("infra") for s in m8["sessions"]) and len(m8["sessions"]) == 5))
    finally:
        agent.time.sleep = real_sleep

    stop.write_text("stop\n", encoding="utf-8")
    c3 = Scripted([reply(text="never asked")])
    r3 = agent.run(TASK, seed + 2, c3, root=root, limits=agent.Limits(root, credit_cap=1e9, stop_file=stop))
    checks.append(("stop file stops before any call", not c3.seen
                   and json.loads((r3 / "metadata.json").read_text())["sessions"][0]["end"] == "stop file"))
    stop.rename(root / f"STOP.used.{seed}")

    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name)
    print(f"{sum(ok for _, ok in checks)} of {len(checks)} checks pass")
    return 0 if all(ok for _, ok in checks) else 1


def test_agent_loop():
    """The pytest entry point: the checks above, skipped when the paper's code is not there."""
    import pytest
    paper = Path(os.environ.get("WSP_PAPER") or Path(__file__).resolve().parents[2] / "weight-space-programming-autoresearch")
    if not (paper / "experiments").is_dir():
        pytest.skip("set WSP_PAPER to a clone of the paper's code")
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
