"""tests/test_rebuild.py: drive the rebuild test (wsp/rebuild.py) with scripted clients. No model is called.

describe: the agent reads model.py, answers with 240 words, is asked to shorten, answers with 150.
rebuild:  the agent writes the paper's published 332-weight GCD network, scores it, then tries a seventh scored run, which is refused.
"""
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
from wsp import agent, rebuild  # noqa: E402
from wsp.agent import Limits  # noqa: E402
from wsp.client import Reply  # noqa: E402

GCD = PAPER / "experiments" / "nl-bottleneck" / "originals" / "gcd-three-digits.py"   # the paper's published 332-weight GCD network


def reply(tools=(), text=""):
    calls = {i: {"id": f"c{i}", "name": n, "arguments": json.dumps(a)} for i, (n, a) in enumerate(tools)}
    return Reply(model="scripted", text=text, stop_reason="tool_calls" if calls else "stop", input_tokens=500,
                 output_tokens=40, tool_calls_by_index=calls, seconds=0.01)


class Scripted:
    model = "Test/Scripted"

    def __init__(self, steps):
        self.steps, self.seen, self.tools_offered = list(steps), [], []

    def chat(self, messages, effort=None, max_completion_tokens=None, extra=None):
        self.seen.append([dict(m) for m in messages])
        self.tools_offered.append([t["function"]["name"] for t in (extra or {}).get("tools", [])])
        return self.steps.pop(0) if self.steps else reply(text="done")


def main():
    root = ROOT / ".test_runs" / f"rebuild-{int(time.time())}"
    stop = root / "STOP"
    checks = []
    d = Scripted([reply([("read", {"file_path": "model.py"})]), reply(text=" ".join(["word"] * 240)),
                  reply(text=" ".join(["short"] * 150))])
    text, calls = rebuild.describe(d, GCD, "gcd-three-digits", root / "describe", limits=Limits(root, stop_file=stop))
    checks.append(("describer offered only the read tool", all(t == ["read"] for t in d.tools_offered)))
    checks.append(("describer gets the read-only system message", d.seen[0][0] == {"role": "system", "content": rebuild.DESCRIBE_SYSTEM}))
    checks.append(("a 240-word answer gets the shorten request", "That is 240 words" in d.seen[2][-1]["content"]))
    checks.append(("final description is the 150-word one", len(text.split()) == 150 and calls == 3))
    dmeta = json.loads((root / "describe" / "meta.json").read_text())
    checks.append(("description and its meta saved", (root / "describe" / "description.txt").exists()
                   and dmeta["words"] == 150 and dmeta["ends"] == ["end of turn", "end of turn"] and dmeta["credits"] > 0))

    gcd = GCD.read_text(encoding="utf-8")
    cmd = "uv run interpretable_transformer.py --task gcd-three-digits"
    steps = [reply([("read", {"file_path": "src/task.py"})]),
             reply([("write", {"file_path": "interpretable_transformer.py", "content": gcd})])]
    steps += [reply([("bash", {"command": cmd})]) for _ in range(7)]
    steps += [reply(text="Final model: FinalRadix4FixedPositions")]
    r = Scripted(steps)
    best, calls = rebuild.rebuild(r, "gcd-three-digits", text, 1.0, 332, root / "r1", limits=Limits(root, stop_file=stop))
    tool_texts = [m["content"] for conv in r.seen for m in conv if m.get("role") == "tool"]
    checks.append(("rebuild may read src/", any("class GCDTask" in t for t in tool_texts)))
    checks.append(("prompt carries the description and the original's score", "100.0% on these 200 examples" in r.seen[0][1]["content"]
                   and "332 parameters" in r.seen[0][1]["content"] and r.seen[0][0]["content"] == agent.SYSTEM))
    checks.append(("a seventh scored run is refused", any("all 6 scored runs are used" in t for t in tool_texts)))
    checks.append(("final network is the GCD one at 100%", best and best["name"] == "FinalRadix4FixedPositions" and best["accuracy"] == 1.0))
    checks.append(("no program.md in a rebuild sandbox", not (root / "r1" / "program.md").exists()))
    rmeta = json.loads((root / "r1" / "meta.json").read_text())
    checks.append(("rebuild meta records its end and final network", rmeta["kind"] == "rebuild" and rmeta["ends"] == ["end of turn"]
                   and rmeta["final"]["name"] == "FinalRadix4FixedPositions" and rmeta["words_given"] == 150))
    # a score from one example must never be picked over the default 200-example run
    checks.append(("only default visible runs can be the final pick", rebuild.default_run(["--task", "gcd-three-digits"])
                   and rebuild.default_run(["--task", "gcd-three-digits", "--n-samples", "200"])
                   and not rebuild.default_run(["--task", "gcd-three-digits", "--n-samples", "1"])
                   and not rebuild.default_run(["--task", "gcd-three-digits", "--seed", "3"])
                   and not rebuild.default_run(["--task", "gcd-three-digits", "--num-examples", "200"])
                   and rebuild.default_run(["--task", "gcd-three-digits", "--verbose"])          # only prints more
                   and rebuild.default_run(["--task", "gcd-three-digits", "--verbose", "--n-samples", "200"])))

    for label, ok in checks:
        print(("PASS " if ok else "FAIL ") + label)
    print(f"{sum(ok for _, ok in checks)} of {len(checks)} checks pass")
    return 0 if all(ok for _, ok in checks) else 1


def test_rebuild():
    """The pytest entry point: the checks above, skipped when the paper's code is not there."""
    import pytest
    paper = Path(os.environ.get("WSP_PAPER") or Path(__file__).resolve().parents[2] / "weight-space-programming-autoresearch")
    if not (paper / "experiments").is_dir():
        pytest.skip("set WSP_PAPER to a clone of the paper's code")
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
