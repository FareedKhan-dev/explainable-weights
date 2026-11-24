"""results/fix_answers.py: what Qwen3.7-Max's first designs and its fixed design answer for the running prompt.

    <paper .venv python> results/fix_answers.py

Builds the saved attempt files of the reversal run (run 1) that matter for the fix the agent describes in its log: the
first design (attempts 4 and 5, logged 0 of 200 and 0 of 3), the rewrite that fixed it (attempts 6 and 7, logged 5 of 5
and 200 of 200), and asks each for its answer to fnm_mod_wfd=, the way the evaluator generates it. CPU only, seconds.
Writes results/fix_answers.json for the g-fix figure."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WS = ROOT.parent
os.environ.setdefault("WSP_PAPER", str(WS / "weight-space-programming-autoresearch"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(ROOT))
from wsp import networks, verify  # noqa: E402

PROMPT = "fnm_mod_wfd="
RUN = ROOT / "runs" / "word-reversal-3x3-qwen37-max-s1" / "attempts"


def main():
    out = dict(provisional=False, prompt=PROMPT, attempts=[])
    for f in sorted(RUN.glob("00[4-7]_*.py")):
        task, net = networks.build(networks.load(f), "word-reversal-3x3")
        answer = verify.answers(net, task, [PROMPT])[0]
        out["attempts"].append(dict(idx=int(f.name[:3]), name=f.stem[4:], answer=answer))
        print(f.name, answer)
    (HERE / "fix_answers.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
