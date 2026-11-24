"""results/hero_answers.py: the output block after the hero, a real capture.

The network Qwen3.7-Max wrote for word reversal (attempt 27 of its first run) answers the first three hidden prompts,
prompts its run never saw, and is then scored on all 2,000 of them with the same greedy decoding the paper's evaluator
uses. The totals must match the recorded pick in qwen_runs.json, or nothing is written.

    <paper .venv python> results/hero_answers.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WS = ROOT.parent
os.environ.setdefault("WSP_PAPER", str(WS / "weight-space-programming-autoresearch"))
sys.path.insert(0, str(ROOT))
from wsp import networks, verify  # noqa: E402
from wsp.tasks import Problem  # noqa: E402

NET = "runs/word-reversal-3x3-qwen37-max-s1/attempts/027_PosRouting_noW_o.py"


def main():
    os.chdir(ROOT)
    import json
    task, net = networks.build(networks.load(NET), "word-reversal-3x3")
    hidden = Problem("word-reversal-3x3").examples(verify.HIDDEN_N, verify.HIDDEN_SEED)
    got = verify.answers(net, task, [e.prompt for e in hidden])
    right = sum(g == e.answer for g, e in zip(got, hidden))
    weights = networks.weights(net)
    nonzero = sum(int((p != 0).sum()) for p in net.parameters())
    pick = next(p["pick"] for p in json.loads((HERE / "qwen_runs.json").read_text(encoding="utf-8"))["picks"]
                if p["task"] == "word-reversal-3x3")
    assert right == len(hidden) == 2000 and pick["hidden"] == 1.0 and weights == pick["weights"] == 11450, (right, pick)
    assert nonzero == 128, nonzero
    lines = [f"prompt {e.prompt}   answer {g}" for e, g in zip(hidden[:3], got[:3])]
    lines += ["", f"{right:,} of {len(hidden):,} hidden prompts right", f"{weights:,} weights, {nonzero} of them not zero"]
    out = HERE / "outputs" / "hero_answers.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(out.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
