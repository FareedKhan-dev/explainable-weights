"""results/floors.py: the guessing floor of every set the post scores on, the score of always giving the commonest answer.

    <paper .venv python> floors.py

For the four dataset problems: the majority share on the visible set (200 examples, seed 0, what the agent saw), on
the hidden set (2,000 training rows outside the agent's pool, seed 1) and on the official labelled split. For GCD:
the share of input pairs whose answer is 001 over all 998,001 inputs and on the hidden set, beside 6/pi^2, the limit
of that share for large numbers. Writes results/floors.json.
"""
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
os.environ.setdefault("WSP_PAPER", str(WS / "weight-space-programming-autoresearch"))
os.environ.setdefault("WSP_HF_HOME", str(Path(__file__).resolve().parents[1] / ".hfh"))
sys.path.insert(0, str(HERE.parent))
from wsp import networks, verify  # noqa: E402


def floor(labels):
    counts = Counter(labels)
    label, top = counts.most_common(1)[0]
    return dict(label=label, share=round(top / len(labels), 4), count=top, n=len(labels))


def main():
    tasks, _ = networks.paper_code()
    out = dict(provisional=False, datasets={}, gcd={})
    for name in verify.DATASETS:
        visible = [e.answer for e in tasks.get_task(name).generate_examples(200, seed=0)]
        hidden = [label for _, label in verify.dataset_hidden(name)]
        official = [label for _, label in verify.dataset_rows(name, verify.BENCHMARK[name])]
        out["datasets"][name] = dict(visible=floor(visible), hidden=floor(hidden), official=dict(split=verify.BENCHMARK[name], **floor(official)))
        print(name, {k: (v["label"], v["share"], v["n"]) for k, v in out["datasets"][name].items() if isinstance(v, dict)})
    coprime = sum(1 for a in range(1, 1000) for b in range(1, 1000) if math.gcd(a, b) == 1)
    hidden = tasks.get_task("gcd-three-digits").generate_examples(verify.HIDDEN_N, seed=verify.HIDDEN_SEED)
    out["gcd"] = dict(inputs=999 * 999, coprime=coprime, share=round(coprime / 999 ** 2, 4),
                      hidden_share=round(sum(e.answer == "001" for e in hidden) / len(hidden), 4),
                      six_over_pi_sq=round(6 / math.pi ** 2, 4))
    print("gcd", out["gcd"])
    (HERE / "floors.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
