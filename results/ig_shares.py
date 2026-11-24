"""results/ig_shares.py: integrated gradients for every character of one sum, from the trained adder (the black box).

    <paper .venv python> results/ig_shares.py

The same call the post's notebook cell makes (REPO_CELLS["ig"]): the step-35,000 checkpoint, the prompt and answer so
far 12345+67890=08023, and the units digit 5 as the target. Here every position's share of the attribution is kept,
not just the top three, for the g-saliency figure. CPU only, seconds. Writes results/ig_shares.json."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import torch  # noqa: E402
from wsp import saliency  # noqa: E402
from wsp.model import SimpleTransformer  # noqa: E402
from wsp.tasks import Problem  # noqa: E402

SEEN, TARGET = "12345+67890=08023", "5"


def main():
    add = Problem("addition-five-digits")
    black_box = SimpleTransformer(len(add.vocab), max_seq_len=18)
    black_box.load_state_dict(torch.load("results/baseline_addition.step35000.pt"))
    attr, gap = saliency.integrated_gradients(black_box.eval(), torch.tensor([add.encode(SEEN)]), add.stoi[TARGET])
    share = (attr.abs() / attr.abs().sum()).tolist()
    top = sorted(range(len(SEEN)), key=lambda k: -share[k])[:3]
    out = dict(provisional=False, seen=SEEN, target=TARGET, shares=[round(s, 4) for s in share],
               attr_sum=round(float(attr.sum()), 2), score_moved=round(float(gap), 2), top=top)
    assert [SEEN[k] for k in top] == ["0", "5", "3"] and top == [10, 4, 16], top   # the cell's printed top three
    (HERE / "ig_shares.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
