"""results/blackbox_wall.py: what the trained black box's weights look like, for the post's first figure.

    <paper .venv python> blackbox_wall.py

Reads the step-35,000 checkpoint (results/baseline_addition.step35000.pt) and writes results/blackbox_wall.json:
the parameter count per tensor, the share of all weights whose size is under 0.01, and one whole tensor (the first
layer's query projection, 64 by 64) rounded to 3 decimals, drawn as the wall of numbers. CPU only, a few seconds.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from wsp.model import SimpleTransformer  # noqa: E402
from wsp.tasks import Problem  # noqa: E402

CKPT = HERE / "baseline_addition.step35000.pt"


def main():
    add = Problem("addition-five-digits")
    model = SimpleTransformer(len(add.vocab), max_seq_len=18)
    model.load_state_dict(torch.load(CKPT))
    state = model.state_dict()
    flat = torch.cat([p.flatten() for p in model.parameters()])
    wall_name = "blocks.0.attn.W_q.weight"
    wall = state[wall_name]
    out = dict(
        provisional=False,
        checkpoint=CKPT.name,
        weights=flat.numel(),
        share_below_0_01=(flat.abs() < 0.01).float().mean().item(),
        tensors={k: list(v.shape) for k, v in state.items()},
        wall=dict(name=wall_name, shape=list(wall.shape), min=wall.min().item(), max=wall.max().item(),
                  values=[[round(x, 3) for x in row] for row in wall.tolist()]),
    )
    (HERE / "blackbox_wall.json").write_text(json.dumps(out) + "\n", encoding="utf-8", newline="\n")
    print(out["weights"], round(out["share_below_0_01"], 4), wall_name, out["wall"]["shape"],
          round(out["wall"]["min"], 3), round(out["wall"]["max"], 3))


if __name__ == "__main__":
    main()
