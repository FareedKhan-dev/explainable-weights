"""results/template_attention.py: how the paper's random template shares out attention at the last prompt position.

Built exactly as the post's computation cell builds the template (torch.manual_seed(0) right before the build), it
runs the running prompt and records, for every head of the first block, the scores of the last position (the `=`)
against every position so far and the softmax shares they turn into. The attention figure of the template section
draws head 1. Nothing is trained and no model is called.

    <paper .venv python> results/template_attention.py
"""
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WS = ROOT.parent
os.environ.setdefault("WSP_PAPER", str(WS / "weight-space-programming-autoresearch"))
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from wsp import config, networks  # noqa: E402

PROMPT = "fnm_mod_wfd="


def main():
    os.chdir(ROOT)
    torch.manual_seed(0)                                         # as in the post's cell
    task, template = networks.build(networks.load(config.EVOLVE / "interpretable_transformer.py"), "word-reversal-3x3")
    attn = template.blocks[0].attn
    seen = {}
    hook = attn.register_forward_hook(lambda mod, inp, out: seen.setdefault("x", inp[0].detach()))
    with torch.no_grad():
        template(torch.tensor([task.encode(PROMPT)]))
    hook.remove()
    x = seen["x"]                                                # the first block's LayerNorm output, (1, T, d_model)
    B, T, D = x.shape
    H, dh = attn.n_heads, attn.d_head
    with torch.no_grad():
        q = attn.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = attn.W_k(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)      # (1, H, T, T)
    last = T - 1
    heads = []
    for h in range(H):
        row = scores[0, h, last, : last + 1]
        share = row.softmax(-1)
        assert abs(float(share.sum()) - 1) < 1e-5
        heads.append(dict(head=h + 1, scores=[round(float(v), 4) for v in row], shares=[round(float(v), 4) for v in share]))
    out = dict(provisional=False, prompt=PROMPT, tokens=list(PROMPT), position=last, layer=1, heads=heads)
    (HERE / "template_attention.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for hd in heads:
        print(f"head {hd['head']}: shares {hd['shares']}  max {max(hd['shares']):.4f} min {min(hd['shares']):.4f}")


if __name__ == "__main__":
    main()
