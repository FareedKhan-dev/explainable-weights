"""results/blackbox_explain.py: explain the trained black box (addition) with the usual tools, and record what they say.

    <paper .venv python> blackbox_explain.py [checkpoint]

1. Hidden accuracy, with the paper's evaluator.
2. Switch-off: each of the 12 attention heads, each whole attention layer and each of the 3 MLP blocks, in turn.
3. Attributions for every answer digit of 200 hidden sums: plain gradients and integrated gradients, as the share each
   input character gets, with the integrated-gradients completeness error (attributions against the score change).
4. A look at the weights.
Writes results/blackbox_explain.json.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
os.environ.setdefault("WSP_PAPER", str(WS / "weight-space-programming-autoresearch"))
os.environ.setdefault("WSP_HF_HOME", str(Path(__file__).resolve().parents[1] / ".hfh"))
sys.path.insert(0, str(HERE.parent))
import torch  # noqa: E402

from wsp import networks, saliency, switchoff, verify  # noqa: E402
from wsp.model import SimpleTransformer  # noqa: E402
from wsp.tasks import Problem  # noqa: E402

NAME = "addition-five-digits"


def load(ckpt):
    """The trained network, read from a copy so a save in progress cannot be caught half written."""
    copy = HERE / "baseline_addition.read.pt"
    for _ in range(5):
        try:
            shutil.copy2(ckpt, copy)
            state = torch.load(copy, map_location="cpu")
            break
        except Exception:
            time.sleep(2)
    problem = Problem(NAME)
    sample = problem.examples(1, 0)[0]
    model = SimpleTransformer(len(problem.vocab), max_seq_len=max(len(sample.prompt) + problem.answer_len, 16))
    model.load_state_dict(state)
    return model.eval(), problem


def attributions(model, problem, n=200):
    """Mean share of attribution per (answer digit, input position), for both methods, and the IG completeness."""
    examples = problem.examples(n, verify.HIDDEN_SEED)
    width = len(examples[0].prompt) + problem.answer_len - 1
    shares = {m: torch.zeros(problem.answer_len, width) for m in ("gradient", "ig")}
    errors, gaps = [], []
    for e in examples:
        for k in range(problem.answer_len):
            ids = torch.tensor([problem.encode(e.prompt + e.answer[:k])], dtype=torch.long)
            target = problem.vocab.index(e.answer[k])
            g = saliency.gradient(model, ids, target)
            attr, gap = saliency.integrated_gradients(model, ids, target)
            shares["gradient"][k, :len(g)] += g / g.sum() / n
            shares["ig"][k, :len(attr)] += attr.abs() / attr.abs().sum() / n
            errors.append(abs(attr.sum().item() - gap) / max(abs(gap), 1e-6))
            gaps.append(gap)
    errors.sort()
    return ({m: [[round(x, 4) for x in row] for row in v.tolist()] for m, v in shares.items()},
            dict(median=errors[len(errors) // 2], p95=errors[int(0.95 * len(errors))], max=errors[-1], n=len(errors),
                 median_gap=sorted(gaps)[len(gaps) // 2]))


def main(ckpt=HERE / "baseline_addition.pt"):
    model, problem = load(ckpt)
    tasks, _ = networks.paper_code()
    task = tasks.get_task(NAME)
    acc, _ = verify.hidden(model, task)
    d_head = model.blocks[0].attn.d_head
    parts = [dict(name=f"layer {l + 1} head {h + 1}", path=f"blocks.{l}.attn.W_o",
                  channels=list(range(h * d_head, (h + 1) * d_head)))
             for l in range(len(model.blocks)) for h in range(model.blocks[l].attn.n_heads)]
    parts += [dict(name=f"layer {l + 1} attention", path=f"blocks.{l}.attn") for l in range(len(model.blocks))]
    parts += [dict(name=f"layer {l + 1} MLP", path=f"blocks.{l}.fc2") for l in range(len(model.blocks))]
    drops = switchoff.drops(model, task, parts)
    shares, completeness = attributions(model, problem)
    w = torch.cat([p.detach().flatten() for p in model.parameters()])
    out = dict(provisional=False, checkpoint=str(ckpt), weights=w.numel(), hidden=acc, switch_off=drops,
               attribution=dict(columns=[f"a{i}" for i in range(1, 6)] + ["+"] + [f"b{i}" for i in range(1, 6)] + ["="]
                                + [f"s{i}" for i in range(1, problem.answer_len)],          # s: answer digits already written
                                rows=[f"answer digit {i + 1}" for i in range(problem.answer_len)], **shares),
               ig_completeness=completeness,
               weights_look=dict(abs_mean=w.abs().mean().item(), share_below_0_01=(w.abs() < 0.01).float().mean().item(),
                                 sample_W_q=[[round(x, 3) for x in r] for r in model.blocks[0].attn.W_q.weight[:4, :6].tolist()]))
    (HERE / "blackbox_explain.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"hidden {acc}; weights {w.numel()}; IG completeness median error {completeness['median']:.4f}, "
          f"p95 {completeness['p95']:.4f}")
    for r in drops:
        print(f"  {r['part']:22s} off -> {r['off']:.4f} (drop {r['drop'] * 100:.1f} points)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
