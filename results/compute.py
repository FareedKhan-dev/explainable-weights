"""results/compute.py: how much arithmetic one answer takes, counted by PyTorch's own FLOP counter.

    <paper .venv python> compute.py

For the reversal prompt fnm_mod_wfd=, the paper's template (random weights, seeded) and Qwen3.7-Max's finished network
each write an 11-letter answer the way the evaluator does: one step per letter, the whole prefix through the network
each time. torch.utils.flop_counter.FlopCounterMode counts every multiplication and addition of the matrix products,
for the whole answer, for each step, and for each layer. Writes results/compute.json. CPU only, a few seconds.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WS = ROOT.parent
os.environ.setdefault("WSP_PAPER", str(WS / "weight-space-programming-autoresearch"))
os.environ.setdefault("WSP_HF_HOME", str(Path(__file__).resolve().parents[1] / ".hfh"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402
from wsp import config, networks, verify  # noqa: E402

PROMPT = "fnm_mod_wfd="
QWEN = ROOT / "runs" / "word-reversal-3x3-qwen37-max-s1" / "attempts" / "027_PosRouting_noW_o.py"


def steps(model, task):
    """Operations per generation step, and per layer at each step, with the step's chosen letter."""
    ids, out = task.encode(PROMPT), []
    for _ in range(task.answer_len):
        with FlopCounterMode(display=False) as count, torch.no_grad():
            logits = model(torch.tensor([ids]))
        per = {str(k): sum(v.values()) for k, v in count.get_flop_counts().items()}
        out.append(dict(total=count.get_total_flops(), modules=per))
        ids = ids + [int(logits[0, -1].argmax())]
    return out, task.decode(ids[-task.answer_len:])


def positions_seen(model):
    """Forward hooks on every weight matrix (nn.Linear): how many positions each one multiplies. Returns the counter
    and the hooks, which the caller removes."""
    seen, hooks = {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            def hook(m, inputs, output, name=name):
                seen[name] = seen.get(name, 0) + inputs[0].numel() // m.in_features
            hooks.append(mod.register_forward_hook(hook))
    return seen, hooks


def measure(model, task):
    seen, hooks = positions_seen(model)
    try:
        with FlopCounterMode(display=False) as count:
            answer = verify.answers(model, task, [PROMPT])[0]
    finally:
        for h in hooks:
            h.remove()
    per_step, stepped = steps(model, task)
    assert stepped == answer and sum(s["total"] for s in per_step) == count.get_total_flops()
    # each weight matrix costs two operations per weight per position it multiplies (one multiply, one add), and only
    # the weights that are not zero do any work
    mods = dict(model.named_modules())
    matrices = []
    for name, positions in seen.items():
        w = mods[name].weight
        matrices.append(dict(name=name, shape=list(w.shape), weights=w.numel(), nonzero=int((w != 0).sum()),
                             positions=positions, ops=2 * w.numel() * positions, nonzero_ops=2 * int((w != 0).sum()) * positions))
    flops = {str(k): v for k, v in count.get_flop_counts()["Global"].items()}
    linear = sum(m["ops"] for m in matrices)
    attention = flops.get("aten.bmm", 0)
    assert linear == flops.get("aten.mm", 0) + flops.get("aten.addmm", 0) and linear + attention == count.get_total_flops()
    return dict(weights=sum(p.numel() for p in model.parameters()),
                nonzero=int(sum((p != 0).sum() for p in model.parameters())), total=count.get_total_flops(),
                answer=answer, steps=per_step, matrices=matrices, linear_ops=linear, attention_ops=attention,
                nonzero_ops=sum(m["nonzero_ops"] for m in matrices))


def main():
    torch.manual_seed(0)                                   # the template's weights are random until an agent writes them
    task, template = networks.build(networks.load(config.EVOLVE / "interpretable_transformer.py"), "word-reversal-3x3")
    _, qwen = networks.build(networks.load(QWEN), "word-reversal-3x3")
    out = dict(provisional=False, prompt=PROMPT, answer_len=task.answer_len, template=measure(template, task),
               qwen=measure(qwen, task))
    assert out["qwen"]["answer"] == "wfd_mod_fnm"
    (HERE / "compute.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for k in ("template", "qwen"):
        m = out[k]
        print(f"{k:9s} {m['weights']:8,d} weights {m['nonzero']:8,d} not zero {m['total']:12,d} operations  answer {m['answer']}")


if __name__ == "__main__":
    main()
