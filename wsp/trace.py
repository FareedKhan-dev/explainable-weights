"""wsp/trace.py: follow one input through a network and record what each named part did for each answer token.

For every generated token it keeps the output of each named part at the newest position, the attention of each
head that is built the template's way (where each head looks), and the margin by which the chosen token won. A
network's own channel names (the constants in its file) turn those numbers into words."""
import math

import torch


def head_weights(module, x):
    """Attention weights of the newest position, per head, for a template-style attention module."""
    B, T, D = x.shape
    H, dh = module.n_heads, module.d_head
    q = module.W_q(x).view(B, T, H, dh).transpose(1, 2)
    k = module.W_k(x).view(B, T, H, dh).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    return scores.masked_fill(mask, float("-inf")).softmax(dim=-1)[0, :, -1]


@torch.no_grad()
def trace(model, task, prompt, parts):
    """One record per answer token: the token, its winning margin, each part's output and each head's focus."""
    seen, hooks = {}, []
    for name, path in parts.items():
        module = model.get_submodule(path)
        # a block the network applies many times (a loop of tied weights) keeps one entry per pass
        hooks.append(module.register_forward_hook(lambda m, a, out, name=name: seen.setdefault(name, []).append(out[0, -1].clone())))
        if all(hasattr(module, w) for w in ("W_q", "W_k", "n_heads", "d_head")):
            hooks.append(module.register_forward_pre_hook(
                lambda m, a, name=name: seen.__setitem__(name + ".focus", head_weights(m, a[0]))))
    ids, records = torch.tensor([task.encode(prompt)], dtype=torch.long), []
    try:
        for _ in range(task.answer_len):
            seen.clear()
            top = torch.topk(model(ids)[0, -1], 2)
            token = int(top.indices[0])
            records.append(dict(token=task.itos[token], margin=float(top.values[0] - top.values[1]), parts=dict(seen)))
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
    finally:
        for h in hooks:
            h.remove()
    return records
