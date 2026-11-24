"""wsp/switchoff.py: switch off one named part of a network at a time and measure what breaks.

A part is a module path (the output of that module is replaced with zeros, so it adds nothing to the residual
stream), or a module path with input channels (zeros on those channels only, which switches off one attention head
just before the output projection). If an explanation is true, every part it names should matter, and should
matter the way the explanation says."""
import contextlib

import torch

from wsp.verify import HIDDEN_N, HIDDEN_SEED, hidden


def zero_channels(x, channels):
    x = x.clone()
    x[..., channels] = 0
    return x


@contextlib.contextmanager
def zeroed(model, names):
    """Inside this block the named weights are zero, and they are restored exactly afterwards. This switches off a part
    that has no module of its own (a network that keeps its few weights on the top module, say). A name is a whole
    weight tensor, or (name, index) for one number in it."""
    params = dict(model.named_parameters())
    targets = [(n, None) if isinstance(n, str) else tuple(n) for n in names]
    saved = {n: params[n].detach().clone() for n, _ in targets}
    try:
        with torch.no_grad():
            for n, idx in targets:
                if idx is None:
                    params[n].zero_()
                else:
                    params[n].view(-1)[idx] = 0
        yield
    finally:
        with torch.no_grad():
            for n, value in saved.items():
                params[n].copy_(value)


@contextlib.contextmanager
def off(model, path=None, channels=None, params=None):
    """Inside this block the part at `path` (or only its input `channels`, or only the weights named in `params`)
    contributes nothing."""
    if params is not None:
        with zeroed(model, params):
            yield
        return
    module = model.get_submodule(path)
    if channels is None:
        handle = module.register_forward_hook(lambda m, args, out: torch.zeros_like(out))
    else:
        handle = module.register_forward_pre_hook(lambda m, args: (zero_channels(args[0], channels),) + tuple(args[1:]))
    try:
        yield
    finally:
        handle.remove()


def drops(model, task, parts, n=HIDDEN_N, seed=HIDDEN_SEED):
    """Hidden accuracy with each named part switched off in turn, beside the intact network's accuracy."""
    intact, _ = hidden(model, task, n, seed)
    rows = []
    for part in parts:
        with off(model, part.get("path"), part.get("channels"), part.get("params")):
            acc, _ = hidden(model, task, n, seed)
        rows.append(dict(part=part["name"], intact=intact, off=acc, drop=intact - acc))
    return rows
