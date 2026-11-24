"""wsp/baseline.py: the black box. Train the same template network the usual way, with gradient descent.

This is the opposite of what the agent is allowed to do, and it is here to measure the problem: a trained network
can reach the same accuracy, but nothing in its weights says how it gets there."""
import json
import random
import time

import torch
import torch.nn.functional as F

from wsp.evaluate import accuracy
from wsp.model import SimpleTransformer
from wsp.tasks import Problem

HIDDEN_SEED = 987654321                          # the same unseen seed the verifier uses


def batch_of(problem, rng, size):
    """Prompt plus answer as one sequence, with the loss counted on the answer characters only."""
    examples = [problem.make(rng) for _ in range(size)]
    ids = torch.tensor([problem.encode(e.prompt + e.answer) for e in examples], dtype=torch.long)
    start = len(examples[0].prompt)              # the first answer character sits right after "="
    return ids[:, :-1], ids[:, 1:], start - 1


def train(name="addition-five-digits", steps=40000, size=256, lr=1e-3, seed=0, log_every=500, out=None):
    torch.manual_seed(seed)
    problem, rng = Problem(name), random.Random(seed)
    sample = problem.make(random.Random(0))
    model = SimpleTransformer(len(problem.vocab), max_seq_len=max(len(sample.prompt) + problem.answer_len, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    curve, t0 = [], time.time()
    for step in range(1, steps + 1):
        model.train()
        x, y, a = batch_of(problem, rng, size)
        loss = F.cross_entropy(model(x)[:, a:].reshape(-1, len(problem.vocab)), y[:, a:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            acc, _ = accuracy(model, problem, problem.examples(2000, HIDDEN_SEED))   # the same 2,000 every time
            curve.append(dict(step=step, loss=round(loss.item(), 5), hidden=acc, seconds=round(time.time() - t0)))
            print(json.dumps(curve[-1]), flush=True)
            if out:
                torch.save(model.state_dict(), out)
    return model, curve
