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


def train_step(model, opt, sched, problem, rng, size):
    model.train()
    x, y, a = batch_of(problem, rng, size)

    # The loss counts the answer characters only
    scores = model(x)[:, a:].reshape(-1, len(problem.vocab))
    loss = F.cross_entropy(scores, y[:, a:].reshape(-1))

    # Nudge every weight against its gradient
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    return loss


def train(name="addition-five-digits", steps=40000, size=256, lr=1e-3, seed=0,
          log_every=500, out=None):
    torch.manual_seed(seed)
    problem = Problem(name)
    rng = random.Random(seed)

    # The template, sized for prompt plus answer
    sample = problem.make(random.Random(0))
    seq_len = max(len(sample.prompt) + problem.answer_len, 16)
    model = SimpleTransformer(len(problem.vocab), max_seq_len=seq_len)

    # The optimizer and its schedule
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    # The training loop
    curve = []
    t0 = time.time()
    for step in range(1, steps + 1):
        loss = train_step(model, opt, sched, problem, rng, size)
        if step % log_every == 0 or step == steps:
            curve.append(checkpoint(model, problem, step, loss, t0, out))
    return model, curve


def checkpoint(model, problem, step, loss, t0, out):
    """Score the network on the same 2,000 hidden sums, print the point and save the weights."""
    acc, _ = accuracy(model, problem, problem.examples(2000, HIDDEN_SEED))
    point = dict(step=step, loss=round(loss.item(), 5), hidden=acc, seconds=round(time.time() - t0))
    print(json.dumps(point), flush=True)
    if out:
        torch.save(model.state_dict(), out)
    return point
