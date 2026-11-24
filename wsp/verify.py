"""wsp/verify.py: score a network on examples it has never seen, and on every input where every input fits.

The hidden sets are the paper's (experiments/nl-bottleneck/eval_program.py): 2,000 examples from a seed no run uses
for the generated problems, and for TREC 2,000 training questions outside the 2,000-question pool the agent saw. The
generated problems also get 20,000 extra examples (seed 555). GCD and dates are small enough to check all of their
inputs, and TREC is scored on its official 500-question test split, which no run ever loads."""
import calendar
import math
import random
from math import gcd

import torch

from wsp.networks import paper_code

HIDDEN_SEED, HIDDEN_N = 987654321, 2000         # the paper's unseen seed and size (experiments/evolve/heldout_fresh_seed.py)
EXTRA_SEED, EXTRA_N = 555, 20000                # the paper's extra set, to catch rare errors 2,000 examples miss
LABELS = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]
ALLOWED = set("abcdefghijklmnopqrstuvwxyz '")


def hidden(model, task, n=HIDDEN_N, seed=HIDDEN_SEED):
    """Accuracy on fresh examples, with the paper's own evaluator."""
    _, evaluator = paper_code()
    acc, details = evaluator.evaluate(model, task, n_samples=n, seed=seed, device="cpu")
    return acc, details


@torch.no_grad()
def answers(model, task, prompts, batch=2048):
    """Greedy answers for many prompts at once, token by token, as the evaluator generates them."""
    _, evaluator = paper_code()
    out = []
    for i in range(0, len(prompts), batch):
        ids = torch.tensor([task.encode(p) for p in prompts[i:i + batch]], dtype=torch.long)
        gen = evaluator.autoregressive_generate(model, ids, task.answer_len)
        out += [task.decode(row.tolist()) for row in gen]
    return out


def exhaustive_gcd(model, task):
    """Every pair from 1 to 999: the number of wrong answers, and the first few of them."""
    pairs = [(a, b) for a in range(1, 1000) for b in range(1, 1000)]
    got = answers(model, task, [f"{a:03d},{b:03d}=" for a, b in pairs])
    wrong = [(p, g) for p, g in zip(pairs, got) if g != f"{gcd(*p):03d}"]
    return len(pairs), len(wrong), wrong[:5]


def all_dates(task):
    """Every date from 1900 to 2099 in all nine formats the task writes, as (prompt, answer) pairs."""
    out = []
    for y in range(1900, 2100):
        for m in range(1, 13):
            name, abbr = task.MONTHS[m - 1], task.MONTHS[m - 1][:3]
            for d in range(1, calendar.monthrange(y, m)[1] + 1):
                texts = [f"{name} {task.ordinal(d)} {y}", f"{name} {d}, {y}", f"{d} {name} {y}",
                         f"the {task.ordinal(d)} of {name}, {y}", f"{abbr}. {d}, {y}", f"{d}-{abbr}-{y}",
                         f"{m}/{d}/{y}", f"{y}.{m:02d}.{d:02d}"] + ([f"{m:02d}/{d:02d}/{y % 100:02d}"] if y >= 2000 else [])
                out += [(t.ljust(task.TEXT_LEN, "_") + "=", f"{y:04d}-{m:02d}-{d:02d}") for t in texts]
    return out


def exhaustive_dates(model, task):
    pairs = all_dates(task)
    got = answers(model, task, [p for p, _ in pairs])
    wrong = [(p, a, g) for (p, a), g in zip(pairs, got) if g != a]
    return len(pairs), len(wrong), wrong[:5]


BENCHMARK = {"sentiment-sst2": "validation", "paraphrase-mrpc": "validation", "nli-snli": "test", "trec-coarse": "test"}
DATASETS = tuple(BENCHMARK)                      # the four problems built from a real dataset (the rest are generated)


def dataset_rows(name, split):
    """(prompt, label) for every example of a split, cleaned exactly as the task cleans its pool
    (experiments/nl-bottleneck/eval_program.py, outside_pool). SNLI drops pairs with no gold label."""
    from datasets import load_dataset
    tasks, _ = paper_code()
    c = tasks._clean_text
    if name == "sentiment-sst2":
        rows = []
        for e in load_dataset("stanfordnlp/sst2", split=split):
            t = " ".join("".join(ch if ch in ALLOWED else " " for ch in e["sentence"].lower()).split())
            if t:
                rows.append((t[:80].ljust(80, "_") + "=", str(e["label"])))
        return rows
    if name == "paraphrase-mrpc":
        return [(f"{c(e['sentence1'], ALLOWED, 60)}|{c(e['sentence2'], ALLOWED, 60)}=", str(e["label"]))
                for e in load_dataset("nyu-mll/glue", "mrpc", split=split)]
    if name == "nli-snli":
        return [(f"{c(e['premise'], ALLOWED, 60)}|{c(e['hypothesis'], ALLOWED, 60)}=", str(e["label"]))
                for e in load_dataset("stanfordnlp/snli", split=split) if e["label"] in (0, 1, 2)]
    if name == "trec-coarse":
        return [(c(e["text"], ALLOWED, 80) + "=", str(e["coarse_label"]))
                for e in load_dataset("CogComp/trec", split=split, revision="refs/convert/parquet")]
    raise ValueError(f"{name} is not built from a dataset")


def dataset_hidden(name, n=HIDDEN_N, seed=1):
    """The paper's hidden set for a dataset problem: training examples outside the agent's pool (the first 2,000),
    sampled with seed 1. MRPC has only 1,668 left, so it gets all of them."""
    rest = dataset_rows(name, "train")[2000:]
    return random.Random(seed).sample(rest, min(n, len(rest)))


def seen_by_agent(name="trec-coarse"):
    """Every example the agent's evaluator could draw: the task's pool, the first 2,000 training examples."""
    return {p for p, _ in dataset_rows(name, "train")[:2000]}


def trec_rows(split):
    return dataset_rows("trec-coarse", split)


def trec_hidden(n=HIDDEN_N, seed=1):
    return dataset_hidden("trec-coarse", n, seed)


def interval(accuracy, n, z=1.96):
    """Wilson's 95% interval for an accuracy measured on n examples."""
    if not n:
        return None
    centre = (accuracy + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def score_pairs(model, task, pairs):
    """Accuracy on fixed (prompt, answer) pairs, judged by the task's own rule."""
    got = answers(model, task, [p for p, _ in pairs])
    return sum(task.is_correct(g, a) for g, (_, a) in zip(got, pairs)) / len(pairs), got


def benchmark(model, task, name):
    """Accuracy on the dataset's own labelled held-out split (SST-2 and MRPC validation, SNLI and TREC test), with a
    confusion count per (true, predicted)."""
    rows = dataset_rows(name, BENCHMARK[name])
    acc, got = score_pairs(model, task, rows)
    confusion = {}
    for (_, t), g in zip(rows, got):
        confusion[(t, g)] = confusion.get((t, g), 0) + 1
    return acc, confusion, len(rows)


def trec_official(model, task):
    """Accuracy on TREC's official test split (500 questions), with a confusion count per (true, predicted)."""
    return benchmark(model, task, "trec-coarse")
