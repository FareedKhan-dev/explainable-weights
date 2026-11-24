"""wsp/audit.py: check that a network was written, not trained, and that its weights do not come from the answers.

Two checks, both from the paper (experiments/evolve/audit_runs.py and audit_data_use.py):
  scan()     looks for training, label reading and file reading in the code the agent wrote,
  data_use() builds the network four times, the last two on altered task data, and compares the weights.
The scan reads every line the agent wrote. It stops only at the evaluator's own top-level main block, so a check
hidden inside a function (the paper's TREC run e22 had one) is still read."""
import contextlib
import io
import random
import re
import tokenize

import torch

from wsp import networks

PATTERNS = {
    "training": [r"torch\.optim", r"\.backward\(", r"optimizer", r"\.fit\(", r"lstsq", r"\bpinv\b", r"requires_grad_\(True\)",
                 r"autograd\.grad"],
    "labels": [r"generate_examples\(", r"_load_pool\(", r"\bevaluate\(", r"\.answer\b", r"load_dataset\("],
    "test_split": [r"""split\s*=\s*['"]test""", r"test\.parquet", r"""\[['"]test['"]\]"""],
    "files": [r"\bopen\(", r"np\.load\(", r"torch\.load\(", r"pickle\.", r"read_csv\("],
}
MAIN = re.compile(r"^if __name__ == ['\"]__main__['\"]:", re.M)


def code_only(source):
    """The agent's code with comments and strings blanked, cut before the evaluator's top-level main block."""
    cut = [m.start() for m in MAIN.finditer(source)]
    source = source[:cut[-1]] if cut else source
    chars, lines = list(source), source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                a, b = starts[tok.start[0] - 1] + tok.start[1], starts[tok.end[0] - 1] + tok.end[1]
                chars[a:b] = [c if c == "\n" else " " for c in chars[a:b]]
    except (tokenize.TokenError, IndentationError):
        pass
    return "".join(chars)


def scan(source, template):
    """Patterns that occur more often in this file's code than in the untouched template."""
    mine, base = code_only(source), code_only(template)
    found = {}
    for kind, pats in PATTERNS.items():
        hits = [p for p in pats if len(re.findall(p, mine)) > len(re.findall(p, base))]
        if hits:
            found[kind] = hits
    return found


def state(module, task):
    with contextlib.redirect_stdout(io.StringIO()):
        model = module.build_model(task)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def same(a, b):
    return a.keys() == b.keys() and all(a[k].shape == b[k].shape and torch.equal(a[k], b[k]) for k in a)


def random_answers(examples, seed):
    """Each answer replaced by a random string of the same length, over the answer characters seen."""
    rng = random.Random(seed)
    chars = sorted({c for e in examples for c in e.answer}) or ["0"]
    return [type(e)(prompt=e.prompt, answer="".join(rng.choice(chars) for _ in e.answer)) for e in examples]


def second_half(cls):
    """The task's own pool loader, run on the second half of the training split instead of the first."""
    import datasets
    real_load = datasets.load_dataset

    def half(*args, **kwargs):
        ds = real_load(*args, **kwargs)
        return ds.select(range(len(ds) // 2, len(ds)))
    datasets.load_dataset = half
    try:
        return list(cls._load_pool())
    finally:
        datasets.load_dataset = real_load


def swap_labels(cls, real_gen):
    """Build C's data: every generated answer, and every label in a dataset task's pool, replaced at random."""
    cls.generate_examples = lambda self, n, seed=None: random_answers(real_gen(self, n, seed=seed), 3)
    labels = None
    if hasattr(cls, "_load_pool"):
        pool = list(cls._load_pool())
        labels, rng = sorted({e[-1] for e in pool}), random.Random(4)
        cls._pool = [e[:-1] + (rng.choice(labels),) for e in pool]
    return labels


def swap_inputs(cls, real_gen, labels):
    """Build B's data: other inputs (another seed, or the second half of the training split), random answers."""
    cls.generate_examples = lambda self, n, seed=None: random_answers(real_gen(self, n, seed=(seed or 0) + 7919), 1)
    if hasattr(cls, "_load_pool"):
        cls._pool, rng = None, random.Random(2)
        cls._pool = [e[:-1] + (rng.choice(labels),) for e in second_half(cls)]


def data_use(path, task_name):
    """independent, label_dependent, input_dependent, nondeterministic or error, as the paper defines them."""
    tasks, evaluator = networks.paper_code()
    module, task = networks.load(path), tasks.get_task(task_name)
    cls, real_eval, real_gen = type(task), evaluator.evaluate, type(task).generate_examples
    real_pool = cls.__dict__.get("_pool") if hasattr(cls, "_pool") else None
    try:
        a, a2 = state(module, task), state(module, task)          # builds A and A2: the file as it is, twice
        if not same(a, a2):
            return "nondeterministic"
        evaluator.evaluate = module.evaluate = lambda *x, **k: (1.0, [])       # every score looks perfect
        labels = swap_labels(cls, real_gen)                       # build C: the same inputs, random answers
        if not same(a, state(module, task)):
            return "label_dependent"
        swap_inputs(cls, real_gen, labels)                        # build B: other inputs, random answers
        return "independent" if same(a, state(module, task)) else "input_dependent"
    except Exception as e:
        return f"error: {type(e).__name__}: {str(e)[:120]}"
    finally:
        evaluator.evaluate, cls.generate_examples = real_eval, real_gen
        if hasattr(cls, "_pool"):
            cls._pool = real_pool
