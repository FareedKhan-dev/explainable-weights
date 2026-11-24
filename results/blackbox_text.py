"""results/blackbox_text.py: real-world black boxes. The paper's template network trained the usual way on TREC and on
sentiment, then explained with the usual tools. EXPLORATORY (amendment 6, item 8).

    <paper .venv python> blackbox_text.py [trec-coarse|sentiment-sst2 ...] [--steps N]

Standard practice throughout:
  - training data: every training example except the hidden sample, so no scored example was trained on; a tenth of
    it is held out as validation, and the checkpoint with the best validation accuracy is kept (never chosen on a
    scored set);
  - scores: the paper's hidden set and the dataset's own labelled split, with 95% Wilson intervals and the
    majority-class floor;
  - explanations: integrated gradients per character, summed into words; the words each class leans on most; and the
    ERASER comprehensiveness test (DeYoung et al., 2020): delete the k words the attribution ranks highest and count
    how often the prediction changes, against deleting k random words.
Writes results/blackbox_<task>.json and results/blackbox_<task>.pt.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
os.environ.setdefault("WSP_PAPER", str(WS / "weight-space-programming-autoresearch"))
os.environ.setdefault("WSP_HF_HOME", str(Path(__file__).resolve().parents[1] / ".hfh"))
sys.path.insert(0, str(HERE.parent))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from wsp import networks, saliency, verify  # noqa: E402
from wsp.model import SimpleTransformer  # noqa: E402

NAMES = {"trec-coarse": ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"], "sentiment-sst2": ["negative", "positive"],
         "paraphrase-mrpc": ["different", "paraphrase"], "nli-snli": ["entailment", "neutral", "contradiction"]}
STEPS = {"trec-coarse": 2000, "sentiment-sst2": 5000, "paraphrase-mrpc": 1000, "nli-snli": 5000}


def data(name, seed=0):
    """Training rows (all but the hidden sample), a validation tenth of them, the hidden set and the benchmark split."""
    train = verify.dataset_rows(name, "train")
    hidden = verify.dataset_hidden(name)
    used = set(hidden)
    rest = [r for r in train if r not in used]
    random.Random(seed).shuffle(rest)
    cut = len(rest) // 10
    return rest[cut:], rest[:cut], hidden, verify.dataset_rows(name, verify.BENCHMARK[name])


def batches(rows, task, size, rng):
    order = list(range(len(rows)))
    rng.shuffle(order)
    for i in range(0, len(order), size):
        chunk = [rows[j] for j in order[i:i + size]]
        x = torch.tensor([task.encode(p) for p, _ in chunk], dtype=torch.long)
        y = torch.tensor([task.stoi[a] for _, a in chunk], dtype=torch.long)
        yield x, y


def train(name, steps, seed=0, lr=1e-3, size=128):
    tasks, _ = networks.paper_code()
    task = tasks.get_task(name)
    tr, val, hidden, bench = data(name, seed)
    torch.manual_seed(seed)
    model = SimpleTransformer(task.vocab_size, max_seq_len=task.prompt_len + task.answer_len)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng, step, best, curve = random.Random(seed), 0, (-1.0, None), []
    while step < steps:
        for x, y in batches(tr, task, size, rng):
            model.train()
            loss = F.cross_entropy(model(x)[:, -1], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % max(1, steps // 20) == 0 or step == steps:
                model.eval()
                acc, _ = verify.score_pairs(model, task, val)
                curve.append(dict(step=step, loss=round(loss.item(), 4), validation=acc))
                print(json.dumps(curve[-1]), flush=True)
                if acc > best[0]:
                    best = (acc, {k: v.detach().clone() for k, v in model.state_dict().items()})
            if step >= steps:
                break
    model.load_state_dict(best[1])
    return model.eval(), task, dict(train=len(tr), validation=len(val), hidden=hidden, bench=bench), curve


def words(prompt):
    """(start, end) of each word in the cleaned text, before the padding."""
    spans, i, text = [], 0, prompt.rstrip("=").rstrip("_")
    for w in text.split(" "):
        if w:
            spans.append((i, i + len(w)))
        i += len(w) + 1
    return spans


def explain(model, task, rows, name, n=200, k=(1, 3), seed=0):
    """Word attributions per class, and comprehensiveness against random deletion."""
    rng, per_class, flips = random.Random(seed), defaultdict(lambda: defaultdict(list)), {f"top{j}": 0 for j in k}
    flips.update({f"random{j}": 0 for j in k})
    counted = 0
    for prompt, _ in rows[:n]:
        ids = torch.tensor([task.encode(prompt)], dtype=torch.long)
        with torch.no_grad():
            pred = model(ids)[0, -1].argmax().item()
        attr, _ = saliency.integrated_gradients(model, ids, pred)
        spans = words(prompt)
        if not spans:
            continue
        counted += 1
        score = [(attr[a:b].sum().item(), a, b) for a, b in spans]
        for s, a, b in score:
            per_class[task.itos[pred]][prompt[a:b]].append(s)
        ranked = sorted(score, reverse=True)
        for j in k:
            for kind, chosen in (("top", ranked[:j]), ("random", rng.sample(score, min(j, len(score))))):
                text = list(prompt)
                for _, a, b in chosen:
                    text[a:b] = " " * (b - a)                 # the word becomes spaces, as unknown characters do
                with torch.no_grad():
                    new = model(torch.tensor([task.encode("".join(text))], dtype=torch.long))[0, -1].argmax().item()
                flips[f"{kind}{j}"] += new != pred
    labels = NAMES[name]
    top_words = {labels[int(c)] if c.isdigit() else c: sorted(((w, sum(v) / len(v), len(v)) for w, v in ws.items()
                                                               if len(v) >= 3), key=lambda t: -t[1])[:8]
                 for c, ws in per_class.items()}
    return dict(examples=counted, comprehensiveness={key: v / counted for key, v in flips.items()}, top_words=top_words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=["trec-coarse", "sentiment-sst2"])
    ap.add_argument("--steps", type=int, default=0, help="0: the STEPS table")
    ap.add_argument("--tag", default="", help="a suffix for the output files, so a longer run keeps the shorter one")
    a = ap.parse_args()
    for name in a.names:
        steps = a.steps or STEPS[name]
        model, task, sets, curve = train(name, steps)
        hid, _ = verify.score_pairs(model, task, sets["hidden"])
        bench, confusion, n = verify.benchmark(model, task, name)
        out = dict(provisional=False, exploratory=True, task=name, weights=networks.weights(model), steps=steps,
                   train_examples=sets["train"], validation_examples=sets["validation"], curve=curve,
                   hidden=dict(accuracy=hid, n=len(sets["hidden"]), ci95=verify.interval(hid, len(sets["hidden"]))),
                   benchmark=dict(split=verify.BENCHMARK[name], accuracy=bench, n=n, ci95=verify.interval(bench, n)),
                   explanation=explain(model, task, sets["hidden"], name))
        stem = f"blackbox_{name}" + (f"_{a.tag}" if a.tag else "")
        torch.save(model.state_dict(), HERE / f"{stem}.pt")
        (HERE / f"{stem}.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
        print(f"{name}: {out['weights']:,} weights; hidden {hid:.4f}; {out['benchmark']['split']} {bench:.4f} on {n}; "
              f"comprehensiveness {out['explanation']['comprehensiveness']}", flush=True)


if __name__ == "__main__":
    main()
