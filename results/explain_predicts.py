"""results/explain_predicts.py: an explanation is faithful if, written as a program, it predicts what the network does.

    <paper .venv python> explain_predicts.py [n]

EXPLORATORY (amendment 2): not one of the sealed hypotheses.

The network is the paper's 332-weight GCD pick (experiments/nl-bottleneck/originals/gcd-three-digits.py). Its own
200-word description says: six parser heads each read one digit, times its place value; Euclid runs as twelve tied
rounds of four triangular-wave folds (scales 64, 16, 4, 1), the last fold also swapping A and B; the answer is A + B.
That description becomes euclid() below, a dozen lines of plain Python with knobs for each part. For every hidden
input, the program's prediction is compared with the network's own output: intact, with each parser head switched
off, and with each fold switched off. A switch-off usually leaves the answer right (a missing digit rarely changes the
gcd), so the accuracy drop says little; the prediction says exactly which answers change, and to what.
Writes results/explain_predicts.json.
"""
import contextlib
import json
import os
import sys
from math import gcd
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
os.environ.setdefault("WSP_PAPER", str(WS / "weight-space-programming-autoresearch"))
os.environ.setdefault("WSP_HF_HOME", str(Path(__file__).resolve().parents[1] / ".hfh"))
sys.path.insert(0, str(HERE.parent))
from wsp import config, networks, switchoff, verify  # noqa: E402
from wsp.programs import FOLDS, ROUTES, euclid, pointer, readout, routing  # noqa: E402,F401

NET = config.PAPER / "experiments" / "nl-bottleneck" / "originals" / "gcd-three-digits.py"
DIGITS = ["A hundreds", "A tens", "A units", "B hundreds", "B tens", "B units"]








def reversal(n):
    """Intact, and each of the three weights switched off alone, then all three: the program against the network."""
    tasks, _ = networks.paper_code()
    net = config.PAPER / "experiments" / "nl-bottleneck" / "originals" / "word-reversal-3x3.py"
    task, model = networks.build(networks.load(net), "word-reversal-3x3")
    assert task.stoi["_"] == 26 and task.stoi["="] == 27 and task.vocab_size == 28
    ex = tasks.get_task("word-reversal-3x3").generate_examples(n, seed=verify.HIDDEN_SEED)
    prompts = [e.prompt for e in ex]
    weights = model.reads.detach().tolist()
    rows = []
    for name, off in [("intact", [])] + [(f"weight {k} ({weights[k]:+g}) off", [("reads", k)]) for k in range(3)] \
            + [("all three weights off", [("reads", k) for k in range(3)])]:
        reads = [0.0 if ("reads", k) in off else w for k, w in enumerate(weights)]
        with switchoff.off(model, params=off) if off else contextlib.nullcontext():
            got = verify.answers(model, task, prompts)
        predicted = [task.decode(pointer(task.encode(q), reads)) for q in prompts]
        truth = [e.answer for e in ex]
        rows.append(dict(case=name, n=n, accuracy=sum(g == t for g, t in zip(got, truth)) / n,
                         predicted_correctly=sum(g == p for g, p in zip(got, predicted)) / n,
                         changed=sum(g != t for g, t in zip(got, truth)),
                         changed_and_predicted=sum(g != t and g == p for g, t, p in zip(got, truth, predicted)),
                         example=dict(prompt=prompts[0], network=got[0], program=predicted[0]),
                         misses=[dict(prompt=q, network=g, program=p) for q, g, p in zip(prompts, got, predicted)
                                 if g != p][:5]))
        r = rows[-1]
        print(f"reversal {name:26s} accuracy {r['accuracy']:.4f}  answers changed {r['changed']:4d}  program predicts "
              f"the network on {r['predicted_correctly']:.2%}  e.g. {r['example']['prompt']} -> {r['example']['network']}",
              flush=True)
    return dict(network=str(net), weights=weights, rows=rows)


QWEN_REVERSAL = (Path(__file__).resolve().parents[1] / "runs" / "word-reversal-3x3-qwen37-max-s1" / "attempts"
                 / "027_PosRouting_noW_o.py")




def qwen_reversal(n):
    """Intact, one route switched off (each of the 11 in turn), and the copy (all values) switched off."""
    tasks, _ = networks.paper_code()
    task, model = networks.build(networks.load(QWEN_REVERSAL), "word-reversal-3x3")
    ex = tasks.get_task("word-reversal-3x3").generate_examples(n, seed=verify.HIDDEN_SEED)
    prompts, truth = [e.prompt for e in ex], [e.answer for e in ex]
    pos = dict(model.named_parameters())["pos_emb.weight"]
    cases = [("intact", [], ())]
    cases += [(f"route at answer position {p} off", [("pos_emb.weight", p * pos.shape[1] + 39 + s)], (p,))
              for p, s in ROUTES.items()]
    cases += [("copy off (all values zero)", ["blocks.0.attn.W_v.weight"], ("values",))]
    rows = []
    for name, params, knob in cases:
        with switchoff.off(model, params=params) if params else contextlib.nullcontext():
            got = verify.answers(model, task, prompts)
        predicted = [task.decode(routing(task.encode(q), knob)) for q in prompts]
        rows.append(dict(case=name, n=n, accuracy=sum(g == t for g, t in zip(got, truth)) / n,
                         predicted_correctly=sum(g == q for g, q in zip(got, predicted)) / n,
                         changed=sum(g != t for g, t in zip(got, truth)),
                         example=dict(prompt=prompts[0], network=got[0], program=predicted[0]),
                         misses=[dict(prompt=a, network=g, program=q) for a, g, q in zip(prompts, got, predicted)
                                 if g != q][:5]))
        r = rows[-1]
        print(f"qwen reversal {name:36s} accuracy {r['accuracy']:.4f}  changed {r['changed']:4d}  program predicts "
              f"{r['predicted_correctly']:.2%}  e.g. {r['example']['prompt']} -> {r['example']['network']}", flush=True)
    return dict(network=str(QWEN_REVERSAL), weights=networks.weights(model), rows=rows)


def main(n=2000):
    tasks, _ = networks.paper_code()
    task, model = networks.build(networks.load(NET), "gcd-three-digits")
    ex = tasks.get_task("gcd-three-digits").generate_examples(n, seed=verify.HIDDEN_SEED)
    pairs = [tuple(int(x) for x in e.prompt.rstrip("=").split(",")) for e in ex]
    prompts = [e.prompt for e in ex]
    cases = [("intact", {}, {})]
    cases += [(f"parser head off: {DIGITS[h]}", dict(path="parser.W_o", channels=[h]), dict(head_off=h)) for h in range(6)]
    cases += [(f"fold off: scale {s}" + (" (and the swap)" if i == 3 else ""), dict(path=f"folds.{i}.fc2"),
               dict(fold_off=i)) for i, (s, _) in enumerate(FOLDS)]
    rows = []
    for name, part, knobs in cases:
        if part:
            with switchoff.off(model, **part):
                got = verify.answers(model, task, prompts)
        else:
            got = verify.answers(model, task, prompts)
        predicted = [euclid(a, b, **knobs) for a, b in pairs]
        truth = [f"{gcd(a, b):03d}" for a, b in pairs]
        rows.append(dict(case=name, n=n, accuracy=sum(g == t for g, t in zip(got, truth)) / n,
                         predicted_correctly=sum(g == p for g, p in zip(got, predicted)) / n,
                         changed=sum(g != t for g, t in zip(got, truth)),
                         changed_and_predicted=sum(g != t and g == p for g, t, p in zip(got, truth, predicted)),
                         misses=[dict(prompt=q, network=g, program=p) for q, g, p in zip(prompts, got, predicted)
                                 if g != p][:5]))
        r = rows[-1]
        print(f"{name:34s} accuracy {r['accuracy']:.4f}  answers changed {r['changed']:4d}  "
              f"program predicts the network on {r['predicted_correctly']:.2%} ({r['changed_and_predicted']} of the changed)",
              flush=True)
    (HERE / "explain_predicts.json").write_text(json.dumps(dict(provisional=False, exploratory=True, network=str(NET),
                                                                rows=rows, reversal=reversal(n),
                                                                qwen_reversal=qwen_reversal(n)), indent=1) + "\n",
                                                encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(*[int(x) for x in sys.argv[1:2]])
