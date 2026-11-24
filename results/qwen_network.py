"""results/qwen_network.py: everything the figures show about the one network Qwen3.7-Max got right.

    <paper .venv python> qwen_network.py

Reads the network the agent wrote (word reversal, run 1, attempt 27), the paper's template for the same task, the
run's attempt log and the analysis in qwen_runs.json. Writes results/qwen_network.json with:
  network     the six weight tensors, each with its shape and every non-zero cell (row, column, value)
  template    the template's tensors and their exact weight counts
  routes      the routing table, read back from the position embeddings
  attention   the real scores and softmax weights of the one head on fnm_mod_wfd=, at answer positions 11 to 21
  spread14    the letter scores at position 14 when its route is switched off (the program's rule)
  staircase   every attempt of the run, in order, with exact weights for the files that built
  run2        the second run's pick and its first perfect attempt
  hero_code   the real lines of write_weights the hero types, each checked against the file
  visible     the 200 visible prompts (seed 0, the evaluator's default) with the network's answers
Nothing here calls a model.
"""
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
os.environ.setdefault("WSP_PAPER", str(WS / "weight-space-programming-autoresearch"))
os.environ.setdefault("WSP_HF_HOME", str(Path(__file__).resolve().parents[1] / ".hfh"))
sys.path.insert(0, str(HERE.parent))
import torch  # noqa: E402

from wsp import config, networks, verify  # noqa: E402

TASK = "word-reversal-3x3"
RUN = Path(__file__).resolve().parents[1] / "runs" / "word-reversal-3x3-qwen37-max-s1"
NET = RUN / "attempts" / "027_PosRouting_noW_o.py"
TEMPLATE = config.EVOLVE / "interpretable_transformer.py"
PROMPT = "fnm_mod_wfd="
HERO_LINES = [187, 203, 212, 217, 227, 233, 239, 272]       # the lines of write_weights that set the weights
LABELS = {"token_emb.weight": "token", "pos_emb.weight": "position", "blocks.0.attn.W_q.weight": "W_q",
          "blocks.0.attn.W_k.weight": "W_k", "blocks.0.attn.W_v.weight": "W_v", "head.weight": "head"}


def tensors(model):
    out = []
    for name, p in model.named_parameters():
        w = p.detach()
        cells = [[int(r), int(c), round(float(w[r, c]), 6)] for r, c in torch.nonzero(w).tolist()] if w.dim() == 2 else []
        out.append(dict(name=name, label=LABELS.get(name, name), shape=list(w.shape), count=w.numel(),
                        nonzero=int((w != 0).sum()), cells=cells))
    return out


@torch.no_grad()
def attention(model, task, prompt, answer):
    """Scores q.k/sqrt(d) and softmax weights of the single head, for the queries at answer positions 11 to 21."""
    ids = torch.tensor([task.encode(prompt + answer[:-1])], dtype=torch.long)    # 22 tokens feed 11 predictions
    T = ids.shape[1]
    x = model.token_emb(ids) + model.pos_emb(torch.arange(T))[None]
    attn = model.blocks[0].attn
    assert attn.n_heads == 1 and attn.d_head == x.shape[-1]
    q, k = attn.W_q(x)[0], attn.W_k(x)[0]
    scores = (q @ k.T) / math.sqrt(attn.d_head)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    weights = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
    rows = range(len(prompt) - 1, T)
    return dict(prompt=prompt, answer=answer, tokens=[task.itos[i] for i in ids[0].tolist()], queries=list(rows),
                scale=math.sqrt(attn.d_head), d_head=attn.d_head,
                scores=[[round(float(scores[r, c]), 4) for c in range(r + 1)] for r in rows],
                weights=[[round(float(weights[r, c]), 6) for c in range(r + 1)] for r in rows])


def spread(tokens, p):
    """The program's rule with the route at p switched off: own letter 50, every letter 100 times its share of 0..p."""
    score = Counter()
    score[tokens[p]] += 50.0
    for t in tokens[:p + 1]:
        score[t] += 100.0 / (p + 1)
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
    return dict(p=p, positions=p + 1, tokens=tokens[:p + 1], scores=[[t, round(s, 4)] for t, s in ranked],
                winner=ranked[0][0])


KEPT = ["token_emb.weight", "pos_emb.weight", "blocks.0.attn.W_q.weight", "blocks.0.attn.W_k.weight",
        "blocks.0.attn.W_v.weight", "head.weight"]
FRESH = r"""
import contextlib, importlib.util, io, json, sys
import torch
sys.path.insert(0, sys.argv[1])
import src.task
spec = importlib.util.spec_from_file_location("template", sys.argv[2])
module = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(module)
    task = src.task.get_task(sys.argv[3])
    model = module.build_model(task)
params = dict(model.named_parameters())
out = {}
for name in json.loads(sys.argv[4]):
    w = params[name].detach().abs().flatten()
    cut = w.kthvalue(int(0.75 * w.numel())).values
    out[name] = dict(shape=list(params[name].shape), top="".join("1" if v > cut else "0" for v in w.tolist()))
print(json.dumps(out))
"""


def template_init():
    """Which cells hold the template's largest random weights (the top quarter by size), built in a fresh process as
    the evaluator builds it: the template keeps its random initialisation, and its first scored run is that network."""
    import subprocess
    env = dict(os.environ, HF_HOME=str(config.HF_HOME), HF_DATASETS_OFFLINE="1")
    p = subprocess.run([sys.executable, "-X", "utf8", "-c", FRESH, str(config.EVOLVE), str(TEMPLATE), TASK, json.dumps(KEPT)],
                       capture_output=True, text=True, env=env, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def staircase():
    rows = [json.loads(line) for line in (RUN / "attempts" / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    run = next(r for r in json.loads((HERE / "qwen_runs.json").read_text(encoding="utf-8"))["runs"]
               if r["run"] == RUN.name)
    exact = {f["sha256"]: f for f in run["files"]}
    steps = []
    for r in rows:
        f = exact.get(r["sha256"], {})
        steps.append(dict(idx=r["idx"], t=r["t"], name=r["name"], exit=r["exit"], logged_accuracy=r["accuracy"],
                          weights=f.get("weights"), visible=f.get("visible"), built=bool(f.get("ok")),
                          sha256=r["sha256"]))
    return steps, run


def main():
    tasks, evaluator = networks.paper_code()
    task, model = networks.build(networks.load(NET), TASK)
    _, template = networks.build(networks.load(TEMPLATE), TASK)
    net = tensors(model)
    weights = networks.weights(model)
    nonzero = sum(t["nonzero"] for t in net)
    assert (weights, nonzero) == (11450, 128), (weights, nonzero)

    pos = dict(model.named_parameters())["pos_emb.weight"].detach()
    routes = {p: int(pos[p, 39:50].argmax()) for p in range(11, 22)}
    assert all(float(pos[p, 39 + s]) == 10.0 for p, s in routes.items())

    answer = verify.answers(model, task, [PROMPT])[0]
    assert answer == "wfd_mod_fnm", answer
    att = attention(model, task, PROMPT, answer)
    for i, p in enumerate(att["queries"]):                      # each query puts all its weight on its route
        assert max(range(p + 1), key=lambda c: att["weights"][i][c]) == routes[p]

    steps, run = staircase()
    run2 = next(r for r in json.loads((HERE / "qwen_runs.json").read_text(encoding="utf-8"))["runs"]
                if r["run"] == "word-reversal-3x3-qwen37-max-s2")

    lines = NET.read_text(encoding="utf-8").splitlines()
    hero = [dict(line=n, text=lines[n - 1].strip()) for n in HERO_LINES]
    assert all(h["text"] and h["text"] in NET.read_text(encoding="utf-8") for h in hero)

    examples = tasks.get_task(TASK).generate_examples(200, seed=0)
    got = verify.answers(model, task, [e.prompt for e in examples])
    acc, _ = evaluator.evaluate(model, task, n_samples=200, seed=0, device="cpu")

    out = dict(
        provisional=False,
        network=dict(file=NET.name, run=RUN.name, attempt=27, weights=weights, nonzero=nonzero, d_model=50,
                     heads=1, layers=1, tensors=net,
                     values={t["label"]: sorted({c[2] for c in t["cells"]}) for t in net}),
        template=dict(file="experiments/evolve/interpretable_transformer.py", weights=networks.weights(template),
                      tensors=[dict(name=n, shape=list(p.shape), count=p.numel()) for n, p in template.named_parameters()],
                      init_top_quarter=template_init()),
        channels=dict(letters=[0, 27], keys=[28, 38], queries=[39, 49], vocab=task.vocab_size),
        routes={str(p): s for p, s in routes.items()},
        attention=att,
        spread14=spread(att["tokens"], 14),
        staircase=steps,
        run2=dict(pick=run2["pick"], first_perfect=run2.get("first_perfect_visible"), calls=run2["calls"]),
        run1=dict(calls=run["calls"], first_perfect=run.get("first_perfect_visible")),
        hero_code=hero,
        visible=dict(n=200, seed=0, accuracy=acc, correct=sum(g == e.answer for g, e in zip(got, examples)),
                     examples=[dict(prompt=e.prompt, answer=e.answer, network=g) for e, g in zip(examples, got)]),
    )
    (HERE / "qwen_network.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"network {weights} weights, {nonzero} not zero; template {out['template']['weights']}; routes {routes}")
    print(f"{PROMPT} -> {answer}; visible {out['visible']['correct']} of 200 (evaluator {acc}); "
          f"spread at 14: {out['spread14']['scores'][:3]}; staircase {len(steps)} attempts")


if __name__ == "__main__":
    main()
