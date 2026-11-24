"""results/cells.py: the article's code cells, in order. results/captures.py runs them in one shared namespace and saves
what each one prints to results/outputs/<id>.txt. They run from the repository root, where runs/ holds the study's
run folders and results/ its checkpoints.

REPO_CELLS run after them, in the same namespace: the article draws or states their result instead of printing it."""

CELLS = {}
REPO_CELLS = {}

CELLS["task"] = '''from wsp.tasks import Problem

problem = Problem("word-reversal-3x3")

# The first three of the agent's visible prompts
for ex in problem.examples(3, seed=0):
    print(ex.prompt, "->", ex.answer)

print(len(problem.vocab), "characters:", "".join(problem.vocab))
print("fnm_mod_wfd= becomes", problem.encode("fnm_mod_wfd="))'''

CELLS["template"] = '''import torch
from wsp import config, networks

torch.manual_seed(0)
path = config.EVOLVE / "interpretable_transformer.py"
task, template = networks.build(networks.load(path), "word-reversal-3x3")

weights = sum(p.numel() for p in template.parameters())
print(f"{weights:,d} weights")'''

CELLS["template_flops"] = '''from torch.utils.flop_counter import FlopCounterMode

ids = task.encode("fnm_mod_wfd=")
work = []

# Each new letter runs the whole sequence again
for _ in range(task.answer_len):
    with FlopCounterMode(display=False) as count, torch.no_grad():
        scores = template(torch.tensor([ids]))
    work.append(count.get_total_flops())

    # The highest score is the next letter
    ids.append(int(scores[0, -1].argmax()))

print("answer", task.decode(ids[-task.answer_len:]))
print(f"first letter {work[0]:,d} operations, last letter {work[-1]:,d}")
print(f"all {len(work)} letters {sum(work):,d} operations")'''

CELLS["bb_weights"] = '''from wsp.model import SimpleTransformer
from wsp.tasks import Problem

add = Problem("addition-five-digits")
black_box = SimpleTransformer(len(add.vocab), max_seq_len=18)
black_box.load_state_dict(torch.load("results/baseline_addition.step35000.pt"))

every = torch.cat([p.detach().flatten() for p in black_box.parameters()])
near_zero = (every.abs() < 0.01).float().mean()
print(f"{every.numel():,d} weights, {near_zero:.1%} of them near zero")

corner = black_box.blocks[0].attn.W_q.weight[:4, :6].detach()
print("the first query table, top left corner:")
print(corner.numpy().round(3))'''

CELLS["embeddings"] = '''run = "runs/word-reversal-3x3-qwen37-max-s1"
qwen_file = f"{run}/attempts/027_PosRouting_noW_o.py"
task, qwen = networks.build(networks.load(qwen_file), "word-reversal-3x3")

w = task.encode("w")[0]
channels = qwen.token_emb.weight[w].nonzero().flatten().tolist()
print(f"w (id {w}) lights channel {channels}")

# Position 8 holds a w, and position 11 the =
for p in (8, 11):
    channels = qwen.pos_emb.weight[p].nonzero().flatten().tolist()
    print(f"position {p} lights channel {channels}")'''

CELLS["scores"] = '''attn = qwen.blocks[0].attn

# Our prompt as vectors
prompt = torch.tensor(task.encode("fnm_mod_wfd="))
h = qwen.token_emb(prompt) + qwen.pos_emb.weight[:12]

# The last position asks, and every position answers
q = attn.W_q(h[11])
k = attn.W_k(h)
scores = (k @ q / 50 ** 0.5).detach()

print("scores :", [round(s, 2) for s in scores.tolist()])
print("softmax:", [round(a, 2) for a in scores.softmax(0).tolist()])'''

CELLS["logits"] = '''# One score per character, at the last position
x = torch.tensor([task.encode("fnm_mod_wfd=")])
with torch.no_grad():
    out = qwen(x)[0, -1]

best = out.topk(2)
for s, i in zip(best.values.tolist(), best.indices.tolist()):
    print(f"{task.decode([i])} scores {s:.1f}")'''

CELLS["whole"] = '''from wsp import verify

with FlopCounterMode(display=False) as count:
    answer = verify.answers(qwen, task, ["fnm_mod_wfd="])[0]
print(f"answer {answer}, {count.get_total_flops():,d} operations")

total = sum(p.numel() for p in qwen.parameters())
nonzero = sum(int((p != 0).sum()) for p in qwen.parameters())
print(f"{nonzero} of {total:,d} weights are not zero")'''

CELLS["route14"] = '''from wsp import programs, switchoff

# Row 14 of the position table, column 39 plus its source
route_14 = ("pos_emb.weight", 14 * 50 + 39 + programs.ROUTES[14])

with switchoff.off(qwen, params=[route_14]):
    print("network:", verify.answers(qwen, task, ["fnm_mod_wfd="])[0])
ids = programs.routing(task.encode("fnm_mod_wfd="), off=(14,))
print("program:", task.decode(ids))'''

CELLS["pointer"] = '''originals = config.PAPER / "experiments" / "nl-bottleneck" / "originals"
source = networks.load(originals / "word-reversal-3x3.py")
_, tiny = networks.build(source, "word-reversal-3x3")

weights = sum(p.numel() for p in tiny.parameters())
answer = verify.answers(tiny, task, ["fnm_mod_wfd="])[0]
print(weights, "weights:", tiny.reads.tolist(), answer)

# Switch off the jump back at each new word
with torch.no_grad():
    tiny.reads[2] = 0.0
answer = verify.answers(tiny, task, ["fnm_mod_wfd="])[0]
print("jump off:", tiny.reads.tolist(), answer)'''

CELLS["gcd"] = '''source = networks.load(originals / "gcd-three-digits.py")
gcd_task, folder = networks.build(source, "gcd-three-digits")

weights = sum(p.numel() for p in folder.parameters())
answer = verify.answers(folder, gcd_task, ["084,036="])[0]
print(weights, "weights:", "084,036= gives", answer)'''

REPO_CELLS["ig"] = '''from wsp import saliency

# The adder has read this much, and the units digit comes next
seen = "12345+67890=08023"
x = torch.tensor([add.encode(seen)])
attr, gap = saliency.integrated_gradients(black_box.eval(), x, add.stoi["5"])

# The three characters that carry the most attribution
share = attr.abs() / attr.abs().sum()
for k in share.topk(3).indices.tolist():
    print(f"{seen[k]} at position {k:2d} carries {share[k]:.1%} of the attribution")
print(f"the attributions add up to {attr.sum():.2f}, and the score moved {gap:.2f}")'''

REPO_CELLS["flops"] = '''import torch
from torch.utils.flop_counter import FlopCounterMode
from wsp import config, networks, verify

# The network Qwen3.7-Max wrote, then the template with random weights
qwen_file = "runs/word-reversal-3x3-qwen37-max-s1/attempts/027_PosRouting_noW_o.py"
task, qwen = networks.build(networks.load(qwen_file), "word-reversal-3x3")
torch.manual_seed(0)
path = config.EVOLVE / "interpretable_transformer.py"
_, template = networks.build(networks.load(path), "word-reversal-3x3")

# Every multiplication and addition of one whole answer, for each network
for name, net in (("template", template), ("qwen", qwen)):
    with FlopCounterMode(display=False) as count:
        answer = verify.answers(net, task, ["fnm_mod_wfd="])[0]
    weights = sum(p.numel() for p in net.parameters())
    flops = count.get_total_flops()
    print(f"{name:9s}{weights:8,d} weights {flops:12,d} operations   answer {answer}")'''

REPO_CELLS["hidden_sets"] = '''from wsp import verify

# Training questions outside the agent's pool
hidden = verify.dataset_hidden("trec-coarse")

# Every question the agent's evaluator could draw
pool = verify.seen_by_agent("trec-coarse")

# The official test split, never loaded in a run
test = verify.trec_rows("test")
print(len(hidden), "hidden questions and", len(test), "official test questions")

# The questions that repeat one from the pool
in_hidden = sum(p in pool for p, _ in hidden)
in_test = sum(p in pool for p, _ in test)
print(in_hidden, "hidden and", in_test, "test questions repeat a pool question")'''

REPO_CELLS["data_use"] = '''from wsp import audit

# A file that counted labels, then the reversal pick
counting = "runs/nli-snli-qwen37-max-s2/attempts/051_NaiveBayesPosition.py"
print(audit.data_use(counting, "nli-snli"))
print(audit.data_use(qwen_file, "word-reversal-3x3"))'''
