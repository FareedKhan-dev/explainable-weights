"""results/cells.py: the article's code cells, in order. results/captures.py runs them in one shared namespace and saves
what each one prints to results/outputs/<id>.txt. They run from the repository root, where runs/ holds the study's
run folders and results/ its checkpoints.

REPO_CELLS run after them, in the same namespace: the article draws or states their result instead of printing it."""

CELLS = {}
REPO_CELLS = {}

CELLS["task"] = '''from wsp.tasks import Problem

problem = Problem("word-reversal-3x3")
for ex in problem.examples(3, seed=0):                  # the first three of the agent's visible prompts
    print(ex.prompt, "->", ex.answer)
print(len(problem.vocab), "characters:", "".join(problem.vocab))
print("fnm_mod_wfd= becomes", problem.encode("fnm_mod_wfd="))'''

CELLS["template_flops"] = '''import torch
from torch.utils.flop_counter import FlopCounterMode
from wsp import config, networks

torch.manual_seed(0)                                            # the template's weights start random
task, template = networks.build(networks.load(config.EVOLVE / "interpretable_transformer.py"), "word-reversal-3x3")
ids, work = task.encode("fnm_mod_wfd="), []
for _ in range(task.answer_len):                                # one new letter per step
    with FlopCounterMode(display=False) as count, torch.no_grad():
        scores = template(torch.tensor([ids]))                  # the whole sequence goes through again
    work.append(count.get_total_flops())
    ids = ids + [int(scores[0, -1].argmax())]                   # the highest score is the next letter
print(f"{sum(p.numel() for p in template.parameters()):,d} weights, answer {task.decode(ids[-task.answer_len:])}")
print(f"first letter {work[0]:,d} operations, last letter {work[-1]:,d}")
print(f"all {len(work)} letters {sum(work):,d} operations")'''

CELLS["bb_weights"] = '''from wsp.model import SimpleTransformer
from wsp.tasks import Problem

add = Problem("addition-five-digits")
black_box = SimpleTransformer(len(add.vocab), max_seq_len=18)              # the template, trained with AdamW
black_box.load_state_dict(torch.load("results/baseline_addition.step35000.pt"))
every = torch.cat([p.detach().flatten() for p in black_box.parameters()])
print(f"{every.numel():,d} weights, {(every.abs() < 0.01).float().mean():.1%} of them near zero")
print("the first query table, top left corner:")
print(black_box.blocks[0].attn.W_q.weight[:4, :6].detach().numpy().round(3))'''

CELLS["embeddings"] = '''qwen_file = "runs/word-reversal-3x3-qwen37-max-s1/attempts/027_PosRouting_noW_o.py"
task, qwen = networks.build(networks.load(qwen_file), "word-reversal-3x3")
w = task.encode("w")[0]
print(f"w (id {w}) lights channel {qwen.token_emb.weight[w].nonzero().flatten().tolist()}")
for p in (8, 11):                                               # the w of fnm_mod_wfd=, and the = after it
    print(f"position {p} lights channel {qwen.pos_emb.weight[p].nonzero().flatten().tolist()}")'''

CELLS["scores"] = '''attn = qwen.blocks[0].attn
h = qwen.token_emb(torch.tensor(task.encode("fnm_mod_wfd="))) + qwen.pos_emb.weight[:12]
q, k = attn.W_q(h[11]), attn.W_k(h)                             # the last position asks, every position answers
scores = (k @ q / 50 ** 0.5).detach()
print("scores :", [round(s, 2) for s in scores.tolist()])
print("softmax:", [round(a, 2) for a in scores.softmax(0).tolist()])'''

CELLS["logits"] = '''with torch.no_grad():
    out = qwen(torch.tensor([task.encode("fnm_mod_wfd=")]))[0, -1]    # one score per character, at the last position
best = out.topk(2)
for s, i in zip(best.values.tolist(), best.indices.tolist()):
    print(f"{task.decode([i])} scores {s:.1f}")'''

CELLS["whole"] = '''from wsp import verify

with FlopCounterMode(display=False) as count:
    answer = verify.answers(qwen, task, ["fnm_mod_wfd="])[0]
nonzero = sum(int((p != 0).sum()) for p in qwen.parameters())
print(f"answer {answer}, {count.get_total_flops():,d} operations")
print(f"{nonzero} of {sum(p.numel() for p in qwen.parameters()):,d} weights are not zero")'''

CELLS["route14"] = '''from wsp import programs, switchoff

route_14 = ("pos_emb.weight", 14 * 50 + 39 + programs.ROUTES[14])     # the one weight that names step 14's source
with switchoff.off(qwen, params=[route_14]):
    print("network:", verify.answers(qwen, task, ["fnm_mod_wfd="])[0])
print("program:", task.decode(programs.routing(task.encode("fnm_mod_wfd="), off=(14,))))'''

CELLS["pointer"] = '''originals = config.PAPER / "experiments" / "nl-bottleneck" / "originals"     # the paper's own networks
_, tiny = networks.build(networks.load(originals / "word-reversal-3x3.py"), "word-reversal-3x3")
print(sum(p.numel() for p in tiny.parameters()), "weights:", tiny.reads.tolist(), verify.answers(tiny, task, ["fnm_mod_wfd="])[0])
with torch.no_grad():
    tiny.reads[2] = 0.0                                          # switch off the jump back at each new word
print("jump off:", tiny.reads.tolist(), verify.answers(tiny, task, ["fnm_mod_wfd="])[0])'''

CELLS["gcd"] = '''gcd_task, folder = networks.build(networks.load(originals / "gcd-three-digits.py"), "gcd-three-digits")
print(sum(p.numel() for p in folder.parameters()), "weights:", "084,036= gives", verify.answers(folder, gcd_task, ["084,036="])[0])'''

REPO_CELLS["ig"] = '''from wsp import saliency

seen = "12345+67890=08023"                                                  # the units digit comes next
x = torch.tensor([add.encode(seen)])
attr, gap = saliency.integrated_gradients(black_box.eval(), x, add.stoi["5"])
share = attr.abs() / attr.abs().sum()
for k in share.topk(3).indices.tolist():
    print(f"{seen[k]} at position {k:2d} carries {share[k]:.1%} of the attribution")
print(f"the attributions add up to {attr.sum():.2f}, and the score moved {gap:.2f}")'''

REPO_CELLS["flops"] = '''import torch
from torch.utils.flop_counter import FlopCounterMode
from wsp import config, networks, verify

qwen_file = "runs/word-reversal-3x3-qwen37-max-s1/attempts/027_PosRouting_noW_o.py"
task, qwen = networks.build(networks.load(qwen_file), "word-reversal-3x3")
torch.manual_seed(0)                                                     # the template's weights start random
_, template = networks.build(networks.load(config.EVOLVE / "interpretable_transformer.py"), "word-reversal-3x3")
for name, net in (("template", template), ("qwen", qwen)):
    with FlopCounterMode(display=False) as count:                        # counts every multiplication and addition
        answer = verify.answers(net, task, ["fnm_mod_wfd="])[0]
    weights = sum(p.numel() for p in net.parameters())
    print(f"{name:9s}{weights:8,d} weights {count.get_total_flops():12,d} operations   answer {answer}")'''

REPO_CELLS["hidden_sets"] = '''from wsp import verify

hidden = verify.dataset_hidden("trec-coarse")                 # training questions outside the agent's pool
pool = verify.seen_by_agent("trec-coarse")                    # every question the agent's evaluator could draw
test = verify.trec_rows("test")                               # the official test split, never loaded in a run
print(len(hidden), "hidden questions and", len(test), "official test questions")
print(sum(p in pool for p, _ in hidden), "hidden and", sum(p in pool for p, _ in test), "test questions repeat a pool question")'''

REPO_CELLS["data_use"] = '''from wsp import audit

print(audit.data_use("runs/nli-snli-qwen37-max-s2/attempts/051_NaiveBayesPosition.py", "nli-snli"))
print(audit.data_use(qwen_file, "word-reversal-3x3"))'''
