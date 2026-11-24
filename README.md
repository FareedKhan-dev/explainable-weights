# Explainable Weights

**An AI agent writes every weight of a small transformer as readable code, and every number in the network can be explained.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab.svg)](pyproject.toml)
[![PyTorch 2.10](https://img.shields.io/badge/pytorch-2.10-ee4c2c.svg)](pyproject.toml)
[![CPU only](https://img.shields.io/badge/hardware-CPU%20only-6b7280.svg)](#quickstart)
[![arXiv 2611.18114](https://img.shields.io/badge/arXiv-2611.18114-b31b1b.svg)](https://arxiv.org/abs/2611.18114)

<p align="center"><img src="images/g-hero.gif" width="100%" alt="The agent writes the network's routing wires one by one, and each letter of the prompt follows its wire to the answer"></p>

Trained networks work, but their weights do not say how. This repository follows the method of the paper
[*Interpretable weight-space programming builds transparent transformers and brain encoding models*](https://arxiv.org/abs/2611.18114): an AI
agent writes every weight of a small transformer by hand, as code, with training forbidden, and a fixed evaluator scores
each attempt. Here the agent is **Qwen3.7-Max**, and every network it wrote is put through explainability tests built
from scratch.

It is the companion code of the article **Building Explainable AI with Agents That Write Every Weight**, and the whole
article is included below as a [walkthrough](#walkthrough).

## Highlights

- **A network written by hand.** For word reversal, Qwen3.7-Max wrote a one-layer transformer with 11,450 weights, of
  which 128 are not zero. It answers 2,000 of 2,000 hidden prompts and 20,000 of 20,000 extra prompts correctly, and
  no weight was trained.
- **Every weight has a job.** The 128 weights that are not zero split into seven jobs: letter codes, position keys,
  routing queries, moving the queries, moving the keys, copying the letter and reading it out.
- **The explanation predicts the network.** A short program read off the weights predicts the network's exact answer,
  prompt by prompt, in 13 switch-off cases on 2,000 hidden prompts each.
- **The words carry the network.** From the agent's own 151-word description, two fresh agents rebuilt the identical
  11,450-weight network on their first scored run. With no description, the control needed 326,960 weights, 28.6 times
  more.
- **Where it stops.** Qwen3.7-Max solved 1 of the 10 problems it tried. The paper's own agent solved all 13 generated
  problems, and its hand-written text classifiers beat the black boxes trained here on the same template.

| | Qwen3.7-Max (this study) | The paper's agent (checked here) |
|---|---|---|
| Generated problems solved, of the 6 both agents tried | 1 | 6 |
| Smallest network that is right on every hidden prompt | 11,450 weights | 3 weights |
| Four text datasets, hidden accuracy against our trained black boxes | below on all four | above on all four |

## Contents

- [Quickstart](#quickstart)
- [Repository layout](#repository-layout)
- [Reproducing the results](#reproducing-the-results)
- [Running the agent yourself](#running-the-agent-yourself)
- [Walkthrough](#walkthrough)
- [Citation](#citation)
- [License](#license)

## Quickstart

Everything here runs on a CPU. The package reuses the paper's tasks, evaluator and template network, so clone the
paper's code beside this repository first, at the commit the runs used:

```bash
git clone https://github.com/csinva/weight-space-programming-autoresearch
cd weight-space-programming-autoresearch && git checkout 4f8eb615 && uv sync && cd ..

git clone https://github.com/FareedKhan-dev/explainable-weights
cd explainable-weights && uv sync
export WSP_PAPER=../weight-space-programming-autoresearch

uv run pytest                                   # the ports, the toolkit, the agent loop and the rebuild test
uv run python results/captures.py --check       # replay every printout of the walkthrough and compare it
```

No test calls a model: the agent-loop and rebuild tests drive the loop with a scripted client. The text-dataset cells
download TREC, SST-2, MRPC and SNLI into `.hfh/` on their first run. On Windows, use `set` (cmd) or `$env:` (PowerShell)
in place of `export`.

## Repository layout

```
explainable-weights/
├── wsp/                    the package
│   ├── agent.py            the weight-writing loop, with the paper's prompts, targets and stopping rule
│   ├── client.py           a streaming client for any OpenAI-compatible chat endpoint, standard library only
│   ├── sandbox.py          the run folder, the agent's six restricted tools, and a snapshot of every attempt
│   ├── config.py           paths, model and limits, set through WSP_ environment variables
│   ├── tasks.py            the problems, ported from the paper, as prompts with exact answers
│   ├── model.py            the template transformer, ported from the paper
│   ├── evaluate.py         the fixed judge: greedy decoding, exact match
│   ├── networks.py         load and build any network an agent wrote
│   ├── baseline.py         the black box: the same template, trained with AdamW
│   ├── saliency.py         gradient saliency and integrated gradients
│   ├── switchoff.py        switch named parts or single weights off, and restore them exactly
│   ├── programs.py         explanations written as programs, read off a network's weights
│   ├── trace.py            follow one input through a network, step by step
│   ├── verify.py           hidden, extra and exhaustive checks, and the dataset splits
│   ├── audit.py            the code scan and the data-use test
│   └── rebuild.py          the describe-and-rebuild test
├── tests/                  ports, toolkit, agent loop and rebuild tests (no model is called)
├── scripts/                run the agent on one problem, on a study, or through the rebuild test
├── runs/                   every network the agent wrote, run by run, and the rebuild test's files
├── results/                every result file, the printouts, the checkpoints, and the scripts that made them
├── images/                 the walkthrough's figures
└── walkthrough.ipynb       the walkthrough's code cells, in order
```

`runs/` and `results/` each have their own README.

## Reproducing the results

Each number in the walkthrough comes from a file in `results/`, and most of those files can be rebuilt from this
repository and the paper's code. The main scripts:

| Command | What it rebuilds |
|---|---|
| `python results/captures.py --check` | every printout in the walkthrough, compared with `results/outputs/` |
| `python results/explain_predicts.py` | the switch-off cases, each predicted by an explanation program |
| `python results/paper_checks.py` | the paper's networks on visible, hidden and extra prompts, and GCD and dates on every input |
| `python results/compare.py` | the two agents side by side, from the result files |
| `python results/train_baseline.py` | the trained black box (about three hours) |
| `python results/blackbox_explain.py` | switch-off and integrated gradients on the trained black box |
| `python results/blackbox_text.py` | the black boxes trained on the four text datasets |

The analysis of the agent's own runs (`results/qwen_runs.json`), the cost in budget units, and the transcript audits
were computed from the conversation logs, which are not published. Their results are in `results/`.

## Running the agent yourself

The loop talks to any OpenAI-compatible chat completions endpoint that supports tool calls:

```bash
export WSP_ENDPOINT=https://<provider>/v1/chat/completions
export WSP_MODEL=Qwen3.7-Max
export WSP_KEY_FILE=api_key.txt                 # a file that holds your API key
python scripts/run_agent.py --probe             # one call, to check that tool calls work
python scripts/run_agent.py --task word-reversal-3x3 --seed 1
python scripts/run_study.py --tasks all --seeds 1 2 --parallel 3
python scripts/run_rebuild.py --tasks word-reversal-3x3
```

Each run stops at its own budget (`--run-cap`), counted from the tokens it logs, and a file named `STOP` in the
repository root stops every run before its next call.

## Walkthrough

The full article, **Building Explainable AI with Agents That Write Every Weight**: *When training hides the logic, an agent writes the weights instead*. It builds the `wsp` package step by step. To run the finished code, follow the [Quickstart](#quickstart) instead of the setup it describes.

Every answer from ChatGPT comes out of billions of multiplications over weights that training set and nobody can read. Explainable AI is the field that tries to show how such a model reaches its answers. A recent [paper on interpretable weight-space programming](https://arxiv.org/abs/2611.18114) puts the problem simply: "Neural networks are powerful predictors, yet their opacity limits their usefulness in science." Its fix is to let an AI agent write every weight of a small transformer by hand, as code a person can read. I rebuilt the method step by step and ran Qwen3.7-Max as that agent. The figure at the top of this page shows it at work.

This is the network Qwen3.7-Max wrote for one task, reversing the order of three words. It answers prompts it never saw while it worked:

```bash
#### OUTPUT ####
prompt fnm_mod_wfd=   answer wfd_mod_fnm
prompt gqk_ymf_kty=   answer kty_ymf_gqk
prompt gdl_ewt_uhi=   answer uhi_ewt_gdl

2,000 of 2,000 hidden prompts right
11,450 weights, 128 of them not zero
```

Each of those 128 weights was set by a line of code, and we will read them one by one later in this walkthrough. Qwen3.7-Max solved word reversal but none of the other 9 tasks it tried, and near the end we put it side by side with the paper's own runs. The agent loop, the checks and every network it wrote are in this repository.

We will build everything in this order:

1. The word reversal task, and the small transformer the agent fills in.
2. The same transformer trained the usual way, and what explainable AI tools can tell us about it.
3. The agent loop that writes the weights by hand.
4. The network Qwen3.7-Max wrote, read one weight table at a time.
5. Two tests that check whether its explanation is true.
6. The results on all 10 tasks, side by side with the paper's own runs.
7. The paper's smallest networks, and where this method stops.

### Project Setup

First, we have to get the paper's code, because our package reuses its tasks, its evaluator and the small transformer every agent starts from. We clone it at the commit our runs used, then create our own project beside it:

```bash
git clone https://github.com/csinva/weight-space-programming-autoresearch
cd weight-space-programming-autoresearch && git checkout 4f8eb615 && uv sync && cd ..
uv init explainable-weights --no-package && cd explainable-weights
uv add torch datasets numpy
export WSP_PAPER=../weight-space-programming-autoresearch
```

The last line tells our package, `wsp`, where the paper's code lives. Everything in this walkthrough runs on a CPU.

### Word Reversal Task

Before we look at any network, we need the task it has to solve. The prompt is three words of three letters, joined by `_` and ending in `=`, and the answer is the same three words in reverse order:

![The three words trade places](images/g-task.gif)

We have nothing to download, because the paper generates every example from a random seed. We rewrite its generator in our package:

```python
# tasks.py
def word_reversal(rng):
    """Three 3-letter words, written back in reverse order: cat_dog_pig= gives pig_dog_cat."""
    words = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(3)) for _ in range(3)]
    return Example("_".join(words) + "=", "_".join(reversed(words)))
```

It picks nine random letters, groups them into three words and returns the prompt and the answer together.

A transformer cannot read letters, though. Each task in our package also has a `Problem` class, kept in the repository, that holds the task's characters and turns text into ids and back. We can print a few examples, the characters and one prompt the way the network sees it:

```python
from wsp.tasks import Problem

problem = Problem("word-reversal-3x3")
for ex in problem.examples(3, seed=0):                  # the first three of the agent's visible prompts
    print(ex.prompt, "->", ex.answer)
print(len(problem.vocab), "characters:", "".join(problem.vocab))
print("fnm_mod_wfd= becomes", problem.encode("fnm_mod_wfd="))

#### OUTPUT ####
myn_biq_pmz= -> pmz_biq_myn
jpl_sgq_eje= -> eje_sgq_jpl
ydt_zir_wzt= -> wzt_zir_ydt
28 characters: abcdefghijklmnopqrstuvwxyz_=
fnm_mod_wfd= becomes [5, 13, 12, 26, 12, 14, 3, 26, 22, 5, 3, 27]
```

To the network, `fnm_mod_wfd=` is nothing more than that list of ids. Its job is to write the ids of `wfd_mod_fnm` one at a time, and we will follow this one prompt through the rest of the blog.

The paper scores every network on three separate sets of these prompts:

- **Visible:** 200 prompts from seed 0. It is the one score the agent sees while it works.
- **Hidden:** 2,000 prompts from a seed no run ever uses. No network can be tuned to them.
- **Extra:** a third set of 20,000 prompts from another seed, to catch rare mistakes the hidden set can miss.

A network that is perfect on all three sets has learned the reversal itself, because the agent never saw the hidden or the extra prompts:

![The agent sees only one set](images/g-sets.gif)

Our running prompt, `fnm_mod_wfd=`, is the first of the hidden prompts. Next, we look at the transformer these weights go into.

### Transformer Overview

Each agent in the paper fills in the same small network, so we have to understand it first. It is a transformer, the kind of model behind ChatGPT, only far smaller, and it writes its answer one character at a time.

A transformer is a stack of identical blocks, and each block has two parts:

- **Attention:** every position looks back at the positions before it and pulls in what it needs. In word reversal, this part can fetch the letter to copy.
- **MLP:** two layers that work on each position alone, computing something new from what attention brought in.

We can think of attention as the part that decides where to look, and the MLP as the part that decides what to do with what it found. Our template stacks three of these blocks, and every new letter climbs through all of them:

![Letters climb the blocks to a guess](images/g-arch.gif)

Let us follow one prediction through the template:

1. Each character becomes its id, and the token embedding turns that id into a vector of 64 numbers.
2. The position embedding adds a second vector that says where the character is.
3. Their sum is the residual stream, the running record that every block reads from and writes back to.
4. Each block normalizes the stream with a LayerNorm, runs attention across four heads and adds the result back, then does the same for the MLP.
5. After the last block, the output head scores every character, and the highest score is the next letter.
6. That letter joins the sequence, and the whole sequence goes through the network again.

The template in our animation has random weights, and its pick is wrong. Only the weights decide which letter is right.

### Template Network Code

PyTorch makes each part a small class, and we start with attention. Its two parameters are `d_model`, the width of the residual stream, and `n_heads`, the number of heads that share that width:

```python
# model.py
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.W_q, self.W_k, self.W_v, self.W_o = (nn.Linear(d_model, d_model, bias=False) for _ in range(4))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = (w(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2) for w in (self.W_q, self.W_k, self.W_v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)          # how well each query matches each key
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)       # a token only sees itself and the past
        return self.W_o((attn @ v).transpose(1, 2).reshape(B, T, D))
```

We just coded the attention part. Our `__init__` method creates four weight tables without a bias: `W_q`, `W_k` and `W_v` turn each position's vector into a query, a key and a value, and `W_o` mixes the heads back together.

Our `forward` method scores every query against every key and divides by the square root of the head width. The mask hides the later positions, so a position only sees itself and the past.

Softmax then turns each row of scores into weights that add up to one. Those weights decide how much of each value we mix in. In our random template, the first head spreads the last position's weights over every letter:

![Random weights spread attention thin](images/g-pie.gif)

No letter gets more than a fifth of the weight, so what comes back is a blend of every position. The agent's network will do the opposite and give one letter all of it.

A small block class in our repository wraps this attention and the MLP. The full network puts the two embedding tables in front of the blocks and the output head after them:

```python
# model.py
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=64, n_heads=4, n_layers=3, d_ff=256):
        super().__init__()
        self.token_emb, self.pos_emb = nn.Embedding(vocab_size, d_model), nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.final_ln, self.head = nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        h = self.token_emb(ids) + self.pos_emb(torch.arange(ids.shape[1], device=ids.device))[None]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))                  # one score per character, at every position
```

That completes our template. Its `__init__` method builds the token and position tables, the blocks, a final LayerNorm and the head, and the default arguments in its signature are the paper's sizes.

Our `forward` method adds the two vectors, runs the sum through every block and scores every character at every position.

### Counting the Computation

Even a small transformer does a lot of arithmetic. PyTorch's `FlopCounterMode` counts every multiplication and addition inside a block of code, and we can run our template on our prompt one letter at a time, the way the evaluator does:

```python
import torch
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
print(f"all {len(work)} letters {sum(work):,d} operations")

#### OUTPUT ####
154,368 weights, answer kxrrvjl=cok
first letter 3,692,544 operations, last letter 6,938,624
all 11 letters 58,344,704 operations
```

Our weights are random and the answer is nonsense, but the count is real. Even this small template needs tens of millions of operations for eleven letters.

The last letter costs almost twice as much as the first, because every step sends the whole sequence so far through the network again:

![Each new letter re-reads the sequence](images/g-compute.gif)

The models we use every day are far larger. [GPT-3](https://arxiv.org/abs/2005.14165) has 175 billion weights, and [Kimi K2](https://arxiv.org/abs/2507.20534) has 1 trillion.

A dense model like GPT-3 uses each weight in one multiplication and one addition for every token it writes, about 350 billion operations per token. If we zoom out from our network, we can see how small it is next to them:

![Zooming out from our network](images/g-scale.gif)

Each of those operations multiplies a weight that training set, and no weight comes with a reason.

### Training the Usual Way

To see the problem for ourselves, we will build a black box of our own: the same template, fitted to five-digit addition by gradient descent. Training means we show the network examples, measure how wrong it is, and nudge every weight a little in the direction that makes it less wrong.

Our training loop is the standard one for transformers:

```python
# baseline.py
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
```

We just coded the trainer. It uses AdamW, the usual optimizer for transformers, with a learning rate that decays along a cosine curve.

Each step draws a fresh batch of sums and computes the loss on the answer digits only. Then `backward` and `step` change all the weights at once.

Every `log_every` steps, the trainer scores the network on sums it never trained on and writes a checkpoint to disk. I planned 40,000 steps, but an overnight pause in my runs stopped training at step 35,000, and that checkpoint is our black box. Its curve shows how quickly it learned to add:

![A trained adder learns almost every sum](images/g-train.gif)

When training stopped, our network got 1,999 of 2,000 new sums right.

What we want to know now is whether its weights can tell us how it adds.

### Looking Inside the Weights

Our trained adder works, so its weights must hold some method for carrying digits. We can load the checkpoint and print a few of them:

```python
from wsp.model import SimpleTransformer
from wsp.tasks import Problem

add = Problem("addition-five-digits")
black_box = SimpleTransformer(len(add.vocab), max_seq_len=18)              # the template, trained with AdamW
black_box.load_state_dict(torch.load("results/baseline_addition.step35000.pt"))
every = torch.cat([p.detach().flatten() for p in black_box.parameters()])
print(f"{every.numel():,d} weights, {(every.abs() < 0.01).float().mean():.1%} of them near zero")
print("the first query table, top left corner:")
print(black_box.blocks[0].attn.W_q.weight[:4, :6].detach().numpy().round(3))

#### OUTPUT ####
152,000 weights, 9.0% of them near zero
the first query table, top left corner:
[[-0.053 -0.075  0.072  0.062 -0.139  0.07 ]
 [ 0.074 -0.212  0.17  -0.227  0.076  0.184]
 [-0.056 -0.049 -0.044 -0.08   0.1   -0.03 ]
 [ 0.065  0.103  0.037 -0.048 -0.153 -0.067]]
```

Almost every weight is a small number that is not zero, and none of them carries a label. Set beside a table the agent wrote, the difference is plain:

![Trained weights beside written ones](images/g-wall.gif)

The left grid is our adder's first query table, all of its numbers. The right one is the same kind of table from the network Qwen3.7-Max wrote, where only a few cells are not zero and each has a job we can name.

Making a model's reasoning readable to a person is what explainable AI is about, and we can get there in two ways:

- **Explain after training:** train the model first, then probe it from outside to see which inputs mattered.
- **Explain by design:** build the model so that every part has a meaning a person can read from the start.

Most explainable AI tools take the first road, and this walkthrough takes the second:

![Two ways to explain a model](images/g-blueprint.gif)

Tools like [LIME](https://arxiv.org/abs/1602.04938), [SHAP](https://arxiv.org/abs/1705.07874) and [integrated gradients](https://arxiv.org/abs/1703.01365) are the first kind, and they tell us how much each input mattered for one answer. For the units digit of a sum, integrated gradients credits the two units digits above it:

![Integrated gradients lights the digits it used](images/g-saliency.gif)

That tells us where the network looks, but not how it adds. Switching parts off tells us even less. When I silenced the attention heads one at a time, all but one of them cost between 23 and 98 points of accuracy:

![Switching heads off breaks the adder](images/g-knockout.gif)

Almost every part matters, and none says why.

The paper asked its own agent to predict what an edit would do to Qwen3.5-0.8B just by reading its weights, and reading them helped "only a little".

[Cynthia Rudin](https://arxiv.org/abs/1811.10154) warned years ago that "trying to explain black box models, rather than creating models that are interpretable in the first place, is likely to perpetuate bad practices". So we will stop explaining a trained network and let an agent write one we can read.

### Rules for Writing Weights

Interpretable weight-space programming, the paper's method, is explainable AI by design: an AI agent opens the network's file and writes a function, `write_weights`, that sets every weight by hand. Its instructions, a page called `program.md`, open with the rule that matters most: "The model is NEVER trained."

The rules let our agent do three things:

- **Change the architecture:** layers, heads, channels, or the classes themselves.
- **Write the weights:** fill every table inside `write_weights` with values it chooses.
- **Run the evaluator:** one command scores the network on the visible prompts.

They also forbid four things:

- **Train:** the forbidden list starts with "Run any kind of training, gradient descent, optimizer step, fitting loop, or backprop."
- **Compute the answer in Python:** the transformer must write every letter itself.
- **Borrow weights:** nothing may come from a trained model.
- **Touch the task or the evaluator:** both files are read-only.

The score is the only feedback the agent gets. It writes the weights, runs the evaluator, reads the score and tries again:

![Write the weights, get a score, try again](images/g-loop.gif)

A perfect score is not the whole goal. In the page's own words, "the real criterion is human-understandable structure in the weights".

### Agent Session Loop

The paper ran every one of its agents through a command-line coding tool. To run Qwen3.7-Max instead, I rebuilt the same loop around a plain chat API, with the paper's own prompts and 6 tools for reading, writing and running files.

Every tool call runs inside a sandbox folder, and the only command the agent may run is the evaluator, with nothing added to it:

![The sandbox refuses all but one command](images/g-sandbox.gif)

A run is a chain of sessions, and this is one of them:

```python
# agent.py
def session(client, box, messages, limits, k, state, logs, tools=TOOLS):
    """One session: call, act, repeat, until the agent ends its turn or a limit or the prompt size stops it."""
    s = dict(k=k, calls=0, input_tokens=0, output_tokens=0, reasoning_tokens=0, cached_tokens=0, tools_used={}, cut_offs=0, end="")
    cuts = 0                                                     # cut-off replies in a row
    while not (end := limits.reached(state["calls"])):
        r = call_model(client, box, messages, state, s, logs, tools)
        if r is None or r.status == "infra":
            return s, stopped(s, r)                              # a context error starts a fresh session
        msg, calls = assistant_message(r, state["calls"])
        if r.stop_reason == "length" and not calls:              # cut off while thinking, never an end of turn
            cuts += 1
            if not cut_off(messages, msg, r, s, logs, cuts):
                return s, True                                   # still cut off after three notes: a fresh session
            continue
        cuts = 0
        messages.append(msg)
        logs.turn(k, msg, len(r.reasoning))
        if not calls:
            return finish(s, "end of turn", False)
        messages.extend(act(box, calls, s, logs))
        if (r.input_tokens or 0) > limits.compact_at:
            return finish(s, "compacted", True)
    return finish(s, end, False)
```

We just coded a session. Each turn calls the model, adds its reply to the conversation and runs the tools it asked for. Our session ends when the agent stops calling tools, when the run hits its limit on calls, or when the prompt grows so long that we start afresh.

There are two important points about this loop:

1. A thinking model can run out of output space while it is still thinking. That reply has no tool call, and the loop sends a short note and asks again. After 3 notes in a row it starts a fresh session.
2. A fresh session starts from the paper's continue prompt and the files on disk, never from the old conversation, and that keeps a long run inside the model's context window.

One real session of our reversal run shows both points at work:

![One session, a cut-off and a restart](images/g-session.gif)

I learned the first point the hard way. In my first launch, a reply cut off while thinking counted as a stalled run, every run stopped early, and I fixed the loop before the runs we look at next.

### Running the Agent

Qwen3.7-Max began the word reversal run the way the paper suggests, by printing a few examples to learn the format. Its first words after that were "Now I understand the task." It ran the template unchanged, then wrote its first design.

That design failed, and the agent saw why in the output: "I see the model is generating all 'l's. My weight design isn't routing correctly."

It traced the fault to its position table, writing that "my position embeddings don't encode source positions for prompt positions". Run on our prompt, the two designs show the bug and the fix:

![The bug and the fix, letter by letter](images/g-fix.gif)

The first full check of the rewrite got all 200 visible prompts right.

From there it shrank the network one part at a time and kept a change only when the score stayed perfect. We can read its attempts from the fourteenth on, as the run logged them:

```bash
#### OUTPUT ####
  #  attempt name                weights  visible right
 14  PosRouting_d40                      crashed
 15  PosRouting_4heads_dff32     26,016    0 of 200
 16  PosRouting_dff16            23,952  200 of 200
 17  PosRouting_dff8             22,920  200 of 200
 18  PosRouting_dff0             21,888  200 of 200
 19  PosRouting_dff0                     crashed
 20  PosRouting_dff0                     crashed
 21  PosRouting_d50              14,250  200 of 200
 22  PosRouting_binary6          13,296    0 of 200
 23  PosRouting_binToken          5,211    0 of 200
 24  PosRouting_d52              15,236  200 of 200
 25  PosRouting_noLN             14,150  200 of 200
 26  PosRouting_noLN_v2          13,950  200 of 200
 27  PosRouting_noW_o            11,450  200 of 200
```

Each failure came with a reason in the agent's own words. A width of forty was "too small for our position embedding layout", and four heads failed because "the attention signal gets diluted across too many heads".

Binary codes for the letters failed because "dot products between binary vectors don't give unique maxima". We can see the whole run of 27 attempts at once:

![Every attempt, from template to final](images/g-shrink.gif)

The final attempt has one layer, one head, no MLP, no LayerNorm and no output table, with 11,450 weights in all. In the next four sections, we read every one of them.

### Residual Stream Layout

All the weights the agent wrote follow one plan. Each position carries 50 numbers in the stream, and the agent split them into three zones, each with one job:

- **Which letter:** one channel for each character of the task.
- **Which position:** one channel for each prompt position before the `=`, the position's key.
- **Where to look:** one channel per prompt position again, set only at the answer positions to name the position each one copies.

We can find the plan at the top of `write_weights`, as a source map and the first channel of each zone:

```python
# runs/word-reversal-3x3-qwen37-max-s1/attempts/027_PosRouting_noW_o.py
# Source mapping: prediction position -> source position to copy from
source_map = {
    11: 8, 12: 9, 13: 10,
    14: 7,
    15: 4, 16: 5, 17: 6,
    18: 3,
    19: 0, 20: 1, 21: 2,
}

KEY_START = V        # position key channel starts at dim 28
QUERY_START = V + 11  # routing query channel starts at dim 39
MIN_D = V + 22       # minimum d_model needed = 50
```

For us, the source map is the whole algorithm in one table. The `=` writes the first answer letter by copying position 8, the first letter of the last word.

Each position after it copies the next letter in reverse word order, and two routes copy a `_` to put the separators back. We can see what the plan means for our `w` and the `=` that will copy it:

![Two positions light their own slots](images/g-stream.gif)

The `w` lights its letter and its position, while the `=` lights its own letter and a query for the position of the `w`.

### Token and Position Embeddings

Characters were the first thing the agent wrote weights for. It gave every character a channel of its own, with one number switched on and every other number left at zero.

This kind of vector is called a one-hot code, and one short loop writes a code for every character:

```python
# runs/word-reversal-3x3-qwen37-max-s1/attempts/027_PosRouting_noW_o.py
# Token embeddings: dims 0..V-1 = token identity (one-hot, scale=10)
token_emb_weight = torch.zeros(V, D)
for i in range(V):
    token_emb_weight[i, i] = 10.0
model.token_emb.weight.data = token_emb_weight
```

The table starts as all zeros, and the loop writes `10.0` into channel `i` for the character with id `i`. The position table is built the same way, with two loops instead of one:

```python
# runs/word-reversal-3x3-qwen37-max-s1/attempts/027_PosRouting_noW_o.py
# Position embeddings
pos_emb_weight = torch.zeros(max_seq_len, D)

# Position key channel: dims KEY_START..KEY_START+10 for positions 0-10
for pos in range(11):
    if pos < max_seq_len:
        pos_emb_weight[pos, KEY_START + pos] = 10.0

# Routing query channel: dims QUERY_START..QUERY_START+10 at prediction positions
for pred_pos, src_pos in source_map.items():
    if pred_pos < max_seq_len:
        pos_emb_weight[pred_pos, QUERY_START + src_pos] = 10.0

model.pos_emb.weight.data = pos_emb_weight
```

The first loop gives each prompt position its key, and the second gives each answer position a query that names the position to copy. When we run the network, each position takes one row from each table and adds the two:

![A letter row plus a position row](images/g-lookup.gif)

We can load the finished network and print the channels that our `w` and our two positions switch on:

```python
qwen_file = "runs/word-reversal-3x3-qwen37-max-s1/attempts/027_PosRouting_noW_o.py"
task, qwen = networks.build(networks.load(qwen_file), "word-reversal-3x3")
w = task.encode("w")[0]
print(f"w (id {w}) lights channel {qwen.token_emb.weight[w].nonzero().flatten().tolist()}")
for p in (8, 11):                                               # the w of fnm_mod_wfd=, and the = after it
    print(f"position {p} lights channel {qwen.pos_emb.weight[p].nonzero().flatten().tolist()}")

#### OUTPUT ####
w (id 22) lights channel [22]
position 8 lights channel [36]
position 11 lights channel [47]
```

Inside the network we loaded, `w` is one number in one channel. The `w` is at position 8 and lights the eighth slot of the key zone, which begins right after the letters.

The `=` lights the eighth slot of the query zone. Both point at the same prompt position, and attention turns that match into a copy.

### Query Key Matching

Attention needs two things from every position: something to ask with, and something to be found by. Those are the query and the key, and the agent wrote both tables so that the two zones meet in one place:

```python
# runs/word-reversal-3x3-qwen37-max-s1/attempts/027_PosRouting_noW_o.py
# W_q: extract routing query from QUERY_START dims -> query dims 0..10
W_q_weight = torch.zeros(D, D)
for src_pos in range(11):
    W_q_weight[src_pos, QUERY_START + src_pos] = 5.0
attn.W_q.weight.data = W_q_weight

# W_k: extract position key from KEY_START dims -> key dims 0..10
W_k_weight = torch.zeros(D, D)
for pos in range(11):
    W_k_weight[pos, KEY_START + pos] = 5.0
attn.W_k.weight.data = W_k_weight
```

Both tables move one zone into the same first eleven channels and multiply it by five. `W_q` moves the where-to-look zone, and `W_k` moves the which-position zone.

After that, a query and a key share a channel only when the query names the key's position. We can work out the first answer step ourselves: the query of the last position, the key of every position, and the score attention gives each pair:

```python
attn = qwen.blocks[0].attn
h = qwen.token_emb(torch.tensor(task.encode("fnm_mod_wfd="))) + qwen.pos_emb.weight[:12]
q, k = attn.W_q(h[11]), attn.W_k(h)                             # the last position asks, every position answers
scores = (k @ q / 50 ** 0.5).detach()
print("scores :", [round(s, 2) for s in scores.tolist()])
print("softmax:", [round(a, 2) for a in scores.softmax(0).tolist()])

#### OUTPUT ####
scores : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 353.55, 0.0, 0.0, 0.0]
softmax: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
```

We get only one score, and it is a large one. The query and the key each hold ten times five, or fifty, in the shared channel.

Their product is 2,500, and dividing by the square root of the width leaves 353.55. Softmax turns a gap that large into a weight of one on that position and zero everywhere else, and attention looks at exactly one letter:

![The last position finds one letter](images/g-match.gif)

In the networks we usually train, attention is a soft blend of many positions. The agent turned it into a hard pointer by choosing weights large enough that only one position can win.

### Copying the Letter

Once attention points at a position, the value table decides what comes back from it. The agent's value table copies the letter zone and doubles it:

```python
# runs/word-reversal-3x3-qwen37-max-s1/attempts/027_PosRouting_noW_o.py
# W_v: copy token identity (dims 0..V-1)
W_v_weight = torch.zeros(D, D)
for i in range(V):
    W_v_weight[i, i] = 2.0
attn.W_v.weight.data = W_v_weight
```

The output head then reads the letter zone and multiplies it by five:

```python
# runs/word-reversal-3x3-qwen37-max-s1/attempts/027_PosRouting_noW_o.py
# Output head: read token identity from dims 0..V-1
head_weight = torch.zeros(V, D)
for i in range(V):
    head_weight[i, i] = 5.0
model.head.weight.data = head_weight
```

At the first answer step, two letters reach the head: the `w` that attention copied, and the `=` that the position already had. We can print the two highest scores:

```python
with torch.no_grad():
    out = qwen(torch.tensor([task.encode("fnm_mod_wfd=")]))[0, -1]    # one score per character, at the last position
best = out.topk(2)
for s, i in zip(best.values.tolist(), best.indices.tolist()):
    print(f"{task.decode([i])} scores {s:.1f}")

#### OUTPUT ####
w scores 100.0
= scores 50.0
```

The copied letter scores ten times two times five, a hundred. The position's own letter scores ten times five, fifty.

The copy wins by a factor of two at every step, and the next letter is always the one the route names:

![The copy outweighs the own letter](images/g-copy.gif)

Side by side, its eleven attention steps reverse the words:

![All eleven steps form a mirror](images/g-mirror.gif)

That is the whole network. We can run it on our full prompt and count its work the way we counted the template's:

```python
from wsp import verify

with FlopCounterMode(display=False) as count:
    answer = verify.answers(qwen, task, ["fnm_mod_wfd="])[0]
nonzero = sum(int((p != 0).sum()) for p in qwen.parameters())
print(f"answer {answer}, {count.get_total_flops():,d} operations")
print(f"{nonzero} of {sum(p.numel() for p in qwen.parameters()):,d} weights are not zero")

#### OUTPUT ####
answer wfd_mod_fnm, 3,986,400 operations
128 of 11,450 weights are not zero
```

We get the right answer, and the count is about 15 times smaller than the template's. Most of those operations still multiply a zero.

Only 29,172 of them involve a weight that is not zero, and we have now read every one of those weights:

![Each written weight has a job](images/g-jobs.gif)

Those seven jobs are the whole network, because every other weight in it is zero.

### Switch Off Test

An explanation is only useful if it is true. If the routing table is the real mechanism, cutting one route should break exactly one letter, and we should be able to say in advance which letter takes its place.

I wrote the explanation down as a program. It reads the routes off the weights and scores the letters the same way the head does:

```python
# programs.py
def routing(ids, off=(), steps=11, vocab=28):
    """Qwen's reversal network as a program, read off its weights: the routed letter is worth 100 and the letter at p
    50, a route switched off spreads its 100 evenly over positions 0..p, and the highest score wins."""
    ids = list(ids)
    for _ in range(steps):
        p = len(ids) - 1
        score = [0.0] * vocab
        score[ids[p]] += 50.0                                   # the residual carries the letter at p itself
        if "values" not in off:
            if p in ROUTES and p not in off:
                score[ids[ROUTES[p]]] += 100.0                  # the routed letter
            else:
                for j in range(p + 1):                          # no route: an even spread over positions 0..p
                    score[ids[j]] += 100.0 / (p + 1)
        ids.append(min(range(vocab), key=lambda v: (-score[v], v)))   # the lowest id wins a tie, as argmax does
    return ids[-steps:]
```

We just coded the routing program. At each answer step it gives the position's own letter fifty and the routed letter a hundred, as the head does.

With a route switched off, it spreads that hundred evenly over every position so far.

We can switch off the single weight that holds route 14 and run the network and our program side by side:

```python
from wsp import programs, switchoff

route_14 = ("pos_emb.weight", 14 * 50 + 39 + programs.ROUTES[14])     # the one weight that names step 14's source
with switchoff.off(qwen, params=[route_14]):
    print("network:", verify.answers(qwen, task, ["fnm_mod_wfd="])[0])
print("program:", task.decode(programs.routing(task.encode("fnm_mod_wfd="), off=(14,))))

#### OUTPUT ####
network: wfddmod_fnm
program: wfddmod_fnm
```

We get the same broken answer from both. The `_` after the first word became a second `d`, and our program named that letter without running the network at all:

![Cut one route and one letter breaks](images/g-switch.gif)

With the route cut, every letter so far takes an equal share, and the position's own `d` tips the balance:

![Why the broken letter is d](images/g-spread.gif)

I ran 13 cases like this on 2,000 hidden prompts each: every route switched off in turn, the copy switched off, and the network left whole.

The program predicted the network's exact answer on every prompt of every case. This test was exploratory, added after the runs and outside the predictions I sealed.

### Rebuild From Words

A second test asks whether the explanation is complete. In the paper's relay, one agent describes the network in at most 200 words, and fresh agents rebuild it from those words alone.

A control agent gets the task and no description at all. Qwen3.7-Max described its network in 151 words:

```bash
#### OUTPUT ####
A single-layer, single-head causal transformer (d_model=50, no MLP, LayerNorm, or output projection)
performs position-based token copying to reverse three 3-letter words. Token embeddings are one-hot
(scale 10) in dims 0 to 27. Position embeddings add a position key (one-hot in dims 28 to 38 for
positions 0 to 10) and a routing query (one-hot in dims 39 to 49, set only at prediction positions
11 to 21). Query and key projections (scale 5) extract these routing channels into dims 0 to 10. The
value projection (scale 2) copies token identity. Matching dot products reach 353 after sqrt-scaling
while mismatches are zero, making softmax a hard argmax. Prediction positions 11 to 21 are
hard-wired to attend to source positions 8-10, 7, 4-6, 3, 0-2, reversing the words and underscores.
The output head reads dims 0 to 27 (scale 5) to produce logits. Accuracy is near-perfect; extreme
attention peaks make autoregressive noise negligible.
```

Two fresh Qwen3.7-Max sessions read only that text. Each one built a network of exactly 11,450 weights on its first scored attempt, and both were perfect on all three prompt sets:

![Words rebuild the same network twice](images/g-rebuild.gif)

The control, with no words to go on, also reached a perfect score, but with 28.6 times the weights.

The description carried the design itself: two readers who had only the words built the very same network.

### Results on Ten Problems

Word reversal was one of 10 tasks I gave Qwen3.7-Max, two runs each. Our ledger shows each pick under the rule I sealed before the runs, with the score of a constant answer wherever it is not zero:

```bash
#### OUTPUT ####
problem         runs  calls  cut off  pick weights  hidden  floor
reversal           2    307        5        11,450  1.0000
addition           2    183       39    16,191,536  0.0060
gcd                2    283        7       151,872  0.6090 0.6090
dates              2    310        6        12,576  0.0000
smallest prime     2    277       12       151,744  0.0100
interpolation      1    123        7     3,166,208  0.0955
trec               2    277        3       828,672  0.4865 0.2255
sst-2              2    280        6        42,464  0.5585 0.5550
mrpc               2    299        3       161,408  0.6553 0.6882
snli               2    279        5       815,778  0.3535 0.3420
all               19   2618       93
```

Only one network in that table is right on every hidden prompt. I also stopped 5 more runs early, on problems I then dropped, to stay inside the budget:

![Ten problems and one solved](images/g-results.gif)

We can sort the other nine failures into five kinds:

1. **Constant answers:** the GCD and smallest prime factor networks give one answer to every prompt, and GCD's score is exactly what always answering `001` gets.
2. **Lookup tables:** for addition, the agent built a table of over sixteen million weights that is almost never right.
3. **Cut-off replies:** addition also had more replies run out of output while thinking than any other problem.
4. **No working design:** every dates network that built scored zero on the visible prompts, and the interpolation pick is a test the agent wrote to copy one input straight to the answer.
5. **Floors:** on the four text datasets, the networks stay close to always guessing the most common class, except the question-type network.

That floor is high because most pairs of numbers share no factor at all:

![Why a constant answer scores so well](images/g-coprime.gif)

We also audited every file the agent wrote for labels fitted by counting, a form of training. The data-use test rebuilds each file with shuffled labels and caught 28.

The transcripts showed 20 others that pasted label counts in as constants, the sentiment pick among them. No file the data-use test caught was a pick.

### Head to Head Comparison

So far we have followed one agent. The paper ran the same method with its own agent and published the networks it wrote, and I rechecked every one of them with the tools from this walkthrough. On the six generated problems both agents tried, we can set the two side by side:

```bash
#### OUTPUT ####
problem           qwen weights   hidden    paper weights   hidden
reversal                11,450   1.0000                3   1.0000
addition            16,191,536   0.0060               27   1.0000
gcd                    151,872   0.6090              332   1.0000
dates                   12,576   0.0000              345   1.0000
prime factor           151,744   0.0100            2,729   1.0000
interpolation        3,166,208   0.0955              864   1.0000
solved                1 of 6                   6 of 6
```

The paper's agent solved all six, with networks from three weights to under three thousand. Qwen3.7-Max solved one, and every network it failed with is larger than the paper's working one, the adder by a factor of almost six hundred thousand:

![Six problems, two agents, one board](images/g-board.gif)

The paper's agent also solved the seven generated problems that Qwen3.7-Max never finished, and none of its generated networks missed a single prompt I checked.

Text is where our comparison gets interesting, because no network is perfect there. The fair yardstick is a black box like the one we trained earlier: the same transformer, trained the usual way on the same dataset. I trained one for each of the four datasets and scored all three kinds of network on the same hidden examples:

```bash
#### OUTPUT ####
dataset   guessing     qwen   our black box    paper
trec        0.2255   0.4865          0.6870   0.8225
sst-2       0.5550   0.5585          0.6990   0.7865
mrpc        0.6882   0.6553          0.6882   0.6894
snli        0.3420   0.3535          0.5625   0.5790
```

The paper's hand-written networks score above our black boxes on all four datasets, by a hair on MRPC, the paraphrase dataset, where both are close to always guessing. Qwen3.7-Max's networks score below both:

![Hand-written networks against black boxes](images/g-podium.gif)

For explainable AI, this is the result I care about most. A network written by hand, with rules a person can follow, did not have to be worse than a trained black box, at least for the paper's agent.

The black boxes are exploratory, and all but the MRPC one trained on more labelled examples than the agent's pool of 2,000.

### Smallest Networks in the Paper

Much smaller networks came out of the paper's runs, and two of them are worth reading in full. The paper's word reversal network has just 3 weights. We can load it, run it on our prompt, and then switch one of its weights off:

```python
originals = config.PAPER / "experiments" / "nl-bottleneck" / "originals"     # the paper's own networks
_, tiny = networks.build(networks.load(originals / "word-reversal-3x3.py"), "word-reversal-3x3")
print(sum(p.numel() for p in tiny.parameters()), "weights:", tiny.reads.tolist(), verify.answers(tiny, task, ["fnm_mod_wfd="])[0])
with torch.no_grad():
    tiny.reads[2] = 0.0                                          # switch off the jump back at each new word
print("jump off:", tiny.reads.tolist(), verify.answers(tiny, task, ["fnm_mod_wfd="])[0])

#### OUTPUT ####
3 weights: [9.0, 1.0, -8.0] wfd_mod_fnm
jump off: [9.0, 1.0, 0.0] wfd_wfd_wfd
```

The paper's agent rebuilt attention itself. Each key is just a position counted from one, each value is a letter stored as one number, and attention picks the single position whose key is nearest the query. In the agent's own words, "Only the 3 query coefficients remain, at the cost of a non-standard attention rule." Those three numbers drive a reading head along our prompt:

![Three numbers read the prompt backwards](images/g-pointer.gif)

The head starts at the ninth position, steps one position right for each character, and jumps back eight positions at each new word. With the jump switched off, the head never moves back. It reads the last word, then walks on into its own answer and reads that word twice more, which is exactly what our output shows.

The paper's GCD network is a bigger idea in 332 weights. It runs Euclid's algorithm, more than two thousand years old, by folding: each fold makes one number smaller or leaves it alone, and never changes the greatest common divisor the two numbers share. We can run it on one pair:

```python
gcd_task, folder = networks.build(networks.load(originals / "gcd-three-digits.py"), "gcd-three-digits")
print(sum(p.numel() for p in folder.parameters()), "weights:", "084,036= gives", verify.answers(folder, gcd_task, ["084,036="])[0])

#### OUTPUT ####
332 weights: 084,036= gives 012
```

That is the right answer, written with three digits as the task asks. We can watch the network fold the first number down, swap the two numbers, and repeat until one of them is zero:

![Folding two numbers down to their GCD](images/g-folds.gif)

A trick like this could work on some inputs and fail on others. So I ran the network on every one of the 998,001 pairs the task allows:

![Every pair checked, none wrong](images/g-allinputs.gif)

Not one of them came out wrong. We cannot test a network harder than that, because there is no input left for it to fail on.

### Known Limits

Seven things narrow what we should take from these results:

- **One agent model:** every run used Qwen3.7-Max, with two runs per problem and one for interpolation.
- **A fixed budget:** each run had the same budget of agent work, and the model's output price is known only as a range.
- **A paused night:** an overnight pause stopped 4 runs mid-session, and I rebuilt their missing session records from the call logs before they resumed.
- **Near ties:** rescoring reproduced 613 logged scores, and 15 came out differently, from networks whose top outputs nearly tie.
- **Exploratory tests:** the trained adder, the text black boxes and the switch-off test came after the runs, outside the predictions I sealed.
- **Tidied code:** the package in this walkthrough was cleaned up after the runs, and its tests and reruns show the same behaviour.
- **The paper's networks:** I never ran the paper's agent. Every network credited to the paper here is the paper's own, and I only checked it.

The paper is open about its own limits. It says that "Programming deep hierarchical representations without backpropagation remains an open problem", and that the method "relies on an agent's written explanations remaining faithful to its weight edits". Our switch-off test is one way to check that second point on a real network.

An agent cannot write the weights of GPT-3 today. For small networks, though, explainable AI by design gives us what training never does: a network whose every number has a reason we can read. The agent loop, the checks and every network it wrote are in this repository.

## Citation

If this repository helps your work, please cite it together with the paper it builds on:

```bibtex
@misc{khan2026explainableweights,
      title={Explainable Weights: an AI agent writes every weight of a transformer},
      author={Fareed Khan},
      year={2026},
      url={https://github.com/FareedKhan-dev/explainable-weights}
}

@misc{mehta2026interpretableweightspaceprogrammingbuilds,
      title={Interpretable weight-space programming builds transparent transformers and brain encoding models},
      author={Yash Mehta and Chandan Singh and Weijia Xu and Richard J. Antonello and Adeen Flinker and Nima Mesgarani and Michael F. Bonner and Nebojsa Jojic and Jianfeng Gao},
      year={2026},
      eprint={2611.18114},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2611.18114}
}
```

## License

This repository is released under the [MIT License](LICENSE). `wsp/model.py`, `wsp/tasks.py` and `wsp/evaluate.py` are
ports of the paper's MIT-licensed code, and every network under `runs/` starts from the paper's template. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The text datasets are not included: the scripts download them, under
their own terms.

## Acknowledgements

The method, the problems, the evaluator and the template network are the work of the paper's authors, who published
their code, prompts and logs. This repository only reruns their loop with another agent and adds the explainability
tests.
