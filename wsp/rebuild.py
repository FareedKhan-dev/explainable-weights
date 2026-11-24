"""wsp/rebuild.py: the rebuild test. If a 200-word description carries the network, a fresh agent can rebuild it.

The protocol and the prompts are the paper's (experiments/nl-bottleneck/03_summarize.py and 04_reconstruct.py):
  describe: an agent with only the read tool and only the network file writes at most 200 words,
  rebuild:  two fresh agents get the task, the description, the original's score and size, and 6 scored runs,
  control:  one fresh agent gets the task alone, with the same budget.
Each rebuild's final network is its best visible score, ties going to fewer weights, the rule used for originals.
Like the paper's single rebuild session, a conversation here never starts over: it runs until the agent stops, a cap
is reached, or its prompt passes CONTEXT_BOUND tokens."""
import json
import re
import shutil
import time

from wsp import config
from wsp.agent import SYSTEM, TOOLS, Limits, Logs, credits, session
from wsp.sandbox import Sandbox

MAX_EXP, CONTEXT_BOUND = 6, 150_000
DESCRIBE_SYSTEM = ("You read files in your working folder with the read tool, the only tool you have. Paths are relative "
                   "to that folder. When you are done, reply with your answer and no tool call.")
DESCRIBE = """Below is a task, and a file model.py in your working directory. model.py is a small transformer whose weights were written by hand (no training) to solve the task; the file builds the model and sets every weight in closed form.
TASK: {name}
{doc}
Prompts are fed token by token (character level); the model generates the answer autoregressively with greedy decoding and is scored by exact match.
Read model.py. Then write a description of what this model learned/implements, for an engineer who will have to rebuild a model with similar accuracy AND similar mistakes from scratch, without ever seeing model.py or its numbers. Rules:
- At most 200 words, plain prose, no code blocks, no pasted tensors.
- Describe the algorithm the network implements (what attention does, what the MLP or readout does, how the answer is produced), not the raw weights. Practical hints are welcome (e.g. "look up bigrams", which features or words matter, thresholds, scale constants that matter, how many layers/heads).
- Say where the model is known to be imperfect or approximate, if it is, so its error pattern can be matched.
Output only the description."""
SHORTEN = "That is {n} words. Rewrite it in at most 190 words, keeping the most important information. Output only the description."
HEAD_RECON = """You are rebuilding a hand-written transformer. An earlier agent wrote, with no training, the weights of a small transformer for the task `{task}`. You will not see that model. All you get is the description below, written by another engineer who did read it. Your job is to hand-write a transformer that implements the same algorithm, so that its accuracy AND its mistakes resemble the original's.
TASK:
{doc}
Prompts are fed token by token (character level); the model generates the answer autoregressively with greedy decoding; score = exact-match accuracy on 200 fixed examples (seed 0). The original model scored {vis:.1%} on these 200 examples and has {npar:,} parameters.
DESCRIPTION OF THE ORIGINAL MODEL ({nw} words):
\"\"\"
{summary}
\"\"\"
"""
HEAD_CONTROL = """You are hand-writing the weights of a small transformer, with no training, for the task `{task}`.
TASK:
{doc}
Prompts are fed token by token (character level); the model generates the answer autoregressively with greedy decoding; score = exact-match accuracy on 200 fixed examples (seed 0). Goal: as high an accuracy as you can get within {maxexp} experiments.
"""
TAIL = """
Your working folder has interpretable_transformer.py (a template transformer; edit the architecture and `write_weights(model, task)` so every parameter is set directly in closed form), src/task.py and src/eval.py (read-only task and scoring code), and results/.
Rules:
- NO training of any kind: no gradient steps, optimizers, fitting loops, least squares or searches over weights. Parameters are set in closed form from the algorithm.
- Do not compute the answer with Python outside the transformer; the network must produce it autoregressively. Do not use pretrained weights.
- Do not fit weights, lexicons or thresholds to examples or labels from the dataset (for dataset tasks write any word lists from your own knowledge{guide}). Do not read anything outside this folder.
- Only edit interpretable_transformer.py and write snapshots; do not modify src/.
Procedure: implement your approach, then score it with exactly `uv run interpretable_transformer.py --task {task}` (set `model_shorthand_name` and `model_description` in the file before each run; each run appends a row to results/overall_results.csv). Run at most {maxexp} experiments in total. After each run save a copy of the file to interpretable_transformers_lib/<model_shorthand_name>.py with the Write tool. Use the experiments to fix bugs and improve the model.{end}"""
END_RECON = " Do not tune it to beat the original: the goal is faithfulness, so do not 'fix' limitations the description says the original has. Finish with a one-line statement of which snapshot is your final model."
END_CONTROL = " Finish with a one-line statement of which snapshot is your final model."
READ_ONLY = [t for t in TOOLS if t["function"]["name"] == "read"]


def task_doc(task_name):
    from wsp.networks import paper_code
    tasks, _ = paper_code()
    return re.sub(r"\n +", "\n", tasks.get_task(task_name).__doc__.strip())


def converse(client, box, prompt, limits, key, system=SYSTEM, tools=TOOLS, follow_ups=()):
    """One conversation in a sandbox: the prompt, the tool calls it leads to, and any follow-up requests.
    Returns the messages, the number of calls, and how each stretch of the conversation ended."""
    limits.log = box.dir / "agent_logs" / "calls.jsonl"
    messages, state, logs, ends = [{"role": "system", "content": system}, {"role": "user", "content": prompt}], \
        dict(calls=0, effort=None), Logs(box.dir, key), []
    logs.write("session_00.jsonl", dict(role="user", content=prompt))
    for follow in (None, *follow_ups):
        if follow is not None:                      # a follow-up request, only after the agent ended its turn
            text = follow(messages)
            if not text:
                break
            messages.append({"role": "user", "content": text})
            logs.write("session_00.jsonl", dict(role="user", content=text))
        for failures in range(5):                   # an API failure: the same conversation again, after a wait
            s, _ = session(client, box, messages, limits, 0, state, logs, tools)
            ends.append(s["end"])
            if not s["end"].startswith("infra"):
                break
            time.sleep(min(600, 60 * 2 ** failures))
        if s["end"] != "end of turn":
            break
    return messages, state["calls"], ends


def save_meta(folder, limits, **meta):
    meta["credits"] = round(credits(limits.log), 2)
    (folder / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")


def describe(client, network_file, task_name, folder, key="", limits=None):
    """At most 200 words about the network, from an agent that may only read it."""
    folder.mkdir(parents=True, exist_ok=True)
    for sub in ("agent_logs", "attempts", "results"):
        (folder / sub).mkdir(exist_ok=True)
    shutil.copy2(network_file, folder / "model.py")
    box, limits = Sandbox(folder), limits or Limits(folder.parent, max_calls=20, compact_at=CONTEXT_BOUND)
    words = lambda m: len((m[-1].get("content") or "").split())
    shorten = lambda m: SHORTEN.format(n=words(m)) if words(m) > 200 else ""
    messages, calls, ends = converse(client, box, DESCRIBE.format(name=task_name, doc=task_doc(task_name)),
                                     limits, key, DESCRIBE_SYSTEM, READ_ONLY, [shorten] * 3)
    text = (messages[-1].get("content") or "").strip() if messages[-1]["role"] == "assistant" else ""
    (folder / "description.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
    save_meta(folder, limits, kind="describe", task=task_name, network=str(network_file), words=len(text.split()),
              calls=calls, ends=ends)
    return text, calls


def rebuild(client, task_name, description, visible, n_params, folder, key="", control=False, limits=None):
    """One fresh agent hand-writes the network again, from the description, or from the task alone as the control."""
    from wsp.sandbox import make_run
    make_run(task_name, folder, program=False)
    doc = task_doc(task_name)
    head = (HEAD_CONTROL.format(task=task_name, doc=doc, maxexp=MAX_EXP) if control else
            HEAD_RECON.format(task=task_name, doc=doc, vis=visible, npar=int(n_params), nw=len(description.split()), summary=description))
    prompt = head + TAIL.format(task=task_name, maxexp=MAX_EXP, guide="" if control else ", guided by the description",
                                end=END_CONTROL if control else END_RECON)
    box = Sandbox(folder, src_readable=True, max_runs=MAX_EXP)
    limits = limits or Limits(folder.parent, max_calls=80, compact_at=CONTEXT_BOUND)
    messages, calls, ends = converse(client, box, prompt, limits, key)
    best = final(folder)
    save_meta(folder, limits, kind="control" if control else "rebuild", task=task_name, calls=calls, ends=ends,
              final=best, words_given=0 if control else len(description.split()))
    return best, calls


def default_run(args):
    """True when a scored run used the evaluator's default visible set (seed 0, 200 examples). --verbose only prints
    more, so it does not change the set."""
    toks = [t for a in args or [] if a != "--verbose" for t in a.split("=", 1)]
    opts = dict(zip(toks[::2], toks[1::2])) if len(toks) % 2 == 0 else {"bad": ""}
    return set(opts) <= {"--task", "--seed", "--n-samples", "--device"} and "--task" in opts \
        and opts.get("--seed", "0") == "0" and opts.get("--n-samples", "200") == "200"


def final(folder):
    """The best visible score among the attempts, ties going to fewer weights: the rebuild's final network.
    Only runs on the default visible set count, so a score from fewer examples can never win."""
    log = folder / "attempts" / "attempts.jsonl"
    rows = [json.loads(l) for l in open(log, encoding="utf-8")] if log.exists() else []
    scored = [r for r in rows if r["accuracy"] is not None and r["n_params"] is not None and default_run(r["args"])]
    return max(scored, key=lambda r: (r["accuracy"], -r["n_params"])) if scored else None
