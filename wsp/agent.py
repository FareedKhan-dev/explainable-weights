"""wsp/agent.py: the weight-writing loop. One agent, one problem, one run folder, until the paper's target is met.

The prompts, the attempt targets and the stopping rule are the paper's own, imported from its launcher, so the only
thing that changes is who writes the weights. A run stops when it reaches its target, when two sessions in a row add
no attempt, after 25 sessions, or when a cap (calls, hours, the shared credit budget, a STOP file) is reached."""
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone

from wsp import config
from wsp.client import AccessDenied, ChatClient, QuotaExhausted
from wsp.sandbox import Sandbox, attempts, make_run

def paper_launcher():
    """The paper's agent launcher: the one script in experiments/evolve that defines its prompts (FIRST, CONT, HINT),
    its problem list (TASKS), its attempt targets (TARGET) and its session cap (MAX_SESSIONS). It is found by what it
    defines, and the prompts are used word for word."""
    for path in sorted(config.EVOLVE.glob("launch_*.py")):
        text = path.read_text(encoding="utf-8")
        if all(f"\n{name} = " in text for name in ("FIRST", "CONT", "HINT", "MAX_SESSIONS")):
            spec = importlib.util.spec_from_file_location("paper_launcher", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"no launcher with the paper's prompts in {config.EVOLVE}: set WSP_PAPER to the paper's clone")


sys.path.insert(0, str(config.EVOLVE))
paper = paper_launcher()  # FIRST, CONT, HINT, TASKS, TARGET, MAX_SESSIONS

SYSTEM = ("You are an autonomous coding agent. You work inside one run folder through the tools you are given: read, "
          "edit, write, glob, grep and bash. Paths are relative to the run folder. The bash tool runs only "
          "`uv run interpretable_transformer.py ...`. Keep working with tool calls; a reply without a tool call ends "
          "your session.")
CAPS = ("stop file", "call cap", "time cap", "credit cap", "run credit cap")

# a reply cut off at the output limit before any tool call did nothing, and the agent did not choose to stop: say so
# and let it go on, as the paper's own harness does when a reply hits its output limit
MAX_OUT, MAX_CUT_OFFS = 32000, 3
CUT_OFF = ("Your last reply reached the output limit of {limit:,} tokens while you were still thinking, before you made a "
           "tool call, so nothing was done. Resume directly: think in shorter steps, make your next tool call now, and "
           "write large files in parts.")


def tool(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


S, I = {"type": "string"}, {"type": "integer"}
TOOLS = [
    tool("read", "Read a text file in the run folder. Lines come back numbered.",
         {"file_path": S, "offset": I, "limit": I}, ["file_path"]),
    tool("write", "Create or overwrite a file in the run folder.", {"file_path": S, "content": S}, ["file_path", "content"]),
    tool("edit", "Replace an exact string in a file. old_string must occur once unless replace_all is true.",
         {"file_path": S, "old_string": S, "new_string": S, "replace_all": {"type": "boolean"}},
         ["file_path", "old_string", "new_string"]),
    tool("glob", "List files in the run folder that match a pattern such as **/*.py.", {"pattern": S}, ["pattern"]),
    tool("grep", "Search files in the run folder for a regular expression.", {"pattern": S, "path": S}, ["pattern"]),
    tool("bash", "Run a command in the run folder. Only `uv run interpretable_transformer.py [args]` is allowed.",
         {"command": S, "timeout": I}, ["command"]),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def credits(log, price=config.PRICE_PER_M_INPUT, out_weight=config.OUTPUT_WEIGHT):
    """Credits spent by the calls in one calls.jsonl, from their tokens at the measured price."""
    if not log.exists():
        return 0.0
    tokens = 0
    for line in open(log, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:                       # a line another run is still writing, counted next time
            continue
        tokens += (r.get("input") or 0) + out_weight * (r.get("output") or 0)
    return tokens * price / 1e6


class Limits:
    """The caps one run checks before each call: a stop file, calls, hours, and credits for the run and the study."""

    def __init__(self, root, max_calls=config.MAX_CALLS, max_hours=config.MAX_HOURS, credit_cap=config.CREDIT_CAP,
                 run_credit_cap=config.RUN_CREDIT_CAP, compact_at=config.COMPACT_AT, stop_file=None, budget_root=None):
        self.root, self.max_calls, self.max_hours = root, max_calls, max_hours
        self.credit_cap, self.run_cap, self.compact_at = credit_cap, run_credit_cap, compact_at
        self.stop_file = stop_file or root.parent / "STOP"
        self.budget_root = budget_root or root              # every call logged anywhere under it counts against the cap
        self.t0, self.log = time.time(), None              # the run sets log to its own calls.jsonl

    def spent(self):
        """Credits so far: this run, and every run under the budget root together."""
        mine = credits(self.log) if self.log else 0.0
        return mine, sum(credits(p) for p in self.budget_root.glob("**/agent_logs/calls.jsonl"))

    def reached(self, calls):
        if self.stop_file.exists():
            return "stop file"
        if calls >= self.max_calls:
            return "call cap"
        if time.time() - self.t0 > self.max_hours * 3600:
            return "time cap"
        mine, everyone = self.spent()
        if mine >= self.run_cap:
            return "run credit cap"
        if everyone >= self.credit_cap:
            return "credit cap"
        return ""


def assistant_message(reply, call_no):
    """The reply as it goes back into the conversation, with an id on every tool call."""
    calls = [dict(t, id=t["id"] or f"call_{call_no}_{i}") for i, t in enumerate(reply.tool_calls)]
    msg = {"role": "assistant", "content": reply.text or ""}
    if calls:
        msg["tool_calls"] = [{"id": t["id"], "type": "function",
                              "function": {"name": t["name"], "arguments": t["arguments"]}} for t in calls]
    return msg, calls


def run_tools(box, calls, used):
    """Carry out each tool call and return the tool messages for the conversation."""
    out = []
    for t in calls:
        used[t["name"]] = used.get(t["name"], 0) + 1
        try:
            args = json.loads(t["arguments"] or "{}")
            text = box.call(t["name"], args) if t["name"] in TOOL_NAMES else f"Error: no tool named {t['name']}"
        except json.JSONDecodeError:
            text = "Error: the tool arguments were not valid JSON (the reply may have been cut off, write less at once)"
        out.append({"role": "tool", "tool_call_id": t["id"], "content": text})
    return out


def call_model(client, box, messages, state, s, logs, tools):
    """One model call, counted and logged. None when the account itself refuses (quota or access)."""
    try:
        r = client.chat(messages, effort=state["effort"], max_completion_tokens=MAX_OUT, extra={"tools": tools})
    except (QuotaExhausted, AccessDenied) as e:
        s["end"] = f"stopped: {type(e).__name__}"
        logs.note(str(e))
        return None
    state["calls"] += 1
    s["calls"] += 1
    for f in ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens"):
        s[f] += getattr(r, f) or 0
    logs.call(s["k"], state["calls"], r, attempts(box.dir))
    return r


def stopped(s, r):
    """Why a failed call ends the session, and whether a fresh session follows (only after a context error)."""
    if r is None:
        return False
    context = any(w in r.error.lower() for w in ("context", "length", "too long", "maximum", "tokens"))
    s["end"] = "context limit" if context else "infra: " + r.error[:200]
    return context


def cut_off(messages, msg, r, s, logs, cuts):
    """Keep what a cut-off reply said and add a note asking for shorter steps. False once MAX_CUT_OFFS notes failed."""
    s["cut_offs"] += 1
    if msg["content"]:
        messages.append(msg)
        logs.turn(s["k"], msg, len(r.reasoning))
    if cuts > MAX_CUT_OFFS:
        s["end"] = "cut off"
        return False
    note = {"role": "user", "content": CUT_OFF.format(limit=MAX_OUT)}
    messages.append(note)
    logs.write(f"session_{s['k']:02d}.jsonl", dict(note, harness="cut off"))
    return True


def act(box, calls, s, logs):
    """Carry out the reply's tool calls and log them, returning the tool messages for the conversation."""
    tool_msgs = run_tools(box, calls, s["tools_used"])
    logs.tools(s["k"], calls, tool_msgs)
    return tool_msgs


def finish(s, end, fresh):
    """Record why the session ended, and whether a fresh session should follow."""
    s["end"] = end
    return s, fresh


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


class Logs:
    """calls.jsonl and session_NN.jsonl, with the key removed from anything written."""

    def __init__(self, folder, key):
        self.dir, self.key = folder / "agent_logs", key

    def clean(self, s):
        return s.replace(self.key, "<key>") if self.key and isinstance(s, str) else s

    def write(self, name, rec):
        with open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(self.clean(json.dumps(rec)) + "\n")

    def call(self, k, n, r, attempts_now):
        self.write("calls.jsonl", dict(t=datetime.now(timezone.utc).isoformat(), session=k, call=n,
                                       seconds=round(r.seconds, 1), status=r.status, stop=r.stop_reason, model=r.model,
                                       input=r.input_tokens, output=r.output_tokens, reasoning=r.reasoning_tokens,
                                       cached=r.cached_tokens, attempts_now=attempts_now,
                                       tools=[t["name"] for t in r.tool_calls], error=r.error or None))

    def turn(self, k, msg, reasoning_chars):
        self.write(f"session_{k:02d}.jsonl", dict(msg, reasoning_chars=reasoning_chars))

    def tools(self, k, calls, tool_msgs):
        for t, m in zip(calls, tool_msgs):
            self.write(f"session_{k:02d}.jsonl", dict(role="tool", name=t["name"], content=m["content"][:4000]))

    def note(self, text):
        self.write("notes.jsonl", dict(t=datetime.now(timezone.utc).isoformat(), note=text))


def run(task, seed, client, root=config.RUNS, limits=None, effort=None, key=""):
    """Run the paper's loop for one problem with one agent, and return the run folder."""
    model = getattr(client, "model", "agent")
    folder = make_run(task, root / f"{task}-{model.split('/')[-1].lower().replace('.', '')}-s{seed}")
    target, limits = paper.TARGET[paper.TASKS[task]], limits or Limits(root)
    limits.log = folder / "agent_logs" / "calls.jsonl"
    meta_path = folder / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else dict(
        run=folder.name, task=task, seed=seed, model=model, effort=effort or "default", target=target,
        price=dict(per_m_input=config.PRICE_PER_M_INPUT, output_weight=config.OUTPUT_WEIGHT),
        caps=dict(run_credits=limits.run_cap, study_credits=limits.credit_cap, compact_at=limits.compact_at),
        sessions=[], started=datetime.now(timezone.utc).isoformat())
    box, logs = Sandbox(folder), Logs(folder, key)
    state = dict(calls=sum(s["calls"] for s in meta["sessions"]), effort=effort)
    messages, stalls, failures = None, 0, 0
    while len(meta["sessions"]) < paper.MAX_SESSIONS:
        n0 = attempts(folder)
        if n0 >= target:
            break
        if messages is None:                        # a new conversation: the first prompt, or the continue prompt
            opener = paper.FIRST if not meta["sessions"] else paper.CONT
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": opener.format(task=task, target=target, n=n0, hint=paper.HINT)}]
        elif failures:                              # the API failed: the same conversation again, after a wait
            time.sleep(min(600, 60 * 2 ** (failures - 1)))
            logs.note(f"session {len(meta['sessions'])} retries the conversation after {failures} API failure(s)")
        else:                                       # the same conversation, resumed as the paper's driver does
            messages.append({"role": "user", "content": paper.CONT.format(task=task, target=target, n=n0)})
        if not failures:
            logs.write(f"session_{len(meta['sessions']):02d}.jsonl", dict(role="user", content=messages[-1]["content"]))
        t0 = time.time()
        s, fresh = session(client, box, messages, limits, len(meta["sessions"]), state, logs)
        s.update(seconds=round(time.time() - t0), attempts_before=n0, attempts_after=attempts(folder),
                 refused_calls=box.refused, run_credits_so_far=round(limits.spent()[0], 2))
        meta["sessions"].append(s)
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"{folder.name} session {s['k']}: {n0} -> {s['attempts_after']} attempts, {s['calls']} calls, "
              f"{s['seconds']}s, end: {s['end']}", flush=True)
        if s["end"] in CAPS or s["end"].startswith("stopped"):
            break
        failures = failures + 1 if s["end"].startswith("infra") else 0
        if failures >= 5:                           # a long outage: stop here, the run resumes from its folder later
            break
        messages = None if fresh else messages
        if s["attempts_after"] > n0:
            stalls = 0
        elif s["end"] == "end of turn":            # only a session the agent ended itself counts as a stall
            stalls += 1
        if stalls >= 2:
            break
    meta.update(finished=datetime.now(timezone.utc).isoformat(), attempts=attempts(folder), calls=state["calls"],
                credits=round(limits.spent()[0], 2))
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return folder


def connect():
    """The live client for the configured agent model, at the endpoint the runner names in WSP_ENDPOINT."""
    if not config.ENDPOINT:
        raise SystemExit("WSP_ENDPOINT is not set, so there is no chat endpoint to call")
    key = config.KEY_FILE.read_text(encoding="utf-8").strip()
    return ChatClient(config.ENDPOINT, key, config.AGENT_MODEL), key
