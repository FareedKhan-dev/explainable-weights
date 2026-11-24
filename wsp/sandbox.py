"""wsp/sandbox.py: one run folder and the restricted tools the agent works through.

The agent may read and edit files inside its folder and run one command, the evaluator. It cannot read the task
code in src/, step outside the folder, or chain shell commands. Every evaluated file is kept, in order, so no attempt
is lost even when the agent forgets to save it."""
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from wsp import config

NAME_RX = re.compile(r"""^model_shorthand_name\s*=\s*["'](.+?)["']""", re.M)
ACC_RX = re.compile(r"^accuracy:\s+([0-9.]+)\s+\((\d+)/(\d+)\)", re.M)
RUN_CMD = ["uv", "run", "interpretable_transformer.py"]
CHAINING = (">", "<", "|", ";", "&", "`", "$(")


def make_run(task, folder, program=True):
    """A fresh run folder laid out like the paper's, with program.md naming the task (a rebuild gets none)."""
    if folder.exists():
        return folder
    folder.mkdir(parents=True)
    shutil.copytree(config.EVOLVE / "src", folder / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(config.EVOLVE / "interpretable_transformer.py", folder / "interpretable_transformer.py")
    shutil.copytree(config.EVOLVE / "results", folder / "results")
    if program:
        text = (config.EVOLVE / "program.md").read_text(encoding="utf-8")
        text = re.sub(r"your task name is `[^`]+`", f"your task name is `{task}`", text, count=1)
        (folder / "program.md").write_text(text, encoding="utf-8", newline="\n")
    for sub in ("interpretable_transformers_lib", "agent_logs", "attempts"):
        (folder / sub).mkdir()
    return folder


def attempts(folder):
    """Attempts as the paper counts them: one row per model name in the results file (none before the first run)."""
    path = folder / "results" / "overall_results.csv"
    if not path.exists():
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def evaluator_env():
    """The evaluator's environment: UTF-8 output (it prints an arrow) and no key or token from ours."""
    env = {k: v for k, v in os.environ.items() if "KEY" not in k.upper() and "TOKEN" not in k.upper()}
    env.update(HF_HOME=str(config.HF_HOME), HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return env


def allowed(command):
    """The command split into words, or a refusal: only `uv run interpretable_transformer.py [args]` may run."""
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = []
    if parts[:3] != RUN_CMD or any(c in command for c in CHAINING):
        raise PermissionError("only `uv run interpretable_transformer.py [args]` may run, with no redirection or chaining")
    return parts


class Sandbox:
    """The tools of one run: read, write, edit, glob, grep and bash, all held inside the folder."""

    def __init__(self, folder, src_readable=False, max_runs=None):
        self.dir = Path(folder).resolve()
        self.refused, self.runs = 0, 0
        self.src_readable, self.max_runs = src_readable, max_runs     # a rebuild may read src/, and scores 6 times
        self.src_hash = self.hash_src()

    def hash_src(self):
        h = hashlib.sha256()
        for p in sorted((self.dir / "src").rglob("*.py")):
            h.update(p.read_bytes())
        return h.hexdigest()

    def path(self, p, write=False):
        """Resolve a path the agent gave, refusing anything outside its workspace."""
        q = (self.dir / p).resolve()
        if self.dir not in q.parents and q != self.dir:
            raise PermissionError(f"{p} is outside the run folder")
        rel = q.relative_to(self.dir).as_posix()
        if (rel == "src" or rel.startswith("src/")) and (write or not self.src_readable):
            raise PermissionError(f"{p} is a link to a folder outside the run folder; the file tools cannot reach it"
                                  if not self.src_readable else f"{p} is read-only task and scoring code")
        if rel.startswith(("attempts", "agent_logs")):
            raise PermissionError(f"{p} belongs to the harness and is not part of your workspace")
        if write and rel == "program.md":
            raise PermissionError("program.md is a link outside the run folder and cannot be changed")
        return q

    def call(self, name, args):
        """Run one tool call and return its text, counting every refusal."""
        try:
            return getattr(self, "t_" + name)(**args)
        except PermissionError as e:
            self.refused += 1
            return f"Error: permission denied: {e}"
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def t_read(self, file_path, offset=None, limit=None):
        lines = self.path(file_path).read_text(encoding="utf-8", errors="replace").splitlines()
        a = max(1, offset or 1)
        b = min(len(lines), a - 1 + (limit or 2000))
        return "\n".join(f"{i:6d}\t{lines[i - 1][:2000]}" for i in range(a, b + 1)) or "(empty file)"

    def t_write(self, file_path, content):
        q = self.path(file_path, write=True)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(content, encoding="utf-8", newline="\n")
        return f"Wrote {len(content.splitlines())} lines to {file_path}"

    def t_edit(self, file_path, old_string, new_string, replace_all=False):
        q = self.path(file_path, write=True)
        text = q.read_text(encoding="utf-8")
        n = text.count(old_string)
        if n == 0:
            return "Error: old_string was not found in the file"
        if n > 1 and not replace_all:
            return f"Error: old_string occurs {n} times; give more context or set replace_all"
        q.write_text(text.replace(old_string, new_string, -1 if replace_all else 1), encoding="utf-8", newline="\n")
        return f"Edited {file_path} ({n if replace_all else 1} replacement)"

    def shown(self, p):
        rel = p.relative_to(self.dir).as_posix()
        return p.is_file() and not rel.startswith(("src/", "agent_logs/", "attempts/"))

    def t_glob(self, pattern):
        hits = [p.relative_to(self.dir).as_posix() for p in self.dir.glob(pattern) if self.shown(p)]
        return "\n".join(sorted(hits)[:500]) or "No files found"

    def t_grep(self, pattern, path="."):
        rx, out, base = re.compile(pattern), [], self.path(path)
        for f in [base] if base.is_file() else [p for p in base.rglob("*") if self.shown(p)]:
            if f.suffix in (".png", ".pyc"):
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    out.append(f"{f.relative_to(self.dir).as_posix()}:{i}:{line[:500]}")
        return "\n".join(out[:300]) or "No matches"

    def t_bash(self, command, timeout=None):
        """Run the evaluator, and only the evaluator, then keep the file it scored."""
        parts = allowed(command)
        if self.max_runs is not None and self.runs >= self.max_runs:
            raise PermissionError(f"all {self.max_runs} scored runs are used, so name your final snapshot now")
        self.runs += 1
        limit = min(600, max(10, int(timeout or 120)))
        source, t0 = (self.dir / "interpretable_transformer.py").read_bytes(), time.time()
        try:
            p = subprocess.run([str(config.PYTHON), "interpretable_transformer.py", *parts[3:]], cwd=self.dir,
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=limit,
                               env=evaluator_env())
            out, code = (p.stdout + p.stderr).strip(), p.returncode
        except subprocess.TimeoutExpired as e:
            out, code = ((e.stdout or "") + (e.stderr or "")) if isinstance(e.stdout, str) else "", f"timeout after {limit}s"
        self.keep_attempt(source, parts[3:], out, code, time.time() - t0)
        if self.hash_src() != self.src_hash:
            raise RuntimeError("src/ changed during a run, so the run is void")
        if len(out) > 30000:
            out = out[:15000] + "\n[output truncated]\n" + out[-15000:]
        return f"exit code {code}\n{out}"

    def keep_attempt(self, source, args, out, code, seconds):
        """Save the scored file as attempts/NNN_<name>.py and log what the evaluator said about it."""
        log = self.dir / "attempts" / "attempts.jsonl"
        idx = (sum(1 for _ in open(log, encoding="utf-8")) if log.exists() else 0) + 1
        acc, row = ACC_RX.search(out), {}
        if acc:                                     # the evaluator writes this attempt's row last
            with open(self.dir / "results" / "overall_results.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows and rows[-1].get("accuracy") and abs(float(rows[-1]["accuracy"]) - float(acc.group(1))) < 1e-9:
                row = rows[-1]
        m = NAME_RX.search(source.decode("utf-8", errors="replace"))
        name = row.get("model_shorthand_name") or (m.group(1) if m else "unnamed")
        (self.dir / "attempts" / f"{idx:03d}_{re.sub(r'[^A-Za-z0-9_.]+', '_', name)[:80]}.py").write_bytes(source)
        rec = dict(idx=idx, t=datetime.now(timezone.utc).isoformat(), name=name, args=args, exit=code,
                   seconds=round(seconds, 1), accuracy=float(acc.group(1)) if acc else None,
                   correct=int(acc.group(2)) if acc else None, n=int(acc.group(3)) if acc else None,
                   n_params=float(row["n_params"]) if row.get("n_params") else None,
                   description=row.get("description"), sha256=hashlib.sha256(source).hexdigest())
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
