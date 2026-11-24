"""scripts/run_study.py: run several problems and seeds through scripts/run_agent.py, a few at a time, each in its own process.

    python scripts/run_study.py --tasks all --seeds 1 2 --parallel 3 --study main
    python scripts/run_study.py --jobs gcd-three-digits:2 word-reversal-3x3:1 --study main

Each run writes runs/<study>/<task>-s<seed>.out (its console) next to its folder. The study cap, each run's own cap and
the STOP file still apply, because every run checks them before each call. A run whose folder exists resumes from it.
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the paper's 17 problems in its launcher's order: easy, harder, real
PROBLEMS = ["addition-five-digits", "multiplication-five-digits", "sort-five-digits", "digit-counting-10",
            "parity-upto10-bits", "boolean-circuit-5-bits", "linear-interpolation-two-points",
            "word-reversal-3x3", "gcd-three-digits", "decimal-to-binary-8bit", "edit-distance-8", "smallest-prime-factor",
            "sentiment-sst2", "paraphrase-mrpc", "nli-snli", "trec-coarse", "date-normalization"]


def one(job):
    task, seed, a = job
    out = ROOT / "runs" / a.study / f"{task}-s{seed}.out"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as fh:
        p = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_agent.py"), "--task", task, "--seed", str(seed),
                            "--study", a.study, "--run-cap", str(a.run_cap), "--study-cap", str(a.study_cap)],
                           stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    return f"{task} s{seed}: exit {p.returncode} after {round((time.time() - t0) / 60, 1)} min"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--study", default="main")
    ap.add_argument("--jobs", nargs="*", default=None, help="task:seed pairs, instead of every task with every seed")
    ap.add_argument("--run-cap", type=float, default=28.0)
    ap.add_argument("--study-cap", type=float, default=2000.0)
    a = ap.parse_args()
    tasks = PROBLEMS if a.tasks == ["all"] else a.tasks
    pairs = [(j.split(":")[0], int(j.split(":")[1])) for j in a.jobs] if a.jobs else [(t, s) for t in tasks for s in a.seeds]
    assert all(t in PROBLEMS for t, _ in pairs), pairs
    jobs = sorted([(t, s, a) for t, s in pairs], key=lambda j: -PROBLEMS.index(j[0]))   # the longest problems start first
    stamp = lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"launched {stamp()}: " + " ".join(f"{t}:{s}" for t, s, _ in jobs), flush=True)
    (ROOT / "runs" / a.study).mkdir(parents=True, exist_ok=True)
    with open(ROOT / "runs" / a.study / "launches.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(t=stamp(), jobs=[f"{t}:{s}" for t, s, _ in jobs], run_cap=a.run_cap,
                                 study_cap=a.study_cap, parallel=a.parallel)) + "\n")
    with ThreadPoolExecutor(a.parallel) as ex:
        for line in ex.map(one, jobs):
            print(line, flush=True)
    print(f"finished {stamp()}", flush=True)
