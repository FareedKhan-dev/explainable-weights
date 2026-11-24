"""results/score_snapshot.py: build one saved network and score it, in its own process.

    <paper .venv python> score_snapshot.py <mode> <task> <network.py> <out.json>

Modes:
  visible     weights, the visible score (seed 0, 200 examples, the evaluator's defaults), the code scan, and the
              data-use test (four builds, the last two on altered task data);
  hidden      the paper's hidden set: 2,000 fresh examples from its unseen seed, or for TREC 2,000 training questions
              outside the agent's pool (seed 1);
  extra       the generated problems only: 20,000 more fresh examples (seed 555);
  exhaustive  every input, for GCD (998,001 pairs) and dates (620,917 prompts);
  official    TREC's official 500-question test split;
  clean       TREC only: the hidden and official sets without the questions that also occur, word for word, in the
              agent's 2,000-question pool (the dataset repeats some questions), so no score rests on a seen question.
The result goes to <out.json>, so nothing a network prints can corrupt it. results/analyze_runs.py runs this with a
time limit, one network at a time per process.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
os.environ.setdefault("WSP_PAPER", str(WS / "weight-space-programming-autoresearch"))
os.environ.setdefault("WSP_HF_HOME", str(Path(__file__).resolve().parents[1] / ".hfh"))
sys.path.insert(0, str(HERE.parent))
from wsp import audit, config, networks, verify  # noqa: E402


def main(mode, task_name, path, out):
    path, t0, rec = Path(path), time.time(), {"mode": mode}
    try:
        if mode == "visible":
            source = path.read_text(encoding="utf-8", errors="replace")
            template = (config.EVOLVE / "interpretable_transformer.py").read_text(encoding="utf-8")
            rec["scan"] = audit.scan(source, template)
        task, model = networks.build(networks.load(path), task_name)
        rec["weights"], rec["build_seconds"] = networks.weights(model), round(time.time() - t0, 1)
        if mode == "visible":
            rec["accuracy"], _ = verify.hidden(model, task, n=200, seed=0)
            rec["data_use"] = audit.data_use(path, task_name)
        elif mode == "hidden" and task_name in verify.DATASETS:
            rec["accuracy"], _ = verify.score_pairs(model, task, verify.dataset_hidden(task_name))
        elif mode == "benchmark":
            acc, confusion, n = verify.benchmark(model, task, task_name)
            rec.update(accuracy=acc, n=n, split=verify.BENCHMARK[task_name], ci95=verify.interval(acc, n),
                       confusion=[[t, g, c] for (t, g), c in sorted(confusion.items())])
        elif mode == "hidden":
            rec["accuracy"], _ = verify.hidden(model, task)
        elif mode == "extra":
            rec["accuracy"], _ = verify.hidden(model, task, n=verify.EXTRA_N, seed=verify.EXTRA_SEED)
        elif mode == "exhaustive":
            check = {"gcd-three-digits": verify.exhaustive_gcd, "date-normalization": verify.exhaustive_dates}[task_name]
            rec["inputs"], rec["wrong"], rec["first_wrong"] = check(model, task)
        elif mode == "official":
            acc, confusion, n = verify.trec_official(model, task)
            rec.update(accuracy=acc, n=n, confusion=[[t, g, c] for (t, g), c in sorted(confusion.items())])
        elif mode == "clean":
            seen = verify.seen_by_agent(task_name)
            for name, pairs in (("hidden", verify.dataset_hidden(task_name)),
                                ("official", verify.dataset_rows(task_name, verify.BENCHMARK[task_name]))):
                kept = [(p, a) for p, a in pairs if p not in seen]
                acc, _ = verify.score_pairs(model, task, kept)
                rec[name] = dict(accuracy=acc, n=len(kept), dropped=len(pairs) - len(kept),
                                 ci95=verify.interval(acc, len(kept)))
        else:
            raise ValueError(f"unknown mode {mode}")
        rec["ok"] = True
    except BaseException as e:                                  # a network that fails is a result, not a crash
        rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}", where=traceback.format_exc()[-600:])
    rec["seconds"] = round(time.time() - t0, 1)
    Path(out).write_text(json.dumps(rec, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:5])
