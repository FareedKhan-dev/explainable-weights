"""results/repeat_picks.py: score each dataset pick five times, each in a fresh process, and report the spread.

    <paper .venv python> repeat_picks.py [repeats]

Why: some networks give two answers almost equal scores, and floating-point rounding, which can differ from one run to
the next on this CPU, then decides between them. The same file run twice gave 0.68 and then 0.32 (paraphrase seed 2,
attempt 15). A single score of such a network is one draw. So every dataset pick, Qwen's and the paper's
networks alike, is scored `repeats` times on the hidden set and on the dataset's own labelled split. A pick counts as
beating the majority class only if its lowest score does. Writes results/repeat_picks.json.
"""
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
PY = Path(os.environ.get("WSP_PAPER", WS / "weight-space-programming-autoresearch")) / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
RUNS = Path(__file__).resolve().parents[1] / "runs"
ORIGINALS = WS / "weight-space-programming-autoresearch" / "experiments" / "nl-bottleneck" / "originals"
DATASETS = ["sentiment-sst2", "paraphrase-mrpc", "nli-snli", "trec-coarse"]
SCRATCH = HERE / "repeat"


def scores(task, net, tag, repeats):
    out = {}
    for mode in ("hidden", "benchmark"):
        vals = []
        for k in range(repeats):
            dest = SCRATCH / f"{tag}_{mode}_{k}.json"
            if not dest.exists():
                subprocess.run([str(PY), "-X", "utf8", str(HERE / "score_snapshot.py"), mode, task, str(net), str(dest)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
            r = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
            if r.get("ok"):
                vals.append(r["accuracy"])
        out[mode] = dict(values=vals, mean=statistics.mean(vals) if vals else None, low=min(vals, default=None),
                         high=max(vals, default=None), spread=(max(vals) - min(vals)) if vals else None)
    return out


def main(repeats=5):
    SCRATCH.mkdir(exist_ok=True)
    picks = {p["task"]: p["pick"] for p in json.loads((HERE / "qwen_runs.json").read_text(encoding="utf-8"))["picks"]}
    card = json.loads((HERE / "data_card.json").read_text(encoding="utf-8"))["problems"]
    floors = {"trec-coarse": card["trec-coarse"]["hidden"]["majority_accuracy"]}
    out = {}
    for task in DATASETS:
        p = picks[task]
        net = next((RUNS / p["run"] / "attempts").glob(f"{p['idx']:03d}_*.py"))
        out[task] = dict(qwen=dict(run=p["run"], idx=p["idx"], weights=p["weights"], **scores(task, net, f"qwen_{task}", repeats)),
                         paper=dict(weights=None, **scores(task, ORIGINALS / f"{task}.py", f"paper_{task}", repeats)))
        for who in ("qwen", "paper"):
            h, b = out[task][who]["hidden"], out[task][who]["benchmark"]
            print(f"{task:16s} {who:6s} hidden mean {h['mean']:.4f} [{h['low']:.4f}, {h['high']:.4f}]  "
                  f"benchmark mean {b['mean']:.4f} [{b['low']:.4f}, {b['high']:.4f}]", flush=True)
    (HERE / "repeat_picks.json").write_text(json.dumps(dict(provisional=False, repeats=repeats, picks=out), indent=1)
                                           + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(*[int(x) for x in sys.argv[1:2]])
