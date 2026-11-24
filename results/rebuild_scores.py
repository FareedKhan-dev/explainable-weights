"""results/rebuild_scores.py: the rebuild test's results, every network checked as the originals were.

    <paper .venv python> rebuild_scores.py [task]

For the original pick and each rebuild's final network (r1 and r2 from the 200-word description, the control from the
task alone): the exact weights, the visible, hidden and extra sets, and the data-use audit, through
score_snapshot.py. Then the sealed test (H4, amendment 5): both rebuilds at 100% hidden, each at least ten times
smaller than the control. Writes results/rebuild_<task>.json.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
PY = Path(os.environ.get("WSP_PAPER", WS / "weight-space-programming-autoresearch")) / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REBUILD = Path(__file__).resolve().parents[1] / "runs" / "rebuild"
RUNS = Path(__file__).resolve().parents[1] / "runs"


def check(task, net, tag):
    out = {}
    for mode in ("visible", "hidden", "extra"):
        dest = HERE / "rebuild" / f"{task}_{tag}_{mode}.json"
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            subprocess.run([str(PY), "-X", "utf8", str(HERE / "score_snapshot.py"), mode, task, str(net), str(dest)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
        r = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {"ok": False}
        out[mode] = r.get("accuracy")
        out["weights"] = r.get("weights", out.get("weights"))
        if mode == "visible":
            out["data_use"] = r.get("data_use")
    return out


def main(task="word-reversal-3x3"):
    pick = {p["task"]: p["pick"] for p in json.loads((HERE / "qwen_runs.json").read_text(encoding="utf-8"))["picks"]}[task]
    original = next((RUNS / pick["run"] / "attempts").glob(f"{pick['idx']:03d}_*.py"))
    res = dict(provisional=False, task=task, original=dict(file=original.name, **check(task, original, "original")))
    res["description_words"] = json.loads((REBUILD / task / "describe" / "meta.json").read_text(encoding="utf-8"))["words"]
    for d in ("r1", "r2", "control"):
        meta = json.loads((REBUILD / task / d / "meta.json").read_text(encoding="utf-8"))
        fin = meta.get("final")
        if not fin:
            res[d] = dict(final=None, calls=meta["calls"])
            continue
        net = next((REBUILD / task / d / "attempts").glob(f"{fin['idx']:03d}_*.py"))
        res[d] = dict(file=net.name, calls=meta["calls"], scored_runs=fin["idx"], **check(task, net, d))
    ok = [res[d] for d in ("r1", "r2") if res[d].get("hidden") is not None]
    ctrl = res["control"].get("weights")
    res["h4_reversal"] = dict(both_perfect=len(ok) == 2 and all(r["hidden"] == 1.0 for r in ok),
                              control_over_rebuild=[round(ctrl / r["weights"], 1) for r in ok] if ctrl else None,
                              at_least_10x_smaller=bool(ctrl) and all(ctrl / r["weights"] >= 10 for r in ok))
    (HERE / f"rebuild_{task}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    for k in ("original", "r1", "r2", "control"):
        r = res[k]
        print(f"{k:9s} {r.get('file', '-'):26s} weights {r.get('weights')}  visible {r.get('visible')}  hidden {r.get('hidden')}  "
              f"extra {r.get('extra')}  audit {r.get('data_use')}")
    print("H4 on reversal:", res["h4_reversal"])


if __name__ == "__main__":
    main(*sys.argv[1:2])
