"""results/audit_repro.py: the data-use audit, rerun after its post-study split into helpers, must give the same verdicts.

    <paper .venv python> audit_repro.py

The study's verdicts are in qwen_runs.json (every distinct file of the 19 full runs). This reruns the audit, through
the same one-network worker (score_snapshot.py visible), on every label-dependent and nondeterministic file and on 50
independent files drawn with a fixed seed, and writes results/audit_repro.json with each old and new verdict.
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
PY = Path(os.environ.get("WSP_PAPER", WS / "weight-space-programming-autoresearch")) / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
RUNS = Path(__file__).resolve().parents[1] / "runs"
STUBS = {"decimal-to-binary-8bit-qwen37-max-s1", "edit-distance-8-qwen37-max-s1", "edit-distance-8-qwen37-max-s2"}


def main():
    runs = [r for r in json.loads((HERE / "qwen_runs.json").read_text(encoding="utf-8"))["runs"] if r["run"] not in STUBS]
    files = [(r["run"], r["task"], f) for r in runs for f in r["files"] if f.get("data_use")]
    odd = [x for x in files if x[2]["data_use"] in ("label_dependent", "nondeterministic")]
    calm = [x for x in files if x[2]["data_use"] == "independent"]
    pick = odd + random.Random(20261122).sample(calm, 50)
    out, same = [], 0
    for k, (run, task, f) in enumerate(pick, 1):
        net = next((RUNS / run / "attempts").glob(f"{f['idx']:03d}_*.py"))
        dest = HERE / "repro" / f"{run}_{f['idx']:03d}.json"
        dest.parent.mkdir(exist_ok=True)
        subprocess.run([str(PY), "-X", "utf8", str(HERE / "score_snapshot.py"), "visible", task, str(net), str(dest)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
        new = json.loads(dest.read_text(encoding="utf-8")).get("data_use") if dest.exists() else None
        same += new == f["data_use"]
        out.append(dict(run=run, idx=f["idx"], name=f["name"], study=f["data_use"], now=new))
        print(f"{k:3d}/{len(pick)} {run:46s} {f['idx']:3d} study {f['data_use']:17s} now {new}", flush=True)
    res = dict(provisional=False, checked=len(pick), identical=same, label_dependent=sum(x[2]["data_use"] == "label_dependent" for x in odd),
               nondeterministic=sum(x[2]["data_use"] == "nondeterministic" for x in odd), independent_sampled=50, rows=out)
    (HERE / "audit_repro.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"identical verdicts: {same} of {len(pick)}")
    return 0 if same == len(pick) else 1


if __name__ == "__main__":
    sys.exit(main())
