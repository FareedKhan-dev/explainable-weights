"""results/paper_checks.py: the paper's own networks, checked exactly as Qwen's networks are (results/score_snapshot.py).

    <paper .venv python> paper_checks.py

The paper's own picks (experiments/nl-bottleneck/originals, chosen by its 01_select.py): the visible set and the audit,
the hidden and extra sets, every input for GCD and dates, and the official test for TREC.
Each result is cached as results/paper/<task>_<mode>.json. Writes results/paper_checks.json.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
os.environ.setdefault("WSP_PAPER", str(WS / "weight-space-programming-autoresearch"))
sys.path.insert(0, str(HERE.parent))
from wsp import verify  # noqa: E402
PAPER = Path(os.environ.get("WSP_PAPER", WS / "weight-space-programming-autoresearch"))
PY = PAPER / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
ORIGINALS = PAPER / "experiments" / "nl-bottleneck" / "originals"
# all 17 of the paper's problems: the generated ones on the hidden and extra sets (and every input for GCD and dates),
# the dataset ones on the hidden set, their own labelled split and the clean sets
GENERATED = ["addition-five-digits", "multiplication-five-digits", "sort-five-digits", "digit-counting-10",
             "parity-upto10-bits", "boolean-circuit-5-bits", "linear-interpolation-two-points", "word-reversal-3x3",
             "gcd-three-digits", "decimal-to-binary-8bit", "edit-distance-8", "smallest-prime-factor", "date-normalization"]
DATASETS = ["sentiment-sst2", "paraphrase-mrpc", "nli-snli", "trec-coarse"]
MODES = {t: ["visible", "hidden", "extra"] + (["exhaustive"] if t in ("gcd-three-digits", "date-normalization") else [])
         for t in GENERATED}
MODES.update({t: ["visible", "hidden", "benchmark", "clean"] + (["official"] if t == "trec-coarse" else []) for t in DATASETS})


def main():
    out, env = {}, dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    (HERE / "paper").mkdir(exist_ok=True)
    for task, modes in MODES.items():
        out[task] = {}
        for mode in modes:
            cache = HERE / "paper" / f"{task}_{mode}.json"
            if not cache.exists():
                subprocess.run([str(PY), "-X", "utf8", str(HERE / "score_snapshot.py"), mode, task,
                                str(ORIGINALS / f"{task}.py"), str(cache)], env=env, timeout=7200,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            out[task][mode] = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {"ok": False}
            r = out[task][mode]
            n = r.get("n") or {"visible": 200, "hidden": 2000, "extra": 20000, "official": 500}.get(mode)
            if task == "paraphrase-mrpc" and mode == "hidden":
                n = 1668                                    # MRPC has only 1,668 rows outside its pool
            if n and r.get("accuracy") is not None:
                r["ci95"] = verify.interval(r["accuracy"], n)
            print(task, mode, {k: r.get(k) for k in ("ok", "weights", "accuracy", "ci95", "inputs", "wrong", "data_use",
                                                     "error") if k in r}, flush=True)
    (HERE / "paper_checks.json").write_text(json.dumps(dict(provisional=False, source=str(ORIGINALS), checks=out),
                                                      indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
