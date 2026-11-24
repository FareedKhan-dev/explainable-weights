"""results/train_baseline.py: train the black-box baseline (addition) and keep its learning curve and weights.

Writes results/baseline_curve.json (one row per log step) and results/baseline_addition.pt (the trained weights).
Run with the paper repo's environment, which has torch:
    <paper .venv python> results/train_baseline.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from wsp.baseline import train  # noqa: E402

model, curve = train("addition-five-digits", steps=40000, out=str(HERE / "baseline_addition.pt"))
(HERE / "baseline_curve.json").write_text(json.dumps(curve, indent=1) + "\n", encoding="utf-8", newline="\n")
print("final:", curve[-1])
