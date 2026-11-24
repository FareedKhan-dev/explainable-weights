"""results/staircase.py: the attempt log of Qwen3.7-Max's word reversal run (run 1), as the post prints it.

    python results/staircase.py

Names and visible scores come from the run's own attempts.jsonl, written by the agent loop as each attempt ran. A few
quick runs scored only 3 or 5 prompts, and the log says so. The weight counts are the exact ones from rebuilding each
saved file (results/qwen_network.json, staircase), because attempts.jsonl keeps the evaluator's rounded count. Writes results/outputs/staircase_a.txt (attempts 1 to 13) and
staircase_b.txt (attempts 14 to 27), the two blocks the post shows."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "runs" / "word-reversal-3x3-qwen37-max-s1" / "attempts" / "attempts.jsonl"


def main():
    rows = [json.loads(line) for line in RUN.read_text(encoding="utf-8").splitlines() if line.strip()]
    exact = {a["idx"]: a for a in json.loads((HERE / "qwen_network.json").read_text(encoding="utf-8"))["staircase"]}
    assert [r["idx"] for r in rows] == list(range(1, 28)) and set(exact) == set(range(1, 28))
    lines = []
    for r in rows:
        e = exact[r["idx"]]
        if r["accuracy"] is None:                               # the file raised an error, so the evaluator gave no score
            assert r["exit"] != 0 and e["weights"] is None
            lines.append(f"{r['idx']:3d}  {r['name']:25s}{'':>9}  crashed")
            continue
        assert e["visible"] == r["accuracy"]                   # some quick runs used only a few prompts, as logged
        lines.append(f"{r['idx']:3d}  {r['name']:25s}{e['weights']:9,d}  {r['correct']:3d} of {r['n']}")
    head = "  #  attempt name                weights  visible right"
    (HERE / "outputs" / "staircase_a.txt").write_text("\n".join([head] + lines[:13]) + "\n", encoding="utf-8", newline="\n")
    (HERE / "outputs" / "staircase_b.txt").write_text("\n".join([head] + lines[13:]) + "\n", encoding="utf-8", newline="\n")
    print("\n".join([head] + lines))


if __name__ == "__main__":
    main()
