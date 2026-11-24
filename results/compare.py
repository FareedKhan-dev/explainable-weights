"""results/compare.py: Qwen3.7-Max's picks beside the paper's own networks, as the article prints them.

    python results/compare.py

Two tables, read from the result files and checked here:
  compare_generated.txt  the six generated problems both agents ran: each agent's network size and hidden accuracy
                         (Qwen's picks from qwen_runs.json, the paper's networks as checked by paper_checks.json)
  compare_text.txt       the four text datasets on their hidden sets: always guessing the majority class (floors.json),
                         Qwen's pick, the black box I trained on the same data (traces.json) and the paper's network
The paper's networks are the paper's own runs. This study never ran the paper's agent, it only checked its networks."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
J = lambda name: json.loads((HERE / name).read_text(encoding="utf-8"))
GENERATED = [("word-reversal-3x3", "reversal"), ("addition-five-digits", "addition"), ("gcd-three-digits", "gcd"),
             ("date-normalization", "dates"), ("smallest-prime-factor", "prime factor"),
             ("linear-interpolation-two-points", "interpolation")]
TEXT = [("trec-coarse", "trec"), ("sentiment-sst2", "sst-2"), ("paraphrase-mrpc", "mrpc"), ("nli-snli", "snli")]


def main():
    picks = {p["task"]: p["pick"] for p in J("qwen_runs.json")["picks"]}
    paper = J("paper_checks.json")
    paper = paper["checks"] if "checks" in paper else paper
    boxes, floors = J("traces.json")["blackboxes"], J("floors.json")["datasets"]
    rows = ["problem           qwen weights   hidden    paper weights   hidden"]
    solved = {"qwen": 0, "paper": 0}
    for task, name in GENERATED:
        q, c = picks[task], paper[task]
        cw, ch = c["visible"]["weights"], c["hidden"]["accuracy"]
        solved["qwen"] += q["hidden"] == 1.0
        solved["paper"] += ch == 1.0
        rows.append(f"{name:15s}{q['weights']:15,d}{q['hidden']:9.4f}{cw:17,d}{ch:9.4f}")
    assert solved == {"qwen": 1, "paper": 6}, solved
    rows.append(f"solved{'':16s}{solved['qwen']} of {len(GENERATED)}{'':19s}{solved['paper']} of {len(GENERATED)}")
    (HERE / "outputs" / "compare_generated.txt").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(rows))
    text = ["dataset   guessing     qwen   our black box    paper"]
    for task, name in TEXT:
        fl, q, b, c = floors[task]["hidden"]["share"], picks[task]["hidden"], boxes[task]["hidden"], paper[task]["hidden"]["accuracy"]
        assert c > b > q, (task, c, b, q)                 # the paper's network above the black box, Qwen's below both
        text.append(f"{name:8s}{fl:10.4f}{q:9.4f}{b:16.4f}{c:9.4f}")
    (HERE / "outputs" / "compare_text.txt").write_text("\n".join(text) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
