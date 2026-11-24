"""results/traces.py: the worked examples the figures and the post follow, each checked against the real network.

    <paper .venv python> traces.py

  gcd        the paper's GCD network on 084,036: the program's A and B after every fold of the first two rounds, and
             the network's own answer, which must match the program's
  telephone  two real questions from TREC's official test split, one starting with "who" and one with "when": the
             rule the paper's TREC network fires (its own rule table, the highest priority and then the earliest
             match), its real answer, and the trained black box's answer with integrated gradients summed per word
  blackboxes the black box picked for each dataset (the variant with the better validation score): its training and
             validation rows, weights, hidden and official accuracy, and its comprehensiveness
Writes results/traces.json.
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
os.environ.setdefault("WSP_PAPER", str(WS / "weight-space-programming-autoresearch"))
os.environ.setdefault("WSP_HF_HOME", str(Path(__file__).resolve().parents[1] / ".hfh"))
sys.path.insert(0, str(HERE.parent))
import torch  # noqa: E402

from wsp import config, networks, saliency, verify  # noqa: E402
from wsp.model import SimpleTransformer  # noqa: E402
from wsp.programs import FOLDS, euclid  # noqa: E402

ORIG = config.PAPER / "experiments" / "nl-bottleneck" / "originals"
CLASSES = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]


def gcd_trace(a=84, b=36, rounds=2):
    A, B, steps = a, b, []
    for r in range(rounds):
        for i, (s, k) in enumerate(FOLDS):
            fold = A + sum((-1) ** m * 2 * max(0.0, A - m * s * B / 2) for m in range(1, k + 1))
            A, B = (B, fold) if i == len(FOLDS) - 1 else (fold, B)
            steps.append(dict(round=r + 1, scale=s, A=A, B=B, swapped=i == len(FOLDS) - 1))
    task, model = networks.build(networks.load(ORIG / "gcd-three-digits.py"), "gcd-three-digits")
    prompt = f"{a:03d},{b:03d}="
    got = verify.answers(model, task, [prompt])[0]
    assert got == euclid(a, b) == f"{12:03d}", (got, euclid(a, b))
    return dict(prompt=prompt, a=a, b=b, steps=steps, answer=got, program=euclid(a, b))


def rule_pattern(keyword):
    anchored, stem = keyword.startswith("^"), keyword.endswith("*")
    core = re.escape(keyword.strip("^*")).replace(r"\.", ".")
    return re.compile(("^" if anchored else r"\b") + core + (r"\w*" if stem else r"\b"))


def telephone():
    spec = importlib.util.spec_from_file_location("trec_net", ORIG / "trec-coarse.py")
    module = importlib.util.module_from_spec(spec)
    networks.paper_code()
    spec.loader.exec_module(module)
    task, net = networks.build(module, "trec-coarse")
    rows = verify.dataset_rows("trec-coarse", "test")
    tasks, _ = networks.paper_code()
    bb_task = tasks.get_task("trec-coarse")
    bb = SimpleTransformer(bb_task.vocab_size, max_seq_len=bb_task.prompt_len + bb_task.answer_len)
    bb.load_state_dict(torch.load(HERE / "blackbox_trec-coarse_long.pt", map_location="cpu"))
    bb.eval()
    out = []
    for start in ("who invented the telephone", "when was the telephone invented"):
        prompt, label = next((p, lab) for p, lab in rows if p.startswith(start))
        text = prompt.rstrip("=").rstrip("_")
        fired = []
        for priority, cls, keywords in module.RULES:
            for kw in keywords.split("|"):
                m = rule_pattern(kw).search(text)
                if m:
                    fired.append(dict(priority=priority, cls=CLASSES[cls], keyword=kw, at=m.start(), salience=300 * priority - 3 * m.start()))
        fired.sort(key=lambda f: -f["salience"])
        got = CLASSES[int(verify.answers(net, task, [prompt])[0])]
        assert fired and fired[0]["cls"] == got, (start, fired[:2], got)
        ids = torch.tensor([bb_task.encode(prompt)], dtype=torch.long)
        with torch.no_grad():
            pred = bb(ids)[0, -1].argmax().item()
        attr, gap = saliency.integrated_gradients(bb, ids, pred)
        words, i = [], 0
        for w in text.split(" "):
            if w:
                words.append(dict(word=w, ig=round(attr[i:i + len(w)].sum().item(), 4)))
            i += len(w) + 1
        out.append(dict(question=text, truth=CLASSES[int(label)], network=got, rule=fired[0], also=fired[1:4],
                        blackbox=CLASSES[int(bb_task.itos[pred])], words=words, gap=round(gap, 4)))
        print(f"{text!r}: truth {CLASSES[int(label)]}, network {got} by {fired[0]['keyword']!r}, black box {out[-1]['blackbox']}")
    return out


def blackboxes():
    out = {}
    for name in verify.DATASETS:
        variants = {}
        for tag in ("", "_long"):
            f = HERE / f"blackbox_{name}{tag}.json"
            if f.exists():
                d = json.loads(f.read_text(encoding="utf-8"))
                variants[tag or "_short"] = d
        tag, d = max(variants.items(), key=lambda kv: max(c["validation"] for c in kv[1]["curve"]))
        out[name] = dict(variant=tag.strip("_"), weights=d["weights"], steps=d["steps"], train=d["train_examples"],
                         validation=d["validation_examples"], best_validation=round(max(c["validation"] for c in d["curve"]), 4),
                         hidden=d["hidden"]["accuracy"], official=d["benchmark"]["accuracy"],
                         comprehensiveness=d["explanation"]["comprehensiveness"])
        print(name, out[name]["variant"], out[name]["train"], out[name]["hidden"], out[name]["official"])
    return out


def main():
    res = dict(provisional=False, gcd=gcd_trace(), telephone=telephone(), blackboxes=blackboxes())
    (HERE / "traces.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("gcd", [(s["scale"], s["A"], s["B"]) for s in res["gcd"]["steps"]], res["gcd"]["answer"])


if __name__ == "__main__":
    main()
