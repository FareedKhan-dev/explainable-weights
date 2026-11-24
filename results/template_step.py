"""results/template_step.py: the paper's template, with its random starting weights, reading the post's running prompt.

It writes the first answer step's real scores for the 28 characters to results/template_step.json, for the architecture
figure, built exactly as the post's computation cell builds the template (torch.manual_seed(0) right before the build).
Its pick must be the first letter of the template's answer in the captured output, or nothing is written.

    <paper .venv python> results/template_step.py
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WS = ROOT.parent
os.environ.setdefault("WSP_PAPER", str(WS / "weight-space-programming-autoresearch"))
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from wsp import config, networks, verify  # noqa: E402

PROMPT = "fnm_mod_wfd="


def main():
    os.chdir(ROOT)
    torch.manual_seed(0)                                         # as in the post's cell
    task, template = networks.build(networks.load(config.EVOLVE / "interpretable_transformer.py"), "word-reversal-3x3")
    with torch.no_grad():
        logits = template(torch.tensor([task.encode(PROMPT)]))[0, -1]
    probs = logits.softmax(-1)
    vocab = "".join(task.itos[i] for i in range(len(logits)))
    pick = vocab[int(logits.argmax())]
    answer = verify.answers(template, task, [PROMPT])[0]
    assert pick == answer[0], (pick, answer)
    out = dict(provisional=False, prompt=PROMPT, vocab=vocab, pick=pick, answer=answer,
               logits=[round(float(v), 4) for v in logits], probs=[round(float(v), 4) for v in probs])
    (HERE / "template_step.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"template on {PROMPT}: first letter {pick} (answer {answer}), top probability {float(probs.max()):.4f}")


if __name__ == "__main__":
    main()
