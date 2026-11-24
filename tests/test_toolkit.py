"""tests/test_toolkit.py: the explainability toolkit gives the paper's own numbers, and its tools are sound.

Run with the paper repo's environment (it has torch):
    WSP_PAPER=<the paper's clone> python tests/test_toolkit.py      (or: pytest tests)

  - the paper's five picked networks (experiments/nl-bottleneck/originals) score what its results.json says, on the
    visible set and on its hidden sets, TREC's outside-pool set included;
  - switching off weights restores them exactly, and the 3-weight reversal network fails without them;
  - integrated gradients add up to the score change on a random template network;
  - a scored run counts as a visible score only with the evaluator's defaults.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import config, networks, saliency, switchoff, verify  # noqa: E402
from wsp.model import SimpleTransformer  # noqa: E402
from wsp.rebuild import default_run  # noqa: E402
from wsp.tasks import Problem  # noqa: E402

NB = config.PAPER / "experiments" / "nl-bottleneck"
PROBLEMS = ["word-reversal-3x3", "addition-five-digits", "gcd-three-digits", "date-normalization", "trec-coarse"]


def main():
    checks = []
    paper = json.loads((NB / "results.json").read_text(encoding="utf-8"))
    for name in PROBLEMS:
        task, model = networks.build(networks.load(NB / "originals" / f"{name}.py"), name)
        vis, _ = verify.hidden(model, task, n=200, seed=0)
        if name == "trec-coarse":
            hid, _ = verify.score_pairs(model, task, verify.trec_hidden())
        else:
            hid, _ = verify.hidden(model, task)
        want = paper[name]["orig"]
        checks.append((f"{name}: {networks.weights(model):,} weights, visible {vis} and hidden {hid}, as the paper logs",
                       networks.weights(model) == paper[name]["orig_params"] and vis == want["visible"]
                       and hid == want["hidden"]))

    task, model = networks.build(networks.load(NB / "originals" / "word-reversal-3x3.py"), "word-reversal-3x3")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rows = switchoff.drops(model, task, [dict(name="the pointer", params=["reads"])], n=500)
    checks.append(("reversal: the 3 weights are the pointer 9 + i - 8·word", before["reads"].tolist() == [9.0, 1.0, -8.0]))
    checks.append(("reversal: switching them off drops 100 points", rows[0]["intact"] == 1.0 and rows[0]["off"] == 0.0))
    checks.append(("switched-off weights come back exactly", all(torch.equal(before[n], p) for n, p in model.named_parameters())))

    torch.manual_seed(0)
    problem = Problem("addition-five-digits")
    net = SimpleTransformer(len(problem.vocab), max_seq_len=18).eval()
    e = problem.examples(1, 3)[0]
    ids = torch.tensor([problem.encode(e.prompt)], dtype=torch.long)
    attr, gap = saliency.integrated_gradients(net, ids, problem.vocab.index(e.answer[0]), steps=256)
    checks.append((f"integrated gradients add up to the score change ({attr.sum().item():.4f} against {gap:.4f})",
                   abs(attr.sum().item() - gap) <= 0.01 * max(abs(gap), 1e-3)))
    g = saliency.gradient(net, ids, problem.vocab.index(e.answer[0]))
    checks.append(("gradient saliency gives one number per input character", g.shape == (ids.shape[1],)))

    checks.append(("default visible runs are told apart", default_run(["--task", "x"]) and default_run(["--task=x", "--seed", "0"])
                   and not default_run(["--task", "x", "--n-samples", "1"]) and not default_run([])))
    for label, ok in checks:
        print(("PASS " if ok else "FAIL ") + label)
    print(f"{sum(ok for _, ok in checks)} of {len(checks)} checks pass")
    return 0 if all(ok for _, ok in checks) else 1


def test_toolkit():
    """The pytest entry point: the checks above, skipped when the paper's code is not there."""
    import pytest
    paper = Path(os.environ.get("WSP_PAPER") or Path(__file__).resolve().parents[2] / "weight-space-programming-autoresearch")
    if not (paper / "experiments").is_dir():
        pytest.skip("set WSP_PAPER to a clone of the paper's code")
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
