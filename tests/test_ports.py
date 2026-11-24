"""tests/test_ports.py: our tasks, model and evaluator agree with the paper's own code, number for number.

Run with the paper repo's environment (it has torch):
    WSP_PAPER=<the paper's clone> python tests/test_ports.py        (or: pytest tests)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import config, networks  # noqa: E402
from wsp.evaluate import accuracy  # noqa: E402
from wsp.model import SimpleTransformer  # noqa: E402
from wsp.tasks import Problem  # noqa: E402

ORIGINALS = config.PAPER / "experiments" / "nl-bottleneck" / "originals"   # the paper's published networks
NETS = {name: f"{name}.py" for name in ("word-reversal-3x3", "addition-five-digits", "gcd-three-digits",
                                         "date-normalization", "trec-coarse")}


def main():
    tasks, evaluator = networks.paper_code()
    sys.path.insert(0, str(config.EVOLVE))
    import interpretable_transformer as template          # the paper's starting file, for its SimpleTransformer
    checks = []
    for name, rel in NETS.items():
        ours, theirs = Problem(name), tasks.get_task(name)
        for seed in (0, 987654321):
            a, b = ours.examples(200, seed), theirs.generate_examples(200, seed=seed)
            checks.append((f"{name} seed {seed}: same 200 examples",
                           [(e.prompt, e.answer) for e in a] == [(e.prompt, e.answer) for e in b]))
        checks.append((f"{name}: same characters", ours.vocab == theirs.vocab))
        size = max(theirs.seq_len, 16)
        mine = sum(p.numel() for p in SimpleTransformer(len(ours.vocab), max_seq_len=size).parameters())
        paper = sum(p.numel() for p in template.SimpleTransformer(theirs.vocab_size, max_seq_len=size).parameters())
        checks.append((f"{name}: template has {paper:,} weights in both", mine == paper))
        task, model = networks.build(networks.load(ORIGINALS / rel), name)
        ex = ours.examples(2000, 987654321)
        our_acc, _ = accuracy(model, ours, ex)
        their_acc, _ = evaluator.evaluate(model, task, n_samples=2000, seed=987654321, device="cpu")
        checks.append((f"{name}: the paper's network scores {our_acc:.4f} with ours and {their_acc:.4f} with theirs",
                       our_acc == their_acc))
    for label, ok in checks:
        print(("PASS " if ok else "FAIL ") + label)
    print(f"{sum(ok for _, ok in checks)} of {len(checks)} checks pass")
    return 0 if all(ok for _, ok in checks) else 1


def test_ports():
    """The pytest entry point: the checks above, skipped when the paper's code is not there."""
    import pytest
    paper = Path(os.environ.get("WSP_PAPER") or Path(__file__).resolve().parents[2] / "weight-space-programming-autoresearch")
    if not (paper / "experiments").is_dir():
        pytest.skip("set WSP_PAPER to a clone of the paper's code")
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
