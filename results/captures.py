"""results/captures.py: run the post's driver cells for real, in post order, in one shared namespace (like a notebook),
then the cells kept for the companion notebook, and save what each prints to results/outputs/<id>.txt.

    <paper .venv python> captures.py [--check]

Runs from the post's folder. With --check, nothing is written: each fresh capture is compared with the saved file and
any difference is an error, so a capture can never drift from the code that made it.
"""
import contextlib
import io
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WS = ROOT.parent
os.environ.setdefault("WSP_PAPER", str(WS / "weight-space-programming-autoresearch"))
os.environ.setdefault("WSP_HF_HOME", str(Path(__file__).resolve().parents[1] / ".hfh"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from cells import CELLS, REPO_CELLS  # noqa: E402


def main(check=False):
    os.chdir(ROOT)
    space, bad = {}, 0
    for name, code in list(CELLS.items()) + list(REPO_CELLS.items()):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(code, f"<cell {name}>", "exec"), space)
        text = buf.getvalue().rstrip("\n") + "\n"
        dest = HERE / "outputs" / f"{name}.txt"
        if check:
            same = dest.exists() and dest.read_text(encoding="utf-8") == text
            bad += not same
            print(f"{name}: {'same' if same else 'DIFFERENT'}")
        else:
            dest.parent.mkdir(exist_ok=True)
            dest.write_text(text, encoding="utf-8", newline="\n")
            print(f"== {name}\n{text}", end="")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv[1:]))
