"""wsp/networks.py: load a network the agent wrote and build it for its task, the way the paper's own scripts do."""
import contextlib
import hashlib
import importlib.util
import io
import os
import sys

from wsp import config


def paper_code():
    """Put the paper's task and evaluator code (src/) on the path, with the dataset cache offline."""
    os.environ.setdefault("HF_HOME", str(config.HF_HOME))
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    if str(config.EVOLVE) not in sys.path:
        sys.path.insert(0, str(config.EVOLVE))
    import src.eval
    import src.task
    return src.task, src.eval


def load(path):
    """Import one saved network file as its own module, quietly, without running its main block."""
    paper_code()
    name = "net_" + hashlib.sha1(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    if hasattr(module, "SELFTEST_N"):           # some networks check every input while they build, skip that here
        module.SELFTEST_N = 0
    return module


def build(module, task_name):
    """The task object and the built network, in evaluation mode, on the CPU."""
    tasks, _ = paper_code()
    task = tasks.get_task(task_name)
    with contextlib.redirect_stdout(io.StringIO()):
        model = module.build_model(task)
    return task, model.eval().cpu()


def weights(model):
    """The weight count exactly as the paper's evaluator logs it."""
    return sum(p.numel() for p in model.parameters())
