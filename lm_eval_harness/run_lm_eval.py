"""Thin launcher: register the `nanogpt` model, then run lm-eval's CLI.

`lm_eval --include_path` discovers task YAMLs but does NOT import custom model
Python files, so the `nanogpt` model adapter must be imported before the CLI runs.
This launcher does that and forwards all command-line arguments to lm-eval unchanged.

Usage (identical flags to `lm_eval`, just a different entrypoint):

    python lm_eval_harness/run_lm_eval.py \
        --model nanogpt \
        --model_args ckpt=out-niah-demo/ckpt.pt,device=cpu,tokenizer=gpt2 \
        --tasks niah_nanogpt \
        --include_path lm_eval_harness \
        --limit 20
"""

import sys
from pathlib import Path

# lm-eval's results table uses unicode (e.g. the ↑ arrow); on Windows the default
# cp1252 console encoding can't print it. Force UTF-8 so the table renders.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# make both the repo root and this folder importable
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import nanogpt_lm  # noqa: F401  (registers @register_model("nanogpt"))

import lm_eval.tasks as _lm_tasks  # noqa: E402


def _install_relative_to_shim() -> None:
    """Work around an lm-eval bug for tasks loaded via --include_path.

    Some lm-eval builds call ``yaml_path.relative_to(<lm_eval>/tasks)`` while
    pretty-printing the selected tasks. For a task that lives outside the lm-eval
    package (i.e. anything loaded with --include_path) that raises ValueError and
    aborts the run. We swap in a Path subclass whose ``relative_to`` falls back to
    returning the path unchanged instead of crashing. Affects only this process.
    """
    base = type(_lm_tasks.Path())  # concrete WindowsPath / PosixPath

    class _SafePath(base):
        def relative_to(self, *args, **kwargs):
            try:
                return super().relative_to(*args, **kwargs)
            except ValueError:
                return self

    _lm_tasks.Path = _SafePath


_install_relative_to_shim()

from lm_eval.__main__ import cli_evaluate  # noqa: E402

if __name__ == "__main__":
    cli_evaluate()
