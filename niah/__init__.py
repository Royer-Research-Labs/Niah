"""Niah: a small, self-contained Needle-In-A-Haystack long-context eval.

The eval uses likelihood-based multiple-choice scoring with a control-prior
disqualification step (no text generation required), which makes it robust even
for tiny from-scratch models. See ``niah.eval.run_niah`` for the entry point.
"""

from .eval import NiahConfig, run_niah
from .spec import build_niah_spec, load_niah_spec, materialize_niah_jsonl, write_niah_spec
from .tokenizer import TokenizerAdapter, load_tokenizer

__all__ = [
    "NiahConfig",
    "run_niah",
    "build_niah_spec",
    "load_niah_spec",
    "materialize_niah_jsonl",
    "write_niah_spec",
    "TokenizerAdapter",
    "load_tokenizer",
]
