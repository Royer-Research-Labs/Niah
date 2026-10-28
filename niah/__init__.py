"""Niah: a small, self-contained Needle-In-A-Haystack long-context eval.

The eval uses likelihood-based multiple-choice scoring with a control-prior
disqualification step (no text generation required), which makes it robust even
for tiny from-scratch models. See ``niah.eval.run_niah`` for the entry point.
"""

from .attention import needle_attention_profile
from .eval import NiahConfig, run_niah
from .hard import build_niah_hard_spec, iter_niah_hard_rows
from .spec import build_niah_spec, load_niah_spec, locate_needle, materialize_niah_jsonl, write_niah_spec
from .tokenizer import TokenizerAdapter, load_tokenizer

__all__ = [
    "NiahConfig",
    "run_niah",
    "needle_attention_profile",
    "locate_needle",
    "build_niah_spec",
    "build_niah_hard_spec",
    "iter_niah_hard_rows",
    "load_niah_spec",
    "materialize_niah_jsonl",
    "write_niah_spec",
    "TokenizerAdapter",
    "load_tokenizer",
]
