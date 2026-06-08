"""Minimal tokenizer adapter for NIAH.

Supports two backends:
  * ``gpt2`` via ``tiktoken`` (the default; matches the nanoGPT demo model), and
  * any Hugging Face repo id via ``transformers.AutoTokenizer`` (optional).

NIAH keyword "choices" must encode to a single token, so the eval needs the
*same* tokenizer the model was trained with. The nanoGPT demo trains on the
GPT-2 BPE tokenizer, so ``gpt2`` is the right default here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import tiktoken

_TOKENIZER_CACHE: dict[str, "TokenizerAdapter"] = {}


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TokenizerAdapter:
    name: str
    resolved_name: str
    source: str  # "tiktoken" or "transformers"
    vocab_size: int
    eot_token_id: Optional[int]
    _tokenizer: object

    def encode(self, text: str) -> List[int]:
        if self.source == "tiktoken":
            return self._tokenizer.encode_ordinary(text)
        return self._tokenizer.encode(text, add_special_tokens=False)

    def encode_ordinary(self, text: str) -> List[int]:
        return self.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self._tokenizer.decode(ids)


def resolve_tokenizer_name(name: str | None) -> str:
    if not name:
        return "gpt2"
    cleaned = str(name).strip()
    return cleaned or "gpt2"


def load_tokenizer(name: str | None) -> TokenizerAdapter:
    """Load (and cache) a tokenizer adapter by name. Defaults to gpt2."""
    resolved = resolve_tokenizer_name(name)
    cached = _TOKENIZER_CACHE.get(resolved)
    if cached is not None:
        return cached

    if resolved.lower() == "gpt2":
        enc = tiktoken.get_encoding("gpt2")
        adapter = TokenizerAdapter(
            name=resolved,
            resolved_name="gpt2",
            source="tiktoken",
            vocab_size=enc.n_vocab,
            eot_token_id=enc.eot_token,
            _tokenizer=enc,
        )
        _TOKENIZER_CACHE[resolved] = adapter
        return adapter

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            f"transformers is required to load tokenizer '{resolved}'. "
            "Install it with `pip install transformers`, or use 'gpt2'."
        ) from exc

    offline = _env_truthy("TRANSFORMERS_OFFLINE") or _env_truthy("HF_HUB_OFFLINE")
    try:
        tok = AutoTokenizer.from_pretrained(resolved, local_files_only=True, trust_remote_code=True)
    except Exception as exc:
        if offline:
            raise RuntimeError(
                f"Tokenizer '{resolved}' was not in the local Hugging Face cache and offline mode is on."
            ) from exc
        tok = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)

    vocab_size = getattr(tok, "vocab_size", None) or 0
    try:
        vocab_size = max(vocab_size, len(tok))
    except Exception:
        pass

    adapter = TokenizerAdapter(
        name=resolved,
        resolved_name=resolved,
        source="transformers",
        vocab_size=int(vocab_size or 0),
        eot_token_id=getattr(tok, "eos_token_id", None),
        _tokenizer=tok,
    )
    _TOKENIZER_CACHE[resolved] = adapter
    return adapter
