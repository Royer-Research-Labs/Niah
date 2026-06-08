"""NIAH dataset spec: build haystacks, place needles, and (optionally) materialize.

A *spec* is a small JSON-able dict that deterministically describes a NIAH dataset
(tokenizer, context length, question, needle template, candidate choices, and the
depth schedule). From a spec you can either iterate rows lazily (``iter_niah_rows``,
used by the live eval) or materialize a JSONL dataset on disk (``materialize_niah_jsonl``,
used for the LM Eval Harness integration).

Each "row" is one (needle, depth) placement: a haystack of filler text with a single
needle sentence ("The secret keyword is X.") inserted at a given depth. The eval then
asks the model to retrieve X from a fixed set of candidate keywords.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

import numpy as np

from .tokenizer import TokenizerAdapter, load_tokenizer

NIAH_SPEC_VERSION = "niah-v3-spec"
NIAH_LEGACY_VERSION = "niah-v2"

DEFAULT_QUESTION = "### IMPORTANT DATA: The secret keyword is"
DEFAULT_FILLER_PHRASE = (
    "This document contains procedural notes about scheduling, budgeting, and system maintenance. "
    "It references tasks, owners, and brief status updates. "
    "The content includes repeated mentions of priorities, milestones, risk items, and follow-ups. "
    "Some sections contain numbered lists and short headings."
)
DEFAULT_NEEDLE_TEMPLATE = "The secret keyword is{answer}. "

# Verified intersection of single-token words for GPT-2 and Llama tokenizers.
SHARED_CHOICES = [
    " gold", " life", " time", " love", " work", " play", " game",
    " rain", " snow", " wind", " fire", " tree", " bird", " fish",
    " blue", " king", " star", " moon", " cake", " rose", " lion",
    " salt", " milk", " wine", " beer", " road", " hill", " book",
    " door", " hand", " foot", " head", " face", " eye", " ear",
]


def ok_answer_text(s: str, tokenizer) -> bool:
    """A choice is valid if it is a single-token, leading-space, short alnum word."""
    if not s or not s.startswith(" "):
        return False
    if any(c.isspace() for c in s[1:]):
        return False
    if any(c in s for c in "\n\r\t"):
        return False
    if any(ord(c) > 126 for c in s):
        return False
    body = s[1:]
    if not re.fullmatch(r"[a-zA-Z0-9_]{3,5}", body):
        return False
    if body[0] == "0":
        return False
    if len(tokenizer.encode(s)) != 1:
        return False
    return True


def _vocab_ids(tokenizer) -> list[int]:
    if getattr(tokenizer, "source", None) == "tiktoken":
        enc = tokenizer._tokenizer
        return list(range(enc.n_vocab))
    vocab = tokenizer._tokenizer.get_vocab()
    return list(vocab.values())


def sample_random_answer_texts(
    k: int,
    tokenizer,
    seed: int = 0,
    max_tries: int = 2_000_000,
) -> list[str]:
    """Sample ``k`` distinct single-token candidate keywords from the vocab."""
    rng = random.Random(seed)
    ids = _vocab_ids(tokenizer)

    answers: list[str] = []
    seen: set[str] = set()
    tries = 0
    while len(answers) < k and tries < max_tries:
        tries += 1
        s = tokenizer.decode([rng.choice(ids)])
        if s in seen or not ok_answer_text(s, tokenizer):
            continue
        answers.append(s)
        seen.add(s)

    if len(answers) < k:
        raise RuntimeError(f"Only found {len(answers)}/{k} valid choices after {tries} tries.")
    return answers


def make_needle(answer: str, needle_template: str = DEFAULT_NEEDLE_TEMPLATE) -> str:
    return needle_template.format(answer=answer)


def default_depth_schedule(num_samples: int) -> list[float]:
    return [float(x) for x in np.linspace(0.0, 1.0, int(num_samples)).tolist()]


def validate_niah_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("NIAH spec must be a JSON object.")
    if spec.get("version") != NIAH_SPEC_VERSION:
        raise ValueError(
            f"Unsupported NIAH spec version {spec.get('version')!r}; expected {NIAH_SPEC_VERSION!r}."
        )

    tokenizer_name = str(spec.get("canonical_tokenizer", "")).strip()
    if not tokenizer_name:
        raise ValueError("NIAH spec is missing canonical_tokenizer.")

    context_length = int(spec.get("context_length", 0))
    if context_length <= 0:
        raise ValueError("NIAH spec context_length must be > 0.")

    question = str(spec.get("question", "")).strip()
    if not question:
        raise ValueError("NIAH spec is missing question.")

    filler_phrase = str(spec.get("filler_phrase", ""))
    if not filler_phrase:
        raise ValueError("NIAH spec is missing filler_phrase.")

    needle_template = str(spec.get("needle_template", ""))
    if "{answer}" not in needle_template:
        raise ValueError("NIAH spec needle_template must contain '{answer}'.")

    raw_choices = spec.get("choices")
    if not isinstance(raw_choices, list) or not raw_choices:
        raise ValueError("NIAH spec choices must be a non-empty list.")
    choices = [str(choice) for choice in raw_choices]
    if not all(choice.startswith(" ") for choice in choices):
        raise ValueError("All NIAH choices must include a leading space.")

    raw_depths = spec.get("depths")
    if not isinstance(raw_depths, list) or not raw_depths:
        raise ValueError("NIAH spec depths must be a non-empty list.")
    depths = [float(depth) for depth in raw_depths]
    if any(depth < 0.0 or depth > 1.0 for depth in depths):
        raise ValueError("NIAH spec depths must be within [0.0, 1.0].")

    normalized = dict(spec)
    normalized["canonical_tokenizer"] = tokenizer_name
    normalized["context_length"] = context_length
    normalized["question"] = question
    normalized["filler_phrase"] = filler_phrase
    normalized["needle_template"] = needle_template
    normalized["choices"] = choices
    normalized["depths"] = depths
    normalized["seed"] = int(spec.get("seed", 42))
    normalized["num_samples"] = int(spec.get("num_samples", len(depths)))
    normalized["use_shared_choices"] = bool(spec.get("use_shared_choices", False))
    return normalized


def load_niah_spec(path_or_data: str | Path | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(path_or_data, dict):
        return validate_niah_spec(path_or_data)
    with Path(path_or_data).open("r", encoding="utf-8") as f:
        return validate_niah_spec(json.load(f))


def build_niah_spec(
    *,
    tokenizer_name: str,
    tokenizer: TokenizerAdapter | None = None,
    use_shared_choices: bool = False,
    num_samples: int = 100,
    context_length: int = 2048,
    n_choices: int = 64,
    seed: int = 42,
    question: str = DEFAULT_QUESTION,
    filler_phrase: str = DEFAULT_FILLER_PHRASE,
    needle_template: str = DEFAULT_NEEDLE_TEMPLATE,
    choices: Sequence[str] | None = None,
    depths: Sequence[float] | None = None,
) -> Dict[str, Any]:
    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_name)

    if choices is None:
        if use_shared_choices:
            n_choices = min(int(n_choices), len(SHARED_CHOICES))
            resolved_choices = list(SHARED_CHOICES[:n_choices])
        else:
            resolved_choices = sample_random_answer_texts(int(n_choices), tokenizer, seed=int(seed))
    else:
        resolved_choices = [str(choice) for choice in choices]

    if depths is None:
        resolved_depths = default_depth_schedule(int(num_samples))
    else:
        resolved_depths = [float(depth) for depth in depths]

    spec = {
        "version": NIAH_SPEC_VERSION,
        "canonical_tokenizer": tokenizer.resolved_name,
        "canonical_tokenizer_source": tokenizer.source,
        "context_length": int(context_length),
        "question": question,
        "filler_phrase": filler_phrase,
        "needle_template": needle_template,
        "choices": resolved_choices,
        "depths": resolved_depths,
        "seed": int(seed),
        "num_samples": len(resolved_depths),
        "use_shared_choices": bool(use_shared_choices),
    }
    return validate_niah_spec(spec)


def count_niah_rows(spec: Dict[str, Any]) -> int:
    spec = validate_niah_spec(spec)
    return len(spec["choices"]) * len(spec["depths"])


def _load_spec_tokenizer(spec: Dict[str, Any], tokenizer: TokenizerAdapter | None = None):
    if tokenizer is not None:
        return tokenizer
    return load_tokenizer(spec["canonical_tokenizer"])


def _filler_tokens_base(spec: Dict[str, Any], tokenizer) -> list[int]:
    return tokenizer.encode(spec["filler_phrase"] * 100)


def generate_control_haystack(
    spec: Dict[str, Any],
    tokenizer: TokenizerAdapter | None = None,
) -> Dict[str, Any]:
    """A needle-free haystack used to measure each choice's prior likelihood."""
    spec = validate_niah_spec(spec)
    tokenizer = _load_spec_tokenizer(spec, tokenizer=tokenizer)
    filler_tokens_base = _filler_tokens_base(spec, tokenizer)
    ctrl_haystack_tokens = filler_tokens_base[: spec["context_length"]]
    ctrl_haystack = tokenizer.decode(ctrl_haystack_tokens)
    return {
        "haystack": ctrl_haystack,
        "haystack_len": len(tokenizer.encode(ctrl_haystack)),
    }


def iter_niah_rows(
    spec: Dict[str, Any],
    tokenizer: TokenizerAdapter | None = None,
) -> Iterator[Dict[str, Any]]:
    """Yield one {haystack, needle, depth} row per (choice, depth) placement."""
    spec = validate_niah_spec(spec)
    tokenizer = _load_spec_tokenizer(spec, tokenizer=tokenizer)
    filler_tokens_base = _filler_tokens_base(spec, tokenizer)
    context_length = spec["context_length"]

    for answer in spec["choices"]:
        needle_tokens = tokenizer.encode(make_needle(answer, spec["needle_template"]))
        if len(needle_tokens) >= context_length:
            raise ValueError("Needle too long for requested context_length.")

        filler_needed = context_length - len(needle_tokens)
        reps = (filler_needed // len(filler_tokens_base)) + 2
        filler_tokens = (filler_tokens_base * reps)[:filler_needed]

        for depth in spec["depths"]:
            insert_pos = int(filler_needed * float(depth))
            haystack_tokens = (
                filler_tokens[:insert_pos] + needle_tokens + filler_tokens[insert_pos:]
            )
            if len(haystack_tokens) != context_length:
                raise AssertionError("Generated haystack length did not match context_length.")

            yield {
                "haystack": tokenizer.decode(haystack_tokens),
                "needle": answer,
                "depth": f"{float(depth):.2f}",
            }


def legacy_meta_from_spec(
    spec: Dict[str, Any],
    tokenizer: TokenizerAdapter | None = None,
) -> Dict[str, Any]:
    spec = validate_niah_spec(spec)
    control = generate_control_haystack(spec, tokenizer=tokenizer)
    return {
        "version": NIAH_LEGACY_VERSION,
        "generated_from_spec_version": spec["version"],
        "canonical_tokenizer": spec["canonical_tokenizer"],
        "context_length": spec["context_length"],
        "question": spec["question"],
        "choices": list(spec["choices"]),
        "control": control,
    }


def write_niah_spec(spec: Dict[str, Any], output_path: str | Path) -> Path:
    spec = validate_niah_spec(spec)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")
    return output_path


def materialize_niah_jsonl(
    spec: Dict[str, Any],
    output_path: str | Path,
    tokenizer: TokenizerAdapter | None = None,
) -> Dict[str, Any]:
    """Write every row to a JSONL file (plus a companion ``.meta.json``)."""
    spec = validate_niah_spec(spec)
    tokenizer = _load_spec_tokenizer(spec, tokenizer=tokenizer)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = legacy_meta_from_spec(spec, tokenizer=tokenizer)
    meta_path = output_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as mf:
        json.dump(meta, mf)

    row_count = 0
    last_enc_prompt_len = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in iter_niah_rows(spec, tokenizer=tokenizer):
            json.dump(row, f)
            f.write("\n")
            row_count += 1

            test_prompt = f"{row['haystack']}\n\n{spec['question']}"
            last_enc_prompt_len = len(tokenizer.encode(test_prompt))
            if last_enc_prompt_len <= spec["context_length"]:
                raise AssertionError(
                    f"Encoded prompt length mismatch: expected > {spec['context_length']} "
                    f"found {last_enc_prompt_len}"
                )

    return {
        "output_path": output_path,
        "meta_path": meta_path,
        "row_count": row_count,
        "last_prompt_length": last_enc_prompt_len,
        "control_haystack_len": meta["control"]["haystack_len"],
    }
