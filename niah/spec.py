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

NIAH_SPEC_VERSION = "niah-v4-spec"
NIAH_V3_SPEC_VERSION = "niah-v3-spec"
NIAH_LEGACY_VERSION = "niah-v2"

# What `context_length` measures. v4 specs mean the whole scored model input: the prompt
# (haystack + separator + question) plus any continuation tokens fed back in to score a
# multi-token choice, so a context_length equal to the model's block size fits exactly.
# v3 specs meant the haystack alone, and every prompt ran past context_length by the
# question's length; they still load, with that meaning.
LENGTH_BASIS_INPUT = "input"
LENGTH_BASIS_HAYSTACK = "haystack"

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


def make_prompt(haystack: str, question: str) -> str:
    """The scored prompt; every choice is scored as a continuation of it."""
    return f"{haystack}\n\n{question}"


def default_depth_schedule(num_samples: int) -> list[float]:
    return [float(x) for x in np.linspace(0.0, 1.0, int(num_samples)).tolist()]


def validate_niah_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("NIAH spec must be a JSON object.")
    version = spec.get("version")
    if version not in (NIAH_SPEC_VERSION, NIAH_V3_SPEC_VERSION):
        raise ValueError(
            f"Unsupported NIAH spec version {version!r}; expected {NIAH_SPEC_VERSION!r} "
            f"(or {NIAH_V3_SPEC_VERSION!r})."
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
    normalized["length_basis"] = (
        LENGTH_BASIS_INPUT if version == NIAH_SPEC_VERSION else LENGTH_BASIS_HAYSTACK
    )
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


def _repeat_to(tokens: list[int], n: int) -> list[int]:
    return (tokens * (n // len(tokens) + 2))[:n]


def max_choice_tokens(spec: Dict[str, Any], tokenizer) -> int:
    return max(len(tokenizer.encode(choice)) for choice in spec["choices"])


def input_length(spec: Dict[str, Any], tokenizer, haystack: str) -> int:
    """Tokens the model is fed to score the longest choice after this haystack: the
    prompt, plus all but the last token of that choice. Single-token choices are read
    off the prompt's last position, so there the input is exactly the prompt."""
    prompt = tokenizer.encode(make_prompt(haystack, spec["question"]))
    return len(prompt) + max_choice_tokens(spec, tokenizer) - 1


def _fit_to_input_length(spec: Dict[str, Any], tokenizer, build) -> tuple[str, int]:
    """Size a haystack so the scored input is exactly spec["context_length"] tokens.

    ``build(n)`` returns the haystack text built from n filler tokens. The haystack is
    decoded and re-encoded together with the question, so byte-pair merges at the joins
    can move the true length by a token or two; the filler count is corrected until the
    measured input length matches. If merges make the exact length unreachable, the
    longest haystack that stays within context_length is used, never one that overflows.
    """
    target = spec["context_length"]
    n = target - input_length(spec, tokenizer, build(0))
    if n < 0:
        raise ValueError(
            f"context_length {target} is too short for the needle, question and choices "
            f"(they need {target - n} tokens)."
        )
    under: dict[int, tuple[str, int]] = {}
    tried = set()
    while n >= 0 and n not in tried:
        tried.add(n)
        haystack = build(n)
        got = input_length(spec, tokenizer, haystack)
        if got == target:
            return haystack, got
        if got < target:
            under[n] = (haystack, got)
        n += target - got
    # Moving the needle's insert point by a token can change a merge, so the length can
    # jump by two (e.g. 255 -> 257) and the correction above oscillates. Search nearby.
    lo, hi = max(0, min(tried) - 8), max(tried) + 8
    for m in range(lo, hi + 1):
        if m in tried:
            continue
        haystack = build(m)
        got = input_length(spec, tokenizer, haystack)
        if got == target:
            return haystack, got
        if got < target:
            under[m] = (haystack, got)
    if not under:
        # every correction overshot: step down until the input fits
        n = min(tried)
        while n > 0:
            n -= 1
            haystack = build(n)
            got = input_length(spec, tokenizer, haystack)
            if got <= target:
                return haystack, got
        raise ValueError(f"could not fit a haystack within context_length {target}.")
    return max(under.values(), key=lambda hg: hg[1])


def generate_control_haystack(
    spec: Dict[str, Any],
    tokenizer: TokenizerAdapter | None = None,
) -> Dict[str, Any]:
    """A needle-free haystack used to measure each choice's prior likelihood, sized like
    the rows so its prompt is scored at the same length."""
    spec = validate_niah_spec(spec)
    tokenizer = _load_spec_tokenizer(spec, tokenizer=tokenizer)
    filler_tokens_base = _filler_tokens_base(spec, tokenizer)
    if spec["length_basis"] == LENGTH_BASIS_HAYSTACK:
        ctrl_haystack = tokenizer.decode(filler_tokens_base[: spec["context_length"]])
    else:
        ctrl_haystack, _ = _fit_to_input_length(
            spec, tokenizer, lambda n: tokenizer.decode(_repeat_to(filler_tokens_base, n))
        )
    return {
        "haystack": ctrl_haystack,
        "haystack_len": len(tokenizer.encode(ctrl_haystack)),
        "input_len": input_length(spec, tokenizer, ctrl_haystack),
    }


def iter_niah_rows(
    spec: Dict[str, Any],
    tokenizer: TokenizerAdapter | None = None,
) -> Iterator[Dict[str, Any]]:
    """Yield one {haystack, needle, depth} row per (choice, depth) placement.

    For v4 specs every row's scored input (see ``input_length``) is exactly
    context_length tokens; ``input_len`` and ``haystack_len`` record what was built.
    """
    spec = validate_niah_spec(spec)
    tokenizer = _load_spec_tokenizer(spec, tokenizer=tokenizer)
    filler_tokens_base = _filler_tokens_base(spec, tokenizer)
    context_length = spec["context_length"]

    for answer in spec["choices"]:
        needle_tokens = tokenizer.encode(make_needle(answer, spec["needle_template"]))
        if len(needle_tokens) >= context_length:
            raise ValueError("Needle too long for requested context_length.")

        # Filler count that should fill the input once the question is added; the needle's
        # insert point is fixed from it, so fitting only lengthens or shortens the tail.
        # (If the insert point moved with the filler count, a merge at the needle's joins
        # could make the length jump by two and the exact length unreachable.)
        if spec["length_basis"] == LENGTH_BASIS_HAYSTACK:
            filler_estimate = context_length - len(needle_tokens)
        else:
            filler_estimate = max(0, context_length - input_length(
                spec, tokenizer, tokenizer.decode(needle_tokens)))

        for depth in spec["depths"]:
            insert_pos = int(filler_estimate * float(depth))

            def build(filler_needed: int, insert_pos: int = insert_pos) -> str:
                filler_tokens = _repeat_to(filler_tokens_base, max(filler_needed, insert_pos))
                return tokenizer.decode(
                    filler_tokens[:insert_pos] + needle_tokens + filler_tokens[insert_pos:]
                )

            if spec["length_basis"] == LENGTH_BASIS_HAYSTACK:
                haystack = build(filler_estimate)
                got = input_length(spec, tokenizer, haystack)
            else:
                haystack, got = _fit_to_input_length(spec, tokenizer, build)

            yield {
                "haystack": haystack,
                "needle": answer,
                "depth": f"{float(depth):.2f}",
                "haystack_len": len(tokenizer.encode(haystack)),
                "input_len": got,
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
        "length_basis": spec["length_basis"],
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

            last_enc_prompt_len = len(tokenizer.encode(make_prompt(row["haystack"], spec["question"])))
            if spec["length_basis"] == LENGTH_BASIS_INPUT and row["input_len"] > spec["context_length"]:
                raise AssertionError(
                    f"Scored input {row['input_len']} tokens exceeds context_length "
                    f"{spec['context_length']}"
                )

    return {
        "output_path": output_path,
        "meta_path": meta_path,
        "row_count": row_count,
        "last_prompt_length": last_enc_prompt_len,
        "control_haystack_len": meta["control"]["haystack_len"],
    }
