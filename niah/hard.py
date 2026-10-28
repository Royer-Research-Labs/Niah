"""NIAH hard mode: the same retrieval question, under harder conditions.

The standard spec plants one "The secret keyword is X." in one repeated filler paragraph. Hard mode keeps
the goal (retrieve one planted keyword, scored by likelihood over the same candidate choices, with the same
control prior) and removes the two things that make the standard test easy:

* **Natural filler.** The haystack is a contiguous slice of a long natural-text file, starting at a seeded
  offset that differs per placement, so nothing repeats and the needle does not stand out from a periodic
  background.
* **Distractors that require binding.** Besides the target, ``num_distractors`` other facts of the same form
  are planted at seeded depths: "The secret keyword for the <key> project is <answer>." The question names
  one key, and every distractor's answer is itself one of the candidate choices, so a model that retrieves
  "a keyword" without binding it to its key picks a distractor. ``distractor_pick_rate`` reports how often
  the model's top choice was a distractor's answer.

Rows, control haystack and lengths follow the v4 conventions: ``context_length`` is the whole scored input
(haystack + separator + question + all but the last token of the longest choice).
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

from .spec import (
    LENGTH_BASIS_INPUT,
    SHARED_CHOICES,
    default_depth_schedule,
    input_length,
    make_prompt,
    max_choice_tokens,
    sample_random_answer_texts,
)
from .tokenizer import TokenizerAdapter, load_tokenizer

NIAH_HARD_SPEC_VERSION = "niah-v4-hard"
HARD_NEEDLE_TEMPLATE = "The secret keyword for the {key} project is{answer}. "
HARD_QUESTION_TEMPLATE = "### IMPORTANT DATA: The secret keyword for the {key} project is"
DEFAULT_KEYS = ["Alder", "Birch", "Cedar", "Maple", "Willow", "Aspen", "Hazel", "Juniper", "Laurel",
                "Spruce", "Poplar", "Rowan", "Cypress", "Linden", "Sequoia", "Walnut"]
CONTROL_KEY = "Hawthorn"   # never planted: the control question names a key absent from every haystack

_FILLER_CACHE: dict[tuple[str, str], list[int]] = {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_niah_hard_spec(
    *,
    tokenizer_name: str,
    filler_path: str | Path,
    tokenizer: TokenizerAdapter | None = None,
    use_shared_choices: bool = True,
    num_samples: int = 20,
    context_length: int = 2048,
    n_choices: int = 8,
    num_distractors: int = 3,
    seed: int = 42,
    keys: Sequence[str] = DEFAULT_KEYS,
    depths: Sequence[float] | None = None,
) -> Dict[str, Any]:
    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_name)
    filler_path = Path(filler_path)
    if not filler_path.is_file():
        raise ValueError(f"hard mode needs a natural-text filler file; {filler_path} does not exist")
    if use_shared_choices:
        choices = list(SHARED_CHOICES[: min(int(n_choices), len(SHARED_CHOICES))])
    else:
        choices = sample_random_answer_texts(int(n_choices), tokenizer, seed=int(seed))
    if not 0 <= int(num_distractors) < len(choices):
        raise ValueError("num_distractors must be at least 0 and smaller than the number of choices")
    if int(num_distractors) >= len(keys):
        raise ValueError("num_distractors must be smaller than the number of keys")
    spec = {
        "version": NIAH_HARD_SPEC_VERSION,
        "mode": "hard",
        "canonical_tokenizer": tokenizer.resolved_name,
        "canonical_tokenizer_source": tokenizer.source,
        "context_length": int(context_length),
        "length_basis": LENGTH_BASIS_INPUT,
        "question": HARD_QUESTION_TEMPLATE,
        "needle_template": HARD_NEEDLE_TEMPLATE,
        "keys": [str(k) for k in keys],
        "num_distractors": int(num_distractors),
        "filler_path": str(filler_path),
        "filler_sha256": _sha256(filler_path),
        "choices": choices,
        "depths": [float(d) for d in (depths if depths is not None else default_depth_schedule(int(num_samples)))],
        "seed": int(seed),
        "use_shared_choices": bool(use_shared_choices),
    }
    spec["num_samples"] = len(spec["depths"])
    return spec


def _filler_tokens(spec: Dict[str, Any], tokenizer) -> list[int]:
    key = (spec["filler_path"], spec["filler_sha256"])
    if key not in _FILLER_CACHE:
        path = Path(spec["filler_path"])
        if _sha256(path) != spec["filler_sha256"]:
            raise ValueError(f"filler file {path} does not match the spec's sha256")
        _FILLER_CACHE[key] = tokenizer.encode(path.read_text(encoding="utf-8"))
    return _FILLER_CACHE[key]


def _keyed(spec: Dict[str, Any], key: str) -> Dict[str, Any]:
    """A spec view for one key: question and needle template with the key filled in (for input_length,
    _fit_to_input_length and locate_needle)."""
    return {**spec, "question": spec["question"].format(key=key),
            "needle_template": spec["needle_template"].replace("{key}", key)}


def _fit(view: Dict[str, Any], tokenizer, build, start: int) -> tuple[str, int]:
    """Size a haystack so the scored input is exactly view["context_length"] tokens, or the longest that stays
    within it. Like spec._fit_to_input_length, but for natural filler, where re-encoding a decoded slice moves
    the length by a few tokens unpredictably: a few corrections from `start`, then a small local search.
    Each candidate is encoded once."""
    target = view["context_length"]
    extra = max_choice_tokens(view, tokenizer) - 1
    seen: dict[int, tuple[str, int]] = {}

    def measure(n: int) -> tuple[str, int]:
        if n not in seen:
            haystack = build(n)
            seen[n] = (haystack, len(tokenizer.encode(make_prompt(haystack, view["question"]))) + extra)
        return seen[n]

    n = max(0, start)
    for _ in range(6):
        haystack, got = measure(n)
        if got == target:
            return haystack, got
        n = max(0, n + target - got)
    best = None
    for m in sorted(range(max(0, n - 24), n + 25), key=lambda m: abs(m - n)):
        haystack, got = measure(m)
        if got == target:
            return haystack, got
        if got < target and (best is None or got > best[1]):
            best = (haystack, got)
    if best is None:
        raise ValueError(f"could not fit a haystack within context_length {target}")
    return best


def _needle_tokens(spec, tokenizer, key: str, answer: str) -> list[int]:
    return tokenizer.encode(spec["needle_template"].format(key=key, answer=answer))


def _offset(rng: random.Random, filler: list[int], need: int) -> int:
    if len(filler) < need + 64:
        raise ValueError(f"filler text has {len(filler):,} tokens; context_length needs at least {need + 64:,}")
    return rng.randrange(0, len(filler) - need - 64)


def generate_hard_control(spec: Dict[str, Any], tokenizer: TokenizerAdapter | None = None) -> Dict[str, Any]:
    """A needle-free natural-text haystack and a question about a key that is never planted."""
    tokenizer = tokenizer or load_tokenizer(spec["canonical_tokenizer"])
    filler = _filler_tokens(spec, tokenizer)
    view = _keyed(spec, CONTROL_KEY)
    offset = _offset(random.Random(f"{spec['seed']}-control"), filler, spec["context_length"])
    start = spec["context_length"] - input_length(view, tokenizer, "")
    haystack, got = _fit(view, tokenizer, lambda n: tokenizer.decode(filler[offset:offset + n]), start)
    return {"haystack": haystack, "question": view["question"],
            "haystack_len": len(tokenizer.encode(haystack)), "input_len": got}


def iter_niah_hard_rows(spec: Dict[str, Any], tokenizer: TokenizerAdapter | None = None) -> Iterator[Dict[str, Any]]:
    """One row per (choice, depth): natural filler from a seeded offset, the target fact at `depth`, and
    `num_distractors` facts about other keys (answers drawn from the other choices) at seeded depths."""
    tokenizer = tokenizer or load_tokenizer(spec["canonical_tokenizer"])
    filler = _filler_tokens(spec, tokenizer)
    context_length = spec["context_length"]
    for ai, answer in enumerate(spec["choices"]):
        for di, depth in enumerate(spec["depths"]):
            rng = random.Random(f"{spec['seed']}-{ai}-{di}")
            target_key, *other_keys = rng.sample(spec["keys"], spec["num_distractors"] + 1)
            others = rng.sample([c for c in spec["choices"] if c != answer], spec["num_distractors"])
            distractors = [{"key": k, "answer": a, "depth": round(rng.random(), 4)}
                           for k, a in zip(other_keys, others)]
            view = _keyed(spec, target_key)
            planted = [(float(depth), _needle_tokens(spec, tokenizer, target_key, answer))]
            planted += [(d["depth"], _needle_tokens(spec, tokenizer, d["key"], d["answer"])) for d in distractors]
            needles_len = sum(len(t) for _, t in planted)
            # 8 tokens of headroom: merges at the needles' joins can add a token each; the fit below
            # lengthens the tail back to exactly context_length
            filler_estimate = max(0, context_length - 8 - input_length(
                view, tokenizer, tokenizer.decode([t for _, ts in planted for t in ts])))
            offset = _offset(rng, filler, context_length)
            # insert points are fixed from the estimate (see spec.iter_niah_rows); later needles first so
            # earlier insert points stay valid
            inserts = sorted(((int(filler_estimate * d), ts) for d, ts in planted), key=lambda x: x[0])

            def build(n: int, inserts=inserts, offset=offset) -> str:
                body = list(filler[offset:offset + max(n, inserts[-1][0])])
                for pos, ts in reversed(inserts):
                    body[pos:pos] = ts
                return tokenizer.decode(body)

            haystack, got = _fit(view, tokenizer, build, filler_estimate)
            yield {"haystack": haystack, "needle": answer, "depth": f"{float(depth):.2f}",
                   "key": target_key, "question": view["question"], "distractors": distractors,
                   "filler_offset": offset, "needle_tokens": needles_len,
                   "haystack_len": len(tokenizer.encode(haystack)), "input_len": got}


def keyed_spec_for_row(spec: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    """The spec view locate_needle needs to find this row's target fact."""
    return _keyed(spec, row["key"])
