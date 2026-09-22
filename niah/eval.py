"""The NIAH eval: likelihood-based multiple-choice retrieval with a control prior.

Method
------
For a fixed set of candidate keywords ("choices"), we build haystacks that embed
one needle ("The secret keyword is X.") at a range of depths, then ask the model
to retrieve X by scoring every choice as a continuation of:

    <haystack>\n\n### IMPORTANT DATA: The secret keyword is

The choice with the highest log-likelihood is the model's answer; it "passes" the
row if that choice is the true needle.

Control prior
-------------
Small models have lexical biases: some keywords are simply more likely regardless
of the haystack. Before scoring, we run the choices against a *needle-free* control
haystack and disqualify the ones the model already prefers a priori (top-k and/or
large-margin). This isolates genuine retrieval from prior preference, and we also
report a ``prior_adjusted_lift`` (retrieval gap minus the control margin).

No text generation is required, which makes the eval cheap and stable for tiny
from-scratch models. ``run_niah`` returns a flat metrics dict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .common import all_reduce_sum, dist_info
from .model_adapter import NanoGPTAdapter
from .spec import build_niah_spec, generate_control_haystack, iter_niah_rows, make_prompt
from .tokenizer import TokenizerAdapter, load_tokenizer


@dataclass
class NiahConfig:
    """Configuration for a single NIAH run."""

    context_length: int = 1024       # tokens of scored model input (prompt + choice), not haystack
    num_needles: int = 4              # number of candidate keywords (choices)
    placements_per_needle: int = 10   # depth positions per needle
    seed: int = 42
    use_shared_choices: bool = True   # use the curated single-token word list
    top_k_disqualify: int = 1         # disqualify the top-k a-priori-preferred choices
    gap_disqualify: float = 2.0       # ...or any choice with a control margin >= this
    win_retrieval_gap_ll: float = 0.0 # min log-likelihood gap to count a row as a pass
    canonical_tokenizer: str = "gpt2"


def _get_prompt(haystack: str, question: str) -> str:
    return make_prompt(haystack, question)


def _get_choices_with_needle(needle: str, choices: list[str]) -> list[str]:
    return [needle, *[c for c in choices if c != needle]]


@dataclass
class ChoiceScore:
    choice: str
    ll: float
    avg: float
    tok_len: int


@dataclass
class ControlPriorResult:
    ll_by_choice: Dict[str, float]
    rank_by_choice: Dict[str, int]
    control_margin_by_choice: Dict[str, float]
    prior_disqualified: set[str]


def _score_choices(adapter: NanoGPTAdapter, context: str, choices: list[str]) -> list[ChoiceScore]:
    ctx_ids = adapter.encode(context)
    choice_tokens = [(choice, adapter.encode(choice)) for choice in choices]

    # Fast path: all choices are a single token -> one forward pass.
    if all(len(ids) == 1 for _, ids in choice_tokens):
        x = torch.tensor(ctx_ids, dtype=torch.long, device=adapter.device).unsqueeze(0)
        next_token_logits = adapter.get_next_token_logits(x)[0]
        lp = F.log_softmax(next_token_logits.float(), dim=-1)
        return [
            ChoiceScore(choice=choice, ll=float(lp[ids[0]].item()), tok_len=1, avg=float(lp[ids[0]].item()))
            for choice, ids in choice_tokens
        ]

    out: list[ChoiceScore] = []
    for choice, ids in choice_tokens:
        ll = adapter.score_continuation(ctx_ids, ids)
        out.append(ChoiceScore(choice=choice, ll=ll, tok_len=len(ids), avg=ll / len(ids)))
    return out


def _compute_metrics(scores: list[ChoiceScore], needle: str, min_retrieval_gap_ll: float = 0.0) -> Dict[str, Any]:
    by_ll = sorted(scores, key=lambda s: s.ll, reverse=True)
    by_avg = sorted(scores, key=lambda s: s.avg, reverse=True)
    needle_item = next((s for s in scores if s.choice == needle), None)
    if needle_item is None:
        raise ValueError(f"Needle {needle!r} not found in choices")

    rank_ll = next(i for i, s in enumerate(by_ll) if s.choice == needle)
    rank_avg = next(i for i, s in enumerate(by_avg) if s.choice == needle)
    best_wrong_ll = next(s for s in by_ll if s.choice != needle)
    best_wrong_avg = next(s for s in by_avg if s.choice != needle)

    retrieval_gap_ll = needle_item.ll - best_wrong_ll.ll
    margin_avg = needle_item.avg - best_wrong_avg.avg
    best_total = by_ll[0]
    best_avg = by_avg[0]

    return {
        "needle_ll": needle_item.ll,
        "needle_avg": needle_item.avg,
        "rank_ll": rank_ll,
        "rank_avg": rank_avg,
        "best_total_choice": best_total.choice,
        "best_avg_choice": best_avg.choice,
        "best_wrong_ll_choice": best_wrong_ll.choice,
        "retrieval_gap_ll": retrieval_gap_ll,
        "margin_avg": margin_avg,
        "pass_total": (best_total.choice == needle and retrieval_gap_ll >= min_retrieval_gap_ll),
        "pass_avg": (best_avg.choice == needle),
    }


def _score_control(
    adapter: NanoGPTAdapter,
    control_haystack: str,
    question: str,
    choice_list: List[str],
    *,
    top_k_disqualify: int = 1,
    gap_disqualify: float = float("inf"),
) -> ControlPriorResult:
    scores = _score_choices(adapter, _get_prompt(control_haystack, question), choice_list)

    ll_by_choice = {s.choice: float(s.ll) for s in scores}
    ranked = sorted(ll_by_choice.items(), key=lambda kv: kv[1], reverse=True)
    rank_by_choice = {c: i for i, (c, _) in enumerate(ranked)}

    control_margin_by_choice: Dict[str, float] = {}
    if len(ranked) <= 1:
        if ranked:
            control_margin_by_choice[ranked[0][0]] = 0.0
    else:
        best_choice, best_ll = ranked[0]
        second_best_ll = ranked[1][1]
        for choice, ll in ranked:
            best_other = second_best_ll if choice == best_choice else best_ll
            control_margin_by_choice[choice] = float(ll - best_other)

    prior_disqualified: set[str] = set()
    if top_k_disqualify > 0:
        prior_disqualified.update([choice for (choice, _) in ranked[:top_k_disqualify]])
    if len(ranked) > 1 and float(ranked[0][1] - ranked[1][1]) >= gap_disqualify:
        prior_disqualified.add(ranked[0][0])

    return ControlPriorResult(
        ll_by_choice=ll_by_choice,
        rank_by_choice=rank_by_choice,
        control_margin_by_choice=control_margin_by_choice,
        prior_disqualified=prior_disqualified,
    )


def _bucket_of(depth_f: float) -> str:
    if depth_f < 0.34:
        return "early"
    if depth_f < 0.67:
        return "mid"
    return "late"


@torch.no_grad()
def run_niah(
    model,
    *,
    config: NiahConfig | None = None,
    tokenizer: TokenizerAdapter | None = None,
    device: str | None = None,
    ctx=None,
    rows: list[dict] | None = None,
) -> Dict[str, Any]:
    """Run the NIAH eval against a (nanoGPT-style) ``model``.

    Parameters
    ----------
    model : a nanoGPT ``GPT`` (or anything exposing ``forward(idx, targets=None) -> (logits, loss)``
            and ``.config.block_size``).
    config : a ``NiahConfig`` (defaults are used if omitted).
    tokenizer : a ``TokenizerAdapter``; loaded from ``config.canonical_tokenizer`` if omitted.
    device : torch device string; inferred from the model if omitted.
    ctx : an autocast context manager (``nullcontext()`` if omitted).
    rows : optionally provide pre-built rows (each {haystack, needle, depth}); otherwise
           they are generated from the config. The control haystack is always regenerated.
    """
    import contextlib

    config = config or NiahConfig()
    if device is None:
        device = str(next(model.parameters()).device)
    if ctx is None:
        ctx = contextlib.nullcontext()
    if tokenizer is None:
        tokenizer = load_tokenizer(config.canonical_tokenizer)
    canonical_tokenizer = tokenizer.resolved_name

    adapter = NanoGPTAdapter(model=model, ctx=ctx, device=device, tokenizer=tokenizer)
    dist_active, rank, world_size, backend = dist_info()

    if adapter.block_size and config.context_length > adapter.block_size:
        return {
            "error": (
                f"NIAH context_length {config.context_length} exceeds model block_size "
                f"{adapter.block_size}. Reduce context_length or train a longer-context model."
            )
        }

    spec = build_niah_spec(
        tokenizer_name=canonical_tokenizer,
        tokenizer=tokenizer,
        use_shared_choices=bool(config.use_shared_choices),
        num_samples=int(config.placements_per_needle),
        context_length=int(config.context_length),
        n_choices=int(config.num_needles),
        seed=int(config.seed),
    )

    rows_iter = iter(rows) if rows is not None else iter_niah_rows(spec, tokenizer=tokenizer)
    try:
        first_row = next(rows_iter)
    except StopIteration:
        return {"error": "Generated NIAH spec was empty."}

    # context_length is the scored input length (prompt + choice), and every row is built
    # to it exactly, so it fits whenever context_length <= block_size.
    first_input_len = int(first_row.get("input_len", 0)) or len(
        adapter.encode(_get_prompt(first_row["haystack"], spec["question"])))
    if adapter.block_size and first_input_len > adapter.block_size:
        return {
            "error": (
                f"NIAH input length {first_input_len} exceeds model block_size "
                f"{adapter.block_size}. Use context_length <= block_size."
            )
        }

    start = time.time()
    control = generate_control_haystack(spec, tokenizer=tokenizer)
    ctrl = _score_control(
        adapter,
        control_haystack=control["haystack"],
        question=spec["question"],
        choice_list=list(spec["choices"]),
        top_k_disqualify=int(config.top_k_disqualify),
        gap_disqualify=float(config.gap_disqualify),
    )

    filtered_choices = [c for c in spec["choices"] if c not in ctrl.prior_disqualified]
    if not filtered_choices:
        return {
            "task": "niah", "accuracy": 0.0, "total": 0, "correct": 0,
            "duration_s": time.time() - start, "context_length": config.context_length,
            "placements_per_needle": config.placements_per_needle, "num_needles": config.num_needles,
            "evaluated_needles": 0, "disqualified_needles": len(ctrl.prior_disqualified),
            "avg_retrieval_gap_ll": 0.0,
        }

    correct = total = pass_avg = 0
    gap_sum = margin_avg_sum = prior_adjusted_lift_sum = 0.0
    bucket_correct = {"early": 0, "mid": 0, "late": 0}
    bucket_total = {"early": 0, "mid": 0, "late": 0}
    input_lens: list[int] = []
    haystack_lens: list[int] = []

    for row_idx, row in enumerate(chain([first_row], rows_iter)):
        if dist_active and world_size > 1 and row_idx % world_size != rank:
            continue
        needle = row["needle"]
        if needle in ctrl.prior_disqualified:
            continue

        prompt = _get_prompt(row["haystack"], spec["question"])
        if "input_len" in row:
            input_lens.append(int(row["input_len"]))
            haystack_lens.append(int(row["haystack_len"]))
        scores = _score_choices(adapter, prompt, _get_choices_with_needle(needle, filtered_choices))
        metrics = _compute_metrics(scores, needle, min_retrieval_gap_ll=float(config.win_retrieval_gap_ll))

        total += 1
        correct += int(metrics["pass_total"])
        pass_avg += int(metrics["pass_avg"])
        gap_sum += float(metrics["retrieval_gap_ll"])
        margin_avg_sum += float(metrics["margin_avg"])
        prior_adjusted_lift_sum += float(metrics["retrieval_gap_ll"] - ctrl.control_margin_by_choice.get(needle, 0.0))

        bucket = _bucket_of(float(row["depth"]))
        bucket_total[bucket] += 1
        bucket_correct[bucket] += int(metrics["pass_total"])

    duration = time.time() - start
    if dist_active and world_size > 1:
        reduced = all_reduce_sum(
            [float(correct), float(total), float(pass_avg), float(gap_sum), float(margin_avg_sum),
             float(prior_adjusted_lift_sum), float(bucket_correct["early"]), float(bucket_correct["mid"]),
             float(bucket_correct["late"]), float(bucket_total["early"]), float(bucket_total["mid"]),
             float(bucket_total["late"])],
            device=device, backend=backend,
        )
        (correct, total, pass_avg, gap_sum, margin_avg_sum, prior_adjusted_lift_sum,
         bucket_correct["early"], bucket_correct["mid"], bucket_correct["late"],
         bucket_total["early"], bucket_total["mid"], bucket_total["late"]) = reduced

    accuracy = (correct / total) if total else 0.0
    result = {
        "task": "niah",
        "accuracy": accuracy,
        "total": int(total),
        "correct": int(correct),
        "duration_s": duration,
        "context_length": config.context_length,
        "length_basis": spec["length_basis"],
        "input_length_max": max(input_lens) if input_lens else None,
        "input_length_min": min(input_lens) if input_lens else None,
        "haystack_length_mean": (sum(haystack_lens) / len(haystack_lens)) if haystack_lens else None,
        "placements_per_needle": config.placements_per_needle,
        "num_needles": config.num_needles,
        "evaluated_needles": len(filtered_choices),
        "disqualified_needles": len(ctrl.prior_disqualified),
        "avg_retrieval_gap_ll": (gap_sum / total) if total else 0.0,
        "avg_margin_avg": (margin_avg_sum / total) if total else 0.0,
        "avg_prior_adjusted_lift_ll": (prior_adjusted_lift_sum / total) if total else 0.0,
        "avg_pass_accuracy": (pass_avg / total) if total else 0.0,
    }
    for bucket_name in ("early", "mid", "late"):
        n = bucket_total[bucket_name]
        result[f"{bucket_name}_accuracy"] = (bucket_correct[bucket_name] / n) if n else 0.0
        result[f"{bucket_name}_total"] = int(n)
    return result
