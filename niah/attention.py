"""Attention diagnostics on NIAH prompts: where does the answer position look?

The likelihood score (``niah.eval``) says whether a model retrieves; this says which layers
and heads do the retrieving. For each NIAH row it encodes the scored prompt, locates the
needle and its answer token (``niah.spec.locate_needle``), and asks the caller for the
attention row of the *last* prompt position in every layer. That position predicts the
answer. It then averages over rows:

    answer   attention mass on the answer token
    needle   mass on the whole needle sentence
    first    mass on token 0 (the attention sink)

Extracting attention is model-specific, so the model is represented only by a callback:

    last_row_attention(input_ids: LongTensor[1, T]) -> list of (heads, T) tensors, one per layer

where each tensor is the attention distribution of position T-1 in that layer.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import torch

from .spec import build_niah_spec, iter_niah_rows, locate_needle, make_prompt
from .tokenizer import TokenizerAdapter, load_tokenizer

LastRowAttention = Callable[[torch.Tensor], Sequence[torch.Tensor]]


@torch.no_grad()
def needle_attention_profile(
    last_row_attention: LastRowAttention,
    *,
    context_length: int,
    num_needles: int = 4,
    placements_per_needle: int = 5,
    seed: int = 42,
    tokenizer: TokenizerAdapter | None = None,
    device: str | torch.device = "cpu",
    spec: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Per-layer, per-head attention from the answer position onto the needle.

    Returns ``{"context_length", "rows", "skipped", "layers": [{"layer", "answer", "needle",
    "first"}]}`` where each metric is a per-head list averaged over rows, plus
    ``*_mean`` / ``*_max_head`` summaries.
    """
    tokenizer = tokenizer or load_tokenizer("gpt2")
    if spec is None:
        spec = build_niah_spec(
            tokenizer_name=tokenizer.resolved_name, tokenizer=tokenizer, use_shared_choices=True,
            num_samples=placements_per_needle, context_length=context_length,
            n_choices=num_needles, seed=seed,
        )
    sums: Dict[str, List[torch.Tensor]] | None = None
    rows = skipped = 0
    for row in iter_niah_rows(spec, tokenizer=tokenizer):
        ids = tokenizer.encode(make_prompt(row["haystack"], spec["question"]))
        where = locate_needle(ids, row["needle"], spec, tokenizer)
        if where is None:
            skipped += 1
            continue
        start, end, ans = where
        per_layer = last_row_attention(torch.tensor([ids], dtype=torch.long, device=device))
        vals = {
            "answer": [a[:, ans].float().cpu() for a in per_layer],
            "needle": [a[:, start:end].float().sum(-1).cpu() for a in per_layer],
            "first": [a[:, 0].float().cpu() for a in per_layer],
        }
        if sums is None:
            sums = vals
        else:
            for key in sums:
                sums[key] = [s + v for s, v in zip(sums[key], vals[key])]
        rows += 1
    out: Dict[str, Any] = {"context_length": spec["context_length"], "rows": rows, "skipped": skipped, "layers": []}
    if not rows:
        return out
    for i in range(len(sums["answer"])):
        layer: Dict[str, Any] = {"layer": i}
        for key in sums:
            heads = sums[key][i] / rows
            layer[key] = heads.tolist()
            layer[key + "_mean"] = heads.mean().item()
            layer[key + "_max_head"] = heads.max().item()
        out["layers"].append(layer)
    return out
