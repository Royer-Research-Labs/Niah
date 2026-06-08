"""Adapter that exposes a nanoGPT ``GPT`` model to the NIAH scorer.

The eval only needs three operations from a model:
  1. ``encode`` text to token ids,
  2. next-token log-probabilities given a context (fast path for single-token choices),
  3. the summed log-probability of a continuation given a context (multi-token choices).

This targets the vanilla nanoGPT ``GPT.forward(idx, targets=None)`` API
(``model.py``), which returns ``(logits, loss)``. When ``targets`` is passed it
returns logits for *all* positions; otherwise only the last position. We compute
continuation log-probs ourselves from the logits rather than relying on the
model's mean-reduced cross-entropy loss.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from .tokenizer import TokenizerAdapter


class NanoGPTAdapter:
    def __init__(self, model, ctx, device: str, tokenizer: TokenizerAdapter):
        # unwrap DDP / torch.compile wrappers when present
        self.model = getattr(model, "module", model)
        self.ctx = ctx
        self.device = device
        self.tokenizer = tokenizer
        cfg = getattr(self.model, "config", None)
        # nanoGPT exposes block_size; fall back to max_seq_len for compatibility.
        self.block_size = int(getattr(cfg, "block_size", getattr(cfg, "max_seq_len", 0)) or 0)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def get_next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with self.ctx:
            logits, _ = self.model(input_ids)
        # nanoGPT returns (B, 1, V) when targets is None; (B, T, V) otherwise.
        return logits[:, -1, :]

    def _truncate_context_and_continuation(
        self,
        context_ids: List[int],
        continuation_ids: List[int],
    ) -> tuple[List[int], List[int]]:
        context_ids = list(context_ids)
        continuation_ids = list(continuation_ids)

        if self.block_size:
            total_len = len(context_ids) + len(continuation_ids)
            if total_len > self.block_size:
                overflow = total_len - self.block_size
                if overflow >= len(context_ids):
                    drop_from_cont = overflow - len(context_ids)
                    context_ids = []
                    continuation_ids = continuation_ids[drop_from_cont:]
                else:
                    context_ids = context_ids[overflow:]

        if len(continuation_ids) == 0:
            return context_ids, continuation_ids
        if len(context_ids) == 0:
            context_ids = [continuation_ids[0]]
            continuation_ids = continuation_ids[1:]

        return context_ids, continuation_ids

    def score_continuation(self, context_ids: List[int], continuation_ids: List[int]) -> float:
        """Return the summed log-prob of ``continuation_ids`` given ``context_ids``."""
        context_ids, continuation_ids = self._truncate_context_and_continuation(
            context_ids, continuation_ids
        )
        if len(continuation_ids) == 0:
            return float("-inf")

        full_ids = context_ids + continuation_ids
        inputs = full_ids[:-1]
        targets = full_ids[1:]
        # First target index that belongs to the continuation span.
        valid_start = len(context_ids) - 1

        input_tensor = torch.tensor([inputs], dtype=torch.long, device=self.device)
        # Pass a (throwaway) targets tensor purely to force the full-logits branch
        # of nanoGPT's forward; we ignore the returned loss and score manually.
        with self.ctx:
            logits, _ = self.model(input_tensor, input_tensor)

        log_probs = F.log_softmax(logits[0].float(), dim=-1)  # (T, V)
        target_tensor = torch.tensor(targets, dtype=torch.long, device=log_probs.device)
        token_lp = log_probs.gather(-1, target_tensor.unsqueeze(-1)).squeeze(-1)  # (T,)
        score = token_lp[valid_start:].sum().item()
        return float(score)
