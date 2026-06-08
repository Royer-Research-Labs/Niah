"""An LM Evaluation Harness model adapter for nanoGPT checkpoints.

Registers the model type ``nanogpt`` so you can run any lm-eval task (including the
bundled NIAH multiple-choice task) against a locally-trained nanoGPT model:

    lm_eval --model nanogpt \
            --model_args ckpt=out-niah-demo/ckpt.pt,device=cpu,tokenizer=gpt2 \
            --tasks niah --limit 20 \
            --include_path lm_eval_harness

Only ``loglikelihood`` is needed for multiple-choice tasks like NIAH; simple
``generate_until`` and ``loglikelihood_rolling`` implementations are included so
the class satisfies the lm-eval ``LM`` interface for other tasks too.

NOTE: this is the *plain* multiple-choice form of NIAH (argmax over choice
log-likelihoods). The control-prior disqualification that the native runner
(scripts/run_niah.py) applies is not part of the lm-eval task.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

# make the repo root importable regardless of where lm-eval is invoked from
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lm-eval is required for this adapter. Install it with `pip install lm-eval`."
    ) from exc

from model import GPT, GPTConfig  # noqa: E402
from niah.tokenizer import load_tokenizer  # noqa: E402


@register_model("nanogpt")
class NanoGPTLM(LM):
    def __init__(
        self,
        ckpt: str,
        device: str | None = None,
        tokenizer: str = "gpt2",
        dtype: str = "float32",
        max_length: int | None = None,
        batch_size=1,
        **kwargs,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device_type = "cuda" if "cuda" in self.device else "cpu"
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        self.ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        checkpoint = torch.load(ckpt, map_location=self.device)
        self.model = GPT(GPTConfig(**checkpoint["model_args"]))
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self.enc = load_tokenizer(tokenizer)
        self.block_size = int(self.model.config.block_size)
        self._max_length = int(max_length) if max_length else self.block_size
        self._batch_size = int(batch_size)

    @property
    def eot_token_id(self) -> int:
        return self.enc.eot_token_id if self.enc.eot_token_id is not None else 0

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 64

    @property
    def batch_size(self) -> int:
        return self._batch_size

    # --- scoring -----------------------------------------------------------------
    def _encode_pair(self, context: str, continuation: str):
        ctx_ids = self.enc.encode(context) if context else []
        cont_ids = self.enc.encode(continuation)
        if not ctx_ids:
            ctx_ids = [self.eot_token_id]
        return ctx_ids, cont_ids

    @torch.no_grad()
    def _score(self, context: str, continuation: str) -> tuple[float, bool]:
        ctx_ids, cont_ids = self._encode_pair(context, continuation)
        if not cont_ids:
            return 0.0, True

        full = ctx_ids + cont_ids
        ctx_len = len(ctx_ids)
        # keep the right-most (block_size + 1) tokens so the continuation is retained
        if len(full) > self.block_size + 1:
            cut = len(full) - (self.block_size + 1)
            full = full[cut:]
            ctx_len = max(1, ctx_len - cut)

        inputs = torch.tensor([full[:-1]], dtype=torch.long, device=self.device)
        targets = torch.tensor(full[1:], dtype=torch.long, device=self.device)
        with self.ctx:
            logits, _ = self.model(inputs, inputs)  # pass targets to force full logits
        log_probs = F.log_softmax(logits[0].float(), dim=-1)

        valid_start = ctx_len - 1
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        cont_lp = token_lp[valid_start:]
        greedy = log_probs.argmax(dim=-1)[valid_start:]
        is_greedy = bool(torch.equal(greedy, targets[valid_start:]))
        return float(cont_lp.sum().item()), is_greedy

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        out = []
        for request in requests:
            context, continuation = request.args
            out.append(self._score(context, continuation))
        return out

    @torch.no_grad()
    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        out = []
        for request in requests:
            (text,) = request.args
            ids = self.enc.encode(text)
            total = 0.0
            # score in non-overlapping block_size windows, conditioning each on a BOS
            for start in range(0, len(ids), self.block_size):
                window = ids[start:start + self.block_size]
                ll, _ = self._score("", self.enc.decode(window))
                total += ll
            out.append(total)
        return out

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm: bool = False):
        out = []
        for request in requests:
            context, gen_kwargs = request.args
            until = (gen_kwargs or {}).get("until", []) if isinstance(gen_kwargs, dict) else []
            max_new = (gen_kwargs or {}).get("max_gen_toks", self.max_gen_toks) if isinstance(gen_kwargs, dict) else self.max_gen_toks
            ids = self.enc.encode(context)[-self.block_size:]
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            with self.ctx:
                y = self.model.generate(x, max_new_tokens=int(max_new), temperature=1.0, top_k=None)
            text = self.enc.decode(y[0].tolist()[len(ids):])
            for stop in (until or []):
                if stop and stop in text:
                    text = text.split(stop)[0]
                    break
            out.append(text)
        return out
