# NIAH × LM Evaluation Harness

This directory wires the NIAH eval into EleutherAI's
[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) so you
can run it (and any other lm-eval task) against a locally-trained nanoGPT model.

Two pieces are provided:

| File | Purpose |
|------|---------|
| [`nanogpt_lm.py`](nanogpt_lm.py) | An lm-eval model adapter registered as `nanogpt` that loads a nanoGPT `ckpt.pt` and scores log-likelihoods. |
| [`niah_multiple_choice.yaml`](niah_multiple_choice.yaml) | A `multiple_choice` task definition (task name `niah_nanogpt`). |
| [`build_dataset.py`](build_dataset.py) | Generates the self-contained MC JSONL the task reads. |

## Install

```bash
pip install lm-eval
```

## Step by step

**1. Build the dataset.** This materializes NIAH rows into a JSONL where each row
carries its own `context`, `choices`, and `gold` index:

```bash
python lm_eval_harness/build_dataset.py --tokens 192 --samples 5 --choices 4 \
    --out lm_eval_harness/data/niah_192.jsonl
```

(Keep `--tokens` a bit below the model's `block_size` to leave room for the question;
the demo model's `block_size` is 256, so 192 is a safe choice.)

**2. Run lm-eval.** Use the bundled launcher [`run_lm_eval.py`](run_lm_eval.py); it
takes exactly the same flags as the `lm_eval` CLI but first registers the `nanogpt`
model (lm-eval's `--include_path` discovers task YAMLs but not custom model `.py`
files) and applies a small cross-version compatibility shim. `--include_path
lm_eval_harness` points it at the `niah_nanogpt` task:

```bash
python lm_eval_harness/run_lm_eval.py \
        --model nanogpt \
        --model_args ckpt=out-niah-demo/ckpt.pt,device=cpu,tokenizer=gpt2 \
        --tasks niah_nanogpt \
        --include_path lm_eval_harness \
        --limit 20
```

`--model_args` accepts: `ckpt` (required), `device`, `tokenizer`, `dtype`
(`float32`/`bfloat16`/`float16`), `max_length`.

Example output:

```
|   Tasks    |Version|Filter|n-shot|Metric|   |Value|   |Stderr|
|------------|------:|------|-----:|------|---|----:|---|-----:|
|niah_nanogpt|      1|none  |     0|acc   |↑  | 0.25|±  |0.1306|
```

You'll get an `acc` score: the fraction of rows where the highest-likelihood choice
is the true needle.

## How it maps to NIAH

The task is plain multiple-choice: for each row the harness scores every candidate
keyword as a continuation of the haystack+question and picks the argmax. This matches
the core of the native eval (`scripts/run_niah.py`), **minus** the control-prior
disqualification step. For the full method (control prior, retrieval-gap thresholds,
depth buckets, charts), use the native runner. Use this lm-eval path when you want
NIAH reported alongside standard benchmarks in one harness.

## Reusing the model adapter for other tasks

Because `nanogpt` is a general lm-eval model, you can point it at any task:

```bash
python lm_eval_harness/run_lm_eval.py --model nanogpt \
        --model_args ckpt=out-niah-demo/ckpt.pt,device=cpu \
        --tasks lambada_openai,hellaswag --include_path lm_eval_harness --limit 50
```
