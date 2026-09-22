"""Standalone NIAH eval over a trained nanoGPT checkpoint.

Examples
--------
  # single context length
  python scripts/run_niah.py --ckpt out-niah-demo/ckpt.pt --context-length 256 --samples 5

  # sweep several context lengths + write charts
  python scripts/run_niah.py --ckpt out-niah-demo/ckpt.pt \
      --context-length 64 128 256 --csv-out results/niah.csv --chart results/

Writes a JSON file per run and a CSV with one row per context length (the same
schema ``run_niah`` returns), then optionally renders the depth x context heatmap
and accuracy-vs-context charts.
"""

import argparse
import csv
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

# make the repo root importable when run as `python scripts/run_niah.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import GPT, GPTConfig  # noqa: E402
from niah.eval import NiahConfig, run_niah  # noqa: E402
from niah.tokenizer import load_tokenizer  # noqa: E402


def load_model_from_checkpoint(ckpt_path: str, device: str) -> GPT:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."  # torch.compile artifact
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


CSV_FIELDS = [
    "context_length", "accuracy", "correct", "total", "evaluated_needles",
    "disqualified_needles", "avg_retrieval_gap_ll", "avg_prior_adjusted_lift_ll",
    "early_accuracy", "mid_accuracy", "late_accuracy", "duration_s",
]


def _write_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the NIAH eval over a nanoGPT checkpoint.")
    ap.add_argument("--ckpt", required=True, help="Path to a nanoGPT ckpt.pt")
    ap.add_argument("--context-length", type=int, nargs="+", default=[192],
                    help="One or more context lengths (sweep if multiple): tokens of scored model "
                         "input, question included. Each must be <= the model block_size.")
    ap.add_argument("--num-needles", type=int, default=4)
    ap.add_argument("--samples", type=int, default=5, help="Needle depth placements per keyword.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default="gpt2", help="Must match the model's training tokenizer.")
    ap.add_argument("--use-shared-choices", dest="use_shared_choices", action="store_true", default=True)
    ap.add_argument("--random-choices", dest="use_shared_choices", action="store_false",
                    help="Sample random single-token choices instead of the curated word list.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--out", default=None, help="JSON output path (defaults next to --csv-out or ./results).")
    ap.add_argument("--csv-out", default="results/niah.csv")
    ap.add_argument("--chart", default=None, help="Directory to write charts into (optional).")
    args = ap.parse_args()

    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    model = load_model_from_checkpoint(args.ckpt, args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    block_size = model.config.block_size
    print(f"Loaded model: block_size={block_size}, vocab_size={model.config.vocab_size}")

    results: list[dict] = []
    for ctx_len in args.context_length:
        if ctx_len > block_size:
            print(f"[skip] context_length {ctx_len} > model block_size {block_size}")
            continue
        cfg = NiahConfig(
            context_length=ctx_len,
            num_needles=args.num_needles,
            placements_per_needle=args.samples,
            seed=args.seed,
            use_shared_choices=args.use_shared_choices,
            canonical_tokenizer=args.tokenizer,
        )
        metrics = run_niah(model, config=cfg, tokenizer=tokenizer, device=args.device, ctx=ctx)
        if "error" in metrics:
            print(f"[ctx={ctx_len}] ERROR: {metrics['error']}")
            continue
        print(f"[ctx={ctx_len}] accuracy={metrics['accuracy']:.3f}  "
              f"gap={metrics['avg_retrieval_gap_ll']:.2f}  "
              f"early/mid/late={metrics['early_accuracy']:.2f}/"
              f"{metrics['mid_accuracy']:.2f}/{metrics['late_accuracy']:.2f}  "
              f"n={metrics['total']}")
        results.append(metrics)

    if not results:
        print("No successful runs; nothing written.")
        sys.exit(1)

    csv_path = Path(args.csv_out)
    _write_csv(results, csv_path)
    print(f"Wrote CSV: {csv_path}")

    json_path = Path(args.out) if args.out else csv_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote JSON: {json_path}")

    if args.chart:
        from niah.charting import save_charts
        paths = save_charts(results, args.chart)
        print("Wrote charts: " + ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
