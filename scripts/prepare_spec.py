"""Create a NIAH dataset as either a compact runtime spec (.spec.json) or a
materialized JSONL file (one row per needle placement).

The JSONL form is what the LM Evaluation Harness integration consumes
(see lm_eval_harness/). The spec form is a tiny, reproducible descriptor.

Examples
--------
  # materialized JSONL for lm-eval
  python scripts/prepare_spec.py --tokens 256 --samples 5 --choices 4 --format jsonl \
      --out data/niah/niah_256.jsonl

  # compact spec only
  python scripts/prepare_spec.py --tokens 256 --format spec --out data/niah/niah_256.spec.json
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from niah.spec import build_niah_spec, materialize_niah_jsonl, write_niah_spec  # noqa: E402


def _resolve_output_path(raw_out: str, output_format: str) -> Path:
    path = Path(raw_out)
    if output_format == "spec":
        if path.suffix == ".jsonl":
            return path.with_suffix(".spec.json")
    else:
        if path.name.endswith(".spec.json"):
            return path.with_name(path.name[:-10] + ".jsonl")
        if path.suffix == ".json":
            return path.with_suffix(".jsonl")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a NIAH dataset (spec or JSONL).")
    ap.add_argument("--tokens", required=True, type=int, help="Context length in tokens.")
    ap.add_argument("--samples", type=int, default=10, help="Needle depth placements per keyword.")
    ap.add_argument("--choices", type=int, default=4, help="Number of candidate keywords.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--format", choices=["jsonl", "spec"], default="jsonl")
    ap.add_argument("--out", default="data/niah/niah.jsonl")
    ap.add_argument("--tokenizer", default="gpt2", help="Tokenizer name (e.g. gpt2 or an HF repo id).")
    ap.add_argument("--use-shared-choices", action="store_true",
                    help="Use the curated single-token word list instead of random sampling.")
    args = ap.parse_args()

    output_path = _resolve_output_path(args.out, args.format)
    print(f"Creating NIAH dataset ({args.format}) -> {output_path}")
    print(f"  tokenizer={args.tokenizer}  context={args.tokens}  samples={args.samples}  choices={args.choices}")

    spec = build_niah_spec(
        tokenizer_name=args.tokenizer,
        use_shared_choices=args.use_shared_choices,
        num_samples=args.samples,
        context_length=args.tokens,
        n_choices=args.choices,
        seed=args.seed,
    )
    print(f"  canonical_tokenizer={spec['canonical_tokenizer']}  n_choices={len(spec['choices'])}")

    if args.format == "spec":
        spec_path = write_niah_spec(spec, output_path)
        print(f"Wrote spec: {spec_path}")
        return

    stats = materialize_niah_jsonl(spec, output_path)
    print(f"Wrote {stats['row_count']} rows: {stats['output_path']}")
    print(f"Companion meta: {stats['meta_path']}")
    print(f"Control haystack length: {stats['control_haystack_len']} tokens")


if __name__ == "__main__":
    main()
