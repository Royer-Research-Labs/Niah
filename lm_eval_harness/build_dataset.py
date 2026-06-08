"""Build a self-contained NIAH multiple-choice dataset for the LM Eval Harness.

Unlike scripts/prepare_spec.py (which writes the canonical {haystack, needle, depth}
rows), this writes rows that the lm-eval ``multiple_choice`` task can consume directly:

    {"context": "<haystack>\\n\\n<question>", "choices": [...], "gold": <int>, "depth": "0.50"}

Example
-------
  python lm_eval_harness/build_dataset.py --tokens 256 --samples 5 --choices 4 \
      --out lm_eval_harness/data/niah_256.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from niah.spec import build_niah_spec, iter_niah_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a NIAH multiple-choice JSONL for lm-eval.")
    ap.add_argument("--tokens", type=int, default=256, help="Context length in tokens.")
    ap.add_argument("--samples", type=int, default=5, help="Needle depth placements per keyword.")
    ap.add_argument("--choices", type=int, default=4, help="Number of candidate keywords.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--use-shared-choices", action="store_true", default=True)
    ap.add_argument("--random-choices", dest="use_shared_choices", action="store_false")
    ap.add_argument("--out", default="lm_eval_harness/data/niah.jsonl")
    args = ap.parse_args()

    spec = build_niah_spec(
        tokenizer_name=args.tokenizer,
        use_shared_choices=args.use_shared_choices,
        num_samples=args.samples,
        context_length=args.tokens,
        n_choices=args.choices,
        seed=args.seed,
    )
    choices = list(spec["choices"])
    question = spec["question"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in iter_niah_rows(spec):
            record = {
                "context": f"{row['haystack']}\n\n{question}",
                "choices": choices,
                "gold": choices.index(row["needle"]),
                "depth": row["depth"],
            }
            json.dump(record, f)
            f.write("\n")
            n += 1

    print(f"Wrote {n} rows to {out_path}")
    print(f"choices = {choices}")
    print("Point niah_multiple_choice.yaml's dataset_kwargs.data_files at this file.")


if __name__ == "__main__":
    main()
