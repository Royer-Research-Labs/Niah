"""Render NIAH charts from a results CSV or JSON produced by scripts/run_niah.py.

Examples
--------
  python scripts/chart_results.py results/niah.csv --out-dir results/
  python scripts/chart_results.py results/niah.json --out-dir results/ --prefix demo
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from niah.charting import save_charts  # noqa: E402

_FLOAT_KEYS = {
    "accuracy", "avg_retrieval_gap_ll", "avg_prior_adjusted_lift_ll",
    "early_accuracy", "mid_accuracy", "late_accuracy", "duration_s",
}
_INT_KEYS = {"context_length", "correct", "total", "evaluated_needles", "disqualified_needles"}


def _coerce(row: dict) -> dict:
    out = dict(row)
    for k in _FLOAT_KEYS & out.keys():
        try:
            out[k] = float(out[k])
        except (TypeError, ValueError):
            out[k] = 0.0
    for k in _INT_KEYS & out.keys():
        try:
            out[k] = int(float(out[k]))
        except (TypeError, ValueError):
            out[k] = 0
    return out


def load_results(path: Path) -> list[dict]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [_coerce(r) for r in (data if isinstance(data, list) else [data])]
    with path.open("r", newline="", encoding="utf-8") as f:
        return [_coerce(r) for r in csv.DictReader(f)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Chart NIAH results.")
    ap.add_argument("results", help="Path to a results .csv or .json")
    ap.add_argument("--out-dir", default=None, help="Directory for charts (defaults to the results dir).")
    ap.add_argument("--prefix", default="niah")
    args = ap.parse_args()

    path = Path(args.results)
    results = load_results(path)
    if not results:
        print("No rows found in results file.")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else path.parent
    paths = save_charts(results, out_dir, prefix=args.prefix)
    print("Wrote charts: " + ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
