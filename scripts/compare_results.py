"""Compare NIAH results of several models on shared charts.

Each ``--series`` is ``LABEL=FILE[,FILE...]``: one model's results (CSV or JSON from scripts/run_niah.py); several files
for one model (an in-context sweep plus extrapolation sweeps) are merged, one entry per context length.

Examples
--------
  python scripts/compare_results.py \
      --series "dilution=dil_16k.json,dil_extrap.json" --series "softmax + RoPE=rope_16k.json" \
      --trained-length 16384 --chance 0.143 --out-dir results/ --prefix compare

Writes ``<prefix>_accuracy.png`` (accuracy vs context, one line per model, the region past the training context shaded)
and ``<prefix>_heatmap.png`` (depth x context heatmaps stacked on one context axis). When the results carry
per-placement rows (``NiahConfig(record_rows=True)``) it also writes ``<prefix>_grid.png``, the classic fine-grained
needle-depth x context-length grid, one panel per model.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chart_results import load_results  # noqa: E402

from niah.charting import merge_results, plot_comparison_accuracy, plot_comparison_heatmap, plot_depth_grid  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare NIAH results of several models.")
    ap.add_argument("--series", action="append", required=True, metavar="LABEL=FILE[,FILE...]")
    ap.add_argument("--dashed", action="append", default=[], metavar="LABEL", help="draw this series dashed")
    ap.add_argument("--trained-length", type=int, default=None, help="training context, marked on both charts")
    ap.add_argument("--chance", type=float, default=None, help="chance accuracy line (e.g. 1/7 for 7 scored choices)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--prefix", default="niah_compare")
    args = ap.parse_args()

    series = []
    for spec in args.series:
        label, _, files = spec.partition("=")
        results = merge_results(*[load_results(Path(f)) for f in files.split(",") if f])
        series.append({"label": label, "results": results, "linestyle": "--" if label in args.dashed else "-"})
    out = Path(args.out_dir)
    kw = {"title": args.title} if args.title else {}
    a = plot_comparison_accuracy(series, out / f"{args.prefix}_accuracy.png", args.trained_length, args.chance, **kw)
    h = plot_comparison_heatmap(series, out / f"{args.prefix}_heatmap.png", args.trained_length)
    print(f"wrote {a}\nwrote {h}")
    if any(r.get("rows") for s in series for r in s["results"]):   # per-placement rows: the fine depth grid
        print(f"wrote {plot_depth_grid(series, out / f'{args.prefix}_grid.png', args.trained_length)}")


if __name__ == "__main__":
    main()
