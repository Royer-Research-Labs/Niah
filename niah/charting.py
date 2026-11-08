"""Charting for NIAH results (matplotlib only, headless-safe).

Two views, both built from a list of per-run metrics dicts (the dicts returned by
``run_niah``, typically one per context length in a sweep):

  * ``plot_depth_context_heatmap`` - the classic NIAH grid: depth bucket (early/mid/late)
    on the y-axis, context length on the x-axis, retrieval accuracy as color.
  * ``plot_accuracy_vs_context`` - overall accuracy as a function of context length.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless / no-display safe
import matplotlib.pyplot as plt  # noqa: E402

DEPTH_BUCKETS = ["early", "mid", "late"]


def _sorted_by_context(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(results, key=lambda r: int(r.get("context_length", 0)))


def _fmt_len(n: int) -> str:
    """Tick label: 32768 -> 32K, 16000 -> 16K, 896 -> 896."""
    if n >= 1000:
        if n % 1000 == 0:
            return f"{n // 1000}K"
        return f"{n // 1024}K" if n % 1024 == 0 else f"{round(n / 1000)}K"
    return str(n)


def _fmt_ctx(n: int) -> str:
    return f"{n // 1024}k" if n and n % 1024 == 0 else str(n)


def plot_depth_context_heatmap(results: List[Dict[str, Any]], out_path: str | Path) -> Path:
    """Heatmap of accuracy across (depth bucket x context length)."""
    results = _sorted_by_context(results)
    if not results:
        raise ValueError("No results to chart.")

    contexts = [int(r.get("context_length", 0)) for r in results]
    grid = [[float(r.get(f"{bucket}_accuracy", 0.0)) for r in results] for bucket in DEPTH_BUCKETS]

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(results) + 2), 3.2))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels([_fmt_ctx(c) for c in contexts])
    ax.set_yticks(range(len(DEPTH_BUCKETS)))
    ax.set_yticklabels([b.capitalize() for b in DEPTH_BUCKETS])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Needle depth")
    ax.set_title("NIAH retrieval accuracy")

    for i in range(len(DEPTH_BUCKETS)):
        for j in range(len(contexts)):
            ax.text(j, i, f"{grid[i][j]:.0%}", ha="center", va="center", fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_accuracy_vs_context(results: List[Dict[str, Any]], out_path: str | Path) -> Path:
    """Line chart of overall accuracy vs context length."""
    results = _sorted_by_context(results)
    if not results:
        raise ValueError("No results to chart.")

    contexts = [int(r.get("context_length", 0)) for r in results]
    accuracy = [float(r.get("accuracy", 0.0)) for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(contexts, accuracy, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts)
    ax.set_xticklabels([_fmt_ctx(c) for c in contexts])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Overall accuracy")
    ax.set_title("NIAH accuracy vs context length")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def merge_results(*result_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Combine sweeps of one model (e.g. an in-context sweep plus extrapolation sweeps) into one list, one entry per
    context length; where two sweeps share a length, the one with more scored placements is kept."""
    by_ctx: Dict[int, Dict[str, Any]] = {}
    for results in result_lists:
        for r in results:
            c = int(r.get("context_length", 0))
            if c not in by_ctx or int(r.get("total", 0)) > int(by_ctx[c].get("total", 0)):
                by_ctx[c] = r
    return [by_ctx[c] for c in sorted(by_ctx)]


def plot_comparison_accuracy(series: List[Dict[str, Any]], out_path: str | Path, trained_length: int | None = None,
                             chance: float | None = None, title: str = "NIAH retrieval vs context length") -> Path:
    """Accuracy vs context length for several models on one chart. ``series`` items: {"label", "results",
    optional "linestyle"}. ``trained_length`` shades everything past the training context; ``chance`` draws the floor."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    all_ctx = sorted({int(r["context_length"]) for s in series for r in s["results"]})
    if trained_length:
        ax.axvspan(trained_length, max(all_ctx) * 1.15, color="0.92", zorder=0)
        ax.axvline(trained_length, color="0.45", linestyle="--", linewidth=1)
        ax.text(trained_length * 1.04, 0.04, "past the training\ncontext", fontsize=8, color="0.35", va="bottom")
    if chance is not None:
        ax.axhline(chance, color="0.55", linestyle=":", linewidth=1)
        ax.text(max(all_ctx) * 1.1, chance + 0.015, f"chance ({chance:.2f})", fontsize=8, color="0.4", ha="right")
    for s in series:
        res = _sorted_by_context(s["results"])
        ax.plot([int(r["context_length"]) for r in res], [float(r["accuracy"]) for r in res], marker="o",
                markersize=4, linewidth=1.8, linestyle=s.get("linestyle", "-"), label=s["label"])
    ax.set_xscale("log", base=2)
    # ticks at powers of two (plus the training length) keep labels apart on the log axis
    lo, hi = min(all_ctx), max(all_ctx)
    ticks = [2 ** k for k in range(int(math.log2(lo)), math.ceil(math.log2(hi)) + 1) if lo * 0.9 <= 2 ** k <= hi * 1.1]
    ticks = [t for t in ticks if t >= 512] if len(ticks) > 8 else ticks
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1024}K" if t >= 1024 else str(t) for t in ticks], fontsize=8)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(lo * 0.85, hi * 1.15)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("Haystack length (tokens)")
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.01, 0.2), frameon=True)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_comparison_heatmap(series: List[Dict[str, Any]], out_path: str | Path, trained_length: int | None = None,
                            title: str = "NIAH retrieval by needle depth and haystack length") -> Path:
    """Depth x context heatmaps for several models stacked on one shared context axis (a row group per model). Cells a
    model was not evaluated at are left blank; a vertical rule marks the training context."""
    all_ctx = sorted({int(r["context_length"]) for s in series for r in s["results"]})
    rows, labels, groups = [], [], []
    for s in series:
        by_ctx = {int(r["context_length"]): r for r in s["results"]}
        for b in DEPTH_BUCKETS:
            rows.append([float(by_ctx[c][f"{b}_accuracy"]) if c in by_ctx else float("nan") for c in all_ctx])
            labels.append(b)
        groups.append(s["label"])
    import numpy as np
    grid = np.array(rows)
    fig, ax = plt.subplots(figsize=(max(8, 0.62 * len(all_ctx) + 3), 0.42 * len(rows) + 1.6))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center", va="center", fontsize=7,
                        color="black" if v > 0.6 else "white")
    for g in range(1, len(series)):
        ax.axhline(g * len(DEPTH_BUCKETS) - 0.5, color="white", linewidth=3)
    if trained_length:
        past = [j for j, c in enumerate(all_ctx) if c > trained_length]
        if past:
            ax.axvline(past[0] - 0.5, color="red", linewidth=1.5)
    ax.set_xticks(range(len(all_ctx)))
    ax.set_xticklabels([_fmt_len(c) for c in all_ctx], fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    for g, name in enumerate(groups):
        # model name left of the depth labels: x in axes fraction, y in data rows
        ax.annotate(name, xy=(0, g * len(DEPTH_BUCKETS) + 1), xycoords=("axes fraction", "data"), xytext=(-42, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Haystack length (tokens)" + ("  |  red rule: training context" if trained_length else ""))
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Accuracy")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _depth_grid(results: List[Dict[str, Any]], contexts: List[int]):
    """(depths, grid) from per-placement rows: grid[d][c] = share of scored keywords retrieved at that depth and length
    (NaN where the model was not evaluated at that length)."""
    import numpy as np
    cells: Dict[tuple, List[float]] = {}
    for r in results:
        c = int(r["context_length"])
        for row in r.get("rows") or []:
            cells.setdefault((round(float(row["depth"]), 3), c), []).append(1.0 if row["pass"] else 0.0)
    depths = sorted({d for d, _ in cells})
    grid = np.full((len(depths), len(contexts)), np.nan)
    for (d, c), v in cells.items():
        if c in contexts:
            grid[depths.index(d), contexts.index(c)] = sum(v) / len(v)
    return depths, grid


def plot_depth_grid(series: List[Dict[str, Any]], out_path: str | Path, trained_length: int | None = None,
                    title: str = "NIAH retrieval by needle depth and context length") -> Path:
    """The classic fine-grained NIAH grid (needle depth 0-100% x context length), one panel per model on a shared context
    axis. Needs per-placement rows (``run_niah(..., config=NiahConfig(record_rows=True))``); each cell is the share of
    keywords retrieved at that depth and length, so with several keywords per depth the colours are graded rather than
    pass / fail. A vertical rule marks the training context; lengths a model was not run at are left blank."""
    import numpy as np
    all_ctx = sorted({int(r["context_length"]) for s in series for r in s["results"] if r.get("rows")})
    if not all_ctx:
        raise ValueError("No per-placement rows: run with NiahConfig(record_rows=True).")
    fig, axes = plt.subplots(len(series), 1, figsize=(max(8, 0.55 * len(all_ctx) + 3), 3.4 * len(series) + 0.8),
                             squeeze=False)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("white")
    im = None
    for ax, s in zip(axes[:, 0], series):
        depths, grid = _depth_grid(s["results"], all_ctx)
        im = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
                       interpolation="nearest")
        ax.set_xticks(range(len(all_ctx)))
        ax.set_xticklabels([_fmt_len(c) for c in all_ctx], fontsize=7)
        step = max(1, len(depths) // 6)
        ax.set_yticks(range(0, len(depths), step))
        ax.set_yticklabels([f"{depths[i]:.0%}" for i in range(0, len(depths), step)], fontsize=7)
        ax.set_ylabel("Needle depth\n(top = start of input)", fontsize=8)
        ax.set_title(s["label"], fontsize=10, loc="left", fontweight="bold")
        ax.set_xticks(np.arange(-0.5, len(all_ctx)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(depths)), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.4)
        ax.tick_params(which="minor", length=0)
        if trained_length:
            past = [j for j, c in enumerate(all_ctx) if c > trained_length]
            if past:
                ax.axvline(past[0] - 0.5, color="black", linewidth=1.5, linestyle="--")
    axes[-1, 0].set_xlabel("Context length (tokens)" + ("  |  dashed rule: training context" if trained_length else ""))
    fig.suptitle(title, fontsize=12)
    cbar = fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("Share of keywords retrieved")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_charts(results: List[Dict[str, Any]], out_dir: str | Path, prefix: str = "niah") -> List[Path]:
    """Write both charts into ``out_dir`` and return their paths."""
    out_dir = Path(out_dir)
    return [
        plot_depth_context_heatmap(results, out_dir / f"{prefix}_heatmap.png"),
        plot_accuracy_vs_context(results, out_dir / f"{prefix}_accuracy_vs_context.png"),
    ]
