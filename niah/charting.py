"""Charting for NIAH results (matplotlib only, headless-safe).

Two views, both built from a list of per-run metrics dicts (the dicts returned by
``run_niah``, typically one per context length in a sweep):

  * ``plot_depth_context_heatmap`` - the classic NIAH grid: depth bucket (early/mid/late)
    on the y-axis, context length on the x-axis, retrieval accuracy as color.
  * ``plot_accuracy_vs_context`` - overall accuracy as a function of context length.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless / no-display safe
import matplotlib.pyplot as plt  # noqa: E402

DEPTH_BUCKETS = ["early", "mid", "late"]


def _sorted_by_context(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(results, key=lambda r: int(r.get("context_length", 0)))


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


def save_charts(results: List[Dict[str, Any]], out_dir: str | Path, prefix: str = "niah") -> List[Path]:
    """Write both charts into ``out_dir`` and return their paths."""
    out_dir = Path(out_dir)
    return [
        plot_depth_context_heatmap(results, out_dir / f"{prefix}_heatmap.png"),
        plot_accuracy_vs_context(results, out_dir / f"{prefix}_accuracy_vs_context.png"),
    ]
