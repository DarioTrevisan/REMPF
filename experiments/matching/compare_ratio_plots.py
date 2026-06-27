#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rempf.io.npz import load_npz, save_npz
from p4_ratio_plot import compute_ratio_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare matching ratio curves on common axes."
    )
    parser.add_argument("--npz-a", required=True, help="First sweep .npz")
    parser.add_argument("--label-a", required=True, help="Legend label for first curve")
    parser.add_argument("--npz-b", required=True, help="Second sweep .npz")
    parser.add_argument("--label-b", required=True, help="Legend label for second curve")
    parser.add_argument("--out", required=True, help="Output PDF path")
    parser.add_argument(
        "--error-bars",
        choices=["none", "se", "ci95"],
        default="ci95",
        help="Uncertainty bars for the ratio estimator, computed by delta method.",
    )
    parser.add_argument(
        "--out-data",
        default=None,
        help="Optional comparison-data .npz path (default: same stem as --out with .npz)",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional sidecar JSON path (default: same stem as --out with .json)",
    )
    return parser.parse_args()


def _plot_curve(
    ax: plt.Axes,
    stats: dict[str, np.ndarray | float | int | str],
    *,
    label: str,
    error_bars: str,
) -> None:
    n_list = np.asarray(stats["n_list"], dtype=int)
    ratio = np.asarray(stats["ratio"], dtype=float)
    ratio_se = np.asarray(stats["ratio_se"], dtype=float)

    if error_bars == "none":
        ax.plot(n_list, ratio, marker="o", linewidth=1.8, label=label)
        return

    scale = 1.0 if error_bars == "se" else 1.96
    ax.errorbar(
        n_list,
        ratio,
        yerr=scale * ratio_se,
        marker="o",
        linewidth=1.8,
        capsize=3.0,
        label=label,
    )


def main() -> None:
    args = parse_args()
    data_a = load_npz(args.npz_a)
    data_b = load_npz(args.npz_b)
    stats_a = compute_ratio_stats(data_a)
    stats_b = compute_ratio_stats(data_b)

    n_a = np.asarray(stats_a["n_list"], dtype=int)
    n_b = np.asarray(stats_b["n_list"], dtype=int)
    n_union = np.unique(np.concatenate((n_a, n_b)))

    out_pdf = Path(args.out)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    _plot_curve(ax, stats_a, label=args.label_a, error_bars=args.error_bars)
    _plot_curve(ax, stats_b, label=args.label_b, error_bars=args.error_bars)
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_union)
    ax.set_xticklabels([str(n) for n in n_union])
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\mathbb{E}[M_4] / (\mathbb{E}[M_2])^2$")
    ax.set_title("Matching ratio comparison on the torus")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

    out_data = Path(args.out_data) if args.out_data else out_pdf.with_suffix(".npz")
    save_npz(
        str(out_data),
        arrays={
            "n_list_a": n_a,
            "ratio_a": np.asarray(stats_a["ratio"], dtype=float),
            "ratio_se_a": np.asarray(stats_a["ratio_se"], dtype=float),
            "n_list_b": n_b,
            "ratio_b": np.asarray(stats_b["ratio"], dtype=float),
            "ratio_se_b": np.asarray(stats_b["ratio_se"], dtype=float),
        },
        params={
            "problem": "matching_ratio_comparison",
            "source_a": str(args.npz_a),
            "label_a": args.label_a,
            "source_b": str(args.npz_b),
            "label_b": args.label_b,
            "error_bars": args.error_bars,
        },
    )

    out_json = Path(args.out_json) if args.out_json else out_pdf.with_suffix(".json")
    summary = {
        "problem": "matching_ratio_comparison",
        "source_a": str(args.npz_a),
        "label_a": args.label_a,
        "source_b": str(args.npz_b),
        "label_b": args.label_b,
        "plot_output": str(out_pdf),
        "data_output": str(out_data),
        "error_bars": args.error_bars,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
