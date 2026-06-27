#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rempf.io.npz import load_npz, save_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot E[M4] / (E[M2])^2 for matching sweeps."
    )
    parser.add_argument("npz", type=str, help="Sweep output with q_eval containing 2 and 4")
    parser.add_argument("--out", type=str, required=True, help="Output PDF path")
    parser.add_argument(
        "--error-bars",
        choices=["none", "se", "ci95"],
        default="none",
        help="Uncertainty bars for the ratio estimator, computed by delta method from trial samples.",
    )
    parser.add_argument(
        "--out-data",
        type=str,
        default=None,
        help="Optional derived-data .npz path (default: same stem as --out with .npz)",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional sidecar JSON path (default: same stem as --out with .json)",
    )
    return parser.parse_args()


def compute_ratio_stats(data: dict) -> dict[str, np.ndarray | float | int | str]:
    q_eval = np.asarray(data["q_eval"], dtype=float)
    costs = np.asarray(data["costs"], dtype=float)
    n_list = np.asarray(data["n_list"], dtype=int)

    try:
        q2_idx = int(np.where(np.isclose(q_eval, 2.0))[0][0])
        q4_idx = int(np.where(np.isclose(q_eval, 4.0))[0][0])
    except IndexError as exc:
        raise ValueError("Input sweep must contain both q=2 and q=4 in q_eval.") from exc

    second_moments = costs[:, :, q2_idx] / n_list[:, None]
    fourth_moments = costs[:, :, q4_idx] / n_list[:, None]

    mean_second_moment = second_moments.mean(axis=1)
    mean_fourth_moment = fourth_moments.mean(axis=1)
    ratio = mean_fourth_moment / (mean_second_moment**2)

    ratio_se = np.full_like(ratio, np.nan, dtype=float)
    trials = second_moments.shape[1]
    if trials > 1:
        for i in range(len(n_list)):
            sample = np.column_stack((second_moments[i], fourth_moments[i]))
            cov = np.cov(sample, rowvar=False, ddof=1)
            a = mean_second_moment[i]
            b = mean_fourth_moment[i]
            grad = np.array([-2.0 * b / (a**3), 1.0 / (a**2)], dtype=float)
            ratio_var = float(grad @ cov @ grad) / trials
            ratio_se[i] = np.sqrt(max(ratio_var, 0.0))

    return {
        "n_list": n_list,
        "q_eval": q_eval,
        "mean_cost_q2": costs[:, :, q2_idx].mean(axis=1),
        "mean_cost_q4": costs[:, :, q4_idx].mean(axis=1),
        "mean_second_moment": mean_second_moment,
        "mean_fourth_moment": mean_fourth_moment,
        "ratio": ratio,
        "ratio_se": ratio_se,
    }


def main() -> None:
    args = parse_args()
    data = load_npz(args.npz)
    stats = compute_ratio_stats(data)
    n_list = np.asarray(stats["n_list"], dtype=int)
    ratio = np.asarray(stats["ratio"], dtype=float)
    ratio_se = np.asarray(stats["ratio_se"], dtype=float)
    q_eval = np.asarray(stats["q_eval"], dtype=float)

    out_pdf = Path(args.out)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if args.error_bars == "none":
        ax.plot(n_list, ratio, marker="o", linewidth=1.8)
    else:
        scale = 1.0 if args.error_bars == "se" else 1.96
        ax.errorbar(
            n_list,
            ratio,
            yerr=scale * ratio_se,
            marker="o",
            linewidth=1.8,
            capsize=3.0,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_list)
    ax.set_xticklabels([str(n) for n in n_list])
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\mathbb{E}[M_4] / (\mathbb{E}[M_2])^2$")
    ax.set_title(
        f"p={float(data['params']['p_opt']):g}-optimal matching ratio on the {data['params']['domain']}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

    out_data = Path(args.out_data) if args.out_data else out_pdf.with_suffix(".npz")
    params = {
        "problem": "matching_p4_ratio",
        "source_npz": str(args.npz),
        "dim": int(data["params"]["dim"]),
        "domain": str(data["params"]["domain"]),
        "p_opt": float(data["params"]["p_opt"]),
        "q_eval": [float(x) for x in q_eval],
        "n_list": [int(x) for x in n_list],
        "trials": int(data["params"]["trials"]),
        "observable": "E[(1/n) sum |X_i-Y_sigma(i)|^4] / (E[(1/n) sum |X_i-Y_sigma(i)|^2])^2",
    }
    save_npz(
        str(out_data),
        arrays={
            "n_list": n_list,
            "mean_cost_q2": np.asarray(stats["mean_cost_q2"], dtype=float),
            "mean_cost_q4": np.asarray(stats["mean_cost_q4"], dtype=float),
            "mean_second_moment": np.asarray(stats["mean_second_moment"], dtype=float),
            "mean_fourth_moment": np.asarray(stats["mean_fourth_moment"], dtype=float),
            "ratio": ratio,
            "ratio_se": ratio_se,
        },
        params=params,
    )

    out_json = Path(args.out_json) if args.out_json else out_pdf.with_suffix(".json")
    summary = {
        "problem": "matching_p4_ratio",
        "source_npz": str(args.npz),
        "plot_output": str(out_pdf),
        "data_output": str(out_data),
        "dimension_d": int(data["params"]["dim"]),
        "domain": str(data["params"]["domain"]),
        "p_opt": float(data["params"]["p_opt"]),
        "n_values": [int(x) for x in n_list],
        "trials_per_n": int(data["params"]["trials"]),
        "ratio_values": [float(x) for x in ratio],
        "ratio_se": [float(x) for x in ratio_se],
        "error_bars": args.error_bars,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
