#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RATIO_NAMES = ("R22", "R44_self", "R44_cross", "Q2", "I4")
MEAN_NAMES = ("mean_M2_p2", "mean_M4_p2", "mean_M2_p4", "mean_M4_p4")
REQUIRED_COLUMNS = ("n", "seed", "p_opt", "M2", "M4")
IDENTITY_TOL = 1e-10


@dataclass(frozen=True)
class FitResult:
    ratio: str
    model: str
    min_n: int
    points: int
    r_inf: float
    stderr: float


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze matching p=2/p=4 ratio diagnostics from raw table data."
    )
    parser.add_argument("--input", required=True, help="Input .csv, .parquet, or .feather file")
    parser.add_argument("--outdir", required=True, help="Directory for figures and summaries")
    parser.add_argument("--paired", type=parse_bool, default=True, help="Use paired seeds only")
    parser.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap replicates per n")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided CI alpha")
    parser.add_argument("--target-a", type=float, default=1.80788, help="Formal flow a* target")
    parser.add_argument("--rng-seed", type=int, default=12345, help="Bootstrap RNG seed")
    return parser.parse_args()


def load_input(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix in {".feather", ".ft"}:
        df = pd.read_feather(input_path)
    else:
        raise ValueError("Input must be .csv, .parquet, or .feather.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    return df


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings_out: list[str] = []
    work = df.copy()

    if "status" in work.columns:
        status = work["status"].fillna("ok").astype(str).str.lower()
        good = status.isin({"ok", "success", "done", "completed", ""})
        failed_count = int((~good).sum())
        if failed_count:
            warnings_out.append(f"Excluded {failed_count} rows with non-ok status.")
        work = work.loc[good].copy()

    for col in ("n", "p_opt", "M2", "M4"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["seed"] = work["seed"].astype(str)
    work = work.dropna(subset=["n", "p_opt", "M2", "M4"])
    work["n"] = work["n"].astype(int)

    bad_costs = work[(work["M2"] <= 0.0) | (work["M4"] <= 0.0)]
    if not bad_costs.empty:
        raise ValueError(f"All costs must be positive; found {len(bad_costs)} invalid rows.")

    p2 = np.isclose(work["p_opt"].to_numpy(dtype=float), 2.0)
    p4 = np.isclose(work["p_opt"].to_numpy(dtype=float), 4.0)
    dropped = int((~(p2 | p4)).sum())
    if dropped:
        warnings_out.append(f"Ignored {dropped} rows whose p_opt is not close to 2 or 4.")
    work = work.loc[p2 | p4].copy()
    work["p_label"] = np.where(np.isclose(work["p_opt"].to_numpy(dtype=float), 2.0), "p2", "p4")

    dupes = work.duplicated(subset=["n", "seed", "p_label"], keep=False)
    if dupes.any():
        warnings_out.append(
            f"Averaged {int(dupes.sum())} duplicate rows sharing the same (n, seed, p_opt)."
        )
        work = (
            work.groupby(["n", "seed", "p_label"], as_index=False)
            .agg(M2=("M2", "mean"), M4=("M4", "mean"))
            .assign(p_opt=lambda d: np.where(d["p_label"] == "p2", 2.0, 4.0))
        )

    if work.empty:
        raise ValueError("No usable p_opt=2 or p_opt=4 rows remain after filtering.")
    return work, warnings_out


def pivot_by_seed(df_n: pd.DataFrame) -> pd.DataFrame:
    wide = df_n.pivot_table(index="seed", columns="p_label", values=["M2", "M4"], aggfunc="mean")
    wide.columns = [f"{metric}_{label}" for metric, label in wide.columns]
    return wide.reset_index()


def compute_from_wide(wide: pd.DataFrame) -> dict[str, float]:
    means = {
        "mean_M2_p2": float(wide["M2_p2"].mean()),
        "mean_M4_p2": float(wide["M4_p2"].mean()),
        "mean_M2_p4": float(wide["M2_p4"].mean()),
        "mean_M4_p4": float(wide["M4_p4"].mean()),
    }
    r22 = means["mean_M4_p2"] / means["mean_M2_p2"] ** 2
    r44_self = means["mean_M4_p4"] / means["mean_M2_p4"] ** 2
    r44_cross = means["mean_M4_p4"] / means["mean_M2_p2"] ** 2
    q2 = means["mean_M2_p4"] / means["mean_M2_p2"]
    i4 = 1.0 - means["mean_M4_p4"] / means["mean_M4_p2"]
    return {
        **means,
        "R22": float(r22),
        "R44_self": float(r44_self),
        "R44_cross": float(r44_cross),
        "Q2": float(q2),
        "I4": float(i4),
        "identity_error": float(r44_cross - r44_self * q2**2),
    }


def compute_from_unpaired(wide: pd.DataFrame) -> dict[str, float]:
    cols = ["M2_p2", "M4_p2", "M2_p4", "M4_p4"]
    if any(wide[col].dropna().empty for col in cols):
        raise ValueError("Each n must have at least one p=2 and one p=4 sample.")
    return compute_from_wide(wide)


def summarize_one_n(
    df_n: pd.DataFrame,
    *,
    paired: bool,
    bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[dict[str, float], pd.DataFrame, list[str]]:
    notes: list[str] = []
    wide_all = pivot_by_seed(df_n)
    paired_wide = wide_all.dropna(subset=["M2_p2", "M4_p2", "M2_p4", "M4_p4"]).copy()
    analysis_wide = paired_wide if paired else wide_all

    if paired and analysis_wide.empty:
        raise ValueError(f"n={int(df_n['n'].iloc[0])} has no seeds with both optimizers.")
    if len(analysis_wide) < 2:
        notes.append(f"n={int(df_n['n'].iloc[0])} has fewer than two analysis samples.")

    point = compute_from_wide(analysis_wide) if paired else compute_from_unpaired(analysis_wide)
    point["samples_p2"] = int(wide_all["M2_p2"].notna().sum())
    point["samples_p4"] = int(wide_all["M2_p4"].notna().sum())
    point["paired_samples"] = int(len(paired_wide))
    point["analysis_samples"] = int(len(analysis_wide))
    point["n"] = int(df_n["n"].iloc[0])

    boot_rows: list[dict[str, float]] = []
    if bootstrap > 0 and len(analysis_wide) > 0:
        for _ in range(bootstrap):
            if paired:
                idx = rng.integers(0, len(analysis_wide), size=len(analysis_wide))
                block = analysis_wide.iloc[idx]
                sample = compute_from_wide(block)
            else:
                parts: dict[str, pd.Series] = {}
                for col in ["M2_p2", "M4_p2", "M2_p4", "M4_p4"]:
                    values = analysis_wide[col].dropna().reset_index(drop=True)
                    idx = rng.integers(0, len(values), size=len(values))
                    parts[col] = values.iloc[idx].reset_index(drop=True)
                max_len = max(len(values) for values in parts.values())
                block = pd.DataFrame({col: values.reindex(range(max_len)) for col, values in parts.items()})
                sample = compute_from_unpaired(block)
            if all(np.isfinite(sample[name]) for name in (*RATIO_NAMES, *MEAN_NAMES)):
                boot_rows.append(sample)
        if len(boot_rows) != bootstrap:
            notes.append(
                f"n={point['n']} skipped {bootstrap - len(boot_rows)} non-finite bootstrap samples."
            )

    boot_df = pd.DataFrame(boot_rows)
    for name in RATIO_NAMES:
        if not boot_df.empty:
            values = boot_df[name].to_numpy(dtype=float)
            point[f"{name}_se"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            point[f"{name}_ci_low"] = float(np.percentile(values, 100.0 * alpha / 2.0))
            point[f"{name}_ci_high"] = float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0)))
        else:
            point[f"{name}_se"] = float("nan")
            point[f"{name}_ci_low"] = float("nan")
            point[f"{name}_ci_high"] = float("nan")

    if paired and len(analysis_wide) > 0:
        q2_s = analysis_wide["M2_p4"] / analysis_wide["M2_p2"]
        c44_s = analysis_wide["M4_p4"] / analysis_wide["M2_p2"] ** 2
        point["paired_Q2_sample_mean"] = float(q2_s.mean())
        point["paired_C44_sample_mean"] = float(c44_s.mean())
    else:
        point["paired_Q2_sample_mean"] = float("nan")
        point["paired_C44_sample_mean"] = float("nan")

    return point, boot_df.assign(n=point["n"]) if not boot_df.empty else boot_df, notes


def summarize(
    df: pd.DataFrame,
    *,
    paired: bool,
    bootstrap: int,
    alpha: float,
    rng_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(rng_seed)
    summaries: list[dict[str, float]] = []
    boots: list[pd.DataFrame] = []
    notes: list[str] = []
    for n, df_n in df.groupby("n", sort=True):
        point, boot_df, n_notes = summarize_one_n(
            df_n, paired=paired, bootstrap=bootstrap, alpha=alpha, rng=rng
        )
        if abs(point["identity_error"]) > IDENTITY_TOL:
            notes.append(
                f"n={n} identity error {point['identity_error']:.3e} exceeds {IDENTITY_TOL}."
            )
        notes.extend(n_notes)
        summaries.append(point)
        if not boot_df.empty:
            boots.append(boot_df)
    summary = pd.DataFrame(summaries).sort_values("n").reset_index(drop=True)
    boot = pd.concat(boots, ignore_index=True) if boots else pd.DataFrame()
    return summary, boot, notes


def errorbar(
    ax: plt.Axes,
    summary: pd.DataFrame,
    column: str,
    *,
    label: str,
    target: float | None = None,
) -> None:
    yerr = np.vstack(
        [
            summary[column] - summary[f"{column}_ci_low"],
            summary[f"{column}_ci_high"] - summary[column],
        ]
    )
    if not np.isfinite(yerr).all():
        yerr = None
    ax.errorbar(
        summary["n"].to_numpy(dtype=float),
        summary[column].to_numpy(dtype=float),
        yerr=yerr,
        marker="o",
        linewidth=1.8,
        capsize=3.0,
        label=label,
    )
    if target is not None:
        ax.axhline(target, color="0.35", linestyle="--", linewidth=1.0)


def finish_log2_axis(ax: plt.Axes, ylabel: str) -> None:
    n_values = np.asarray(ax.lines[0].get_xdata(), dtype=float) if ax.lines else np.array([])
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(int(n)) for n in n_values])
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def save_ratio_plots(summary: pd.DataFrame, outdir: Path, target_a: float) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    errorbar(ax, summary, "R22", label=r"$R_{22}$")
    errorbar(ax, summary, "R44_self", label=r"$R_{44}^{self}$")
    errorbar(ax, summary, "R44_cross", label=r"$R_{44}^{cross}$")
    ax.axhline(2.0, color="0.25", linestyle="--", linewidth=1.0, label="linear Gaussian ratio 2")
    ax.axhline(target_a, color="0.45", linestyle=":", linewidth=1.2, label=rf"formal flow $a_*={target_a:g}$")
    finish_log2_axis(ax, "ratio")
    ax.set_title("Matching ratio diagnostics")
    fig.tight_layout()
    fig.savefig(outdir / "ratio_curves.pdf")
    fig.savefig(outdir / "ratio_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    errorbar(ax, summary, "R44_self", label=r"$R_{44}^{self}$")
    errorbar(ax, summary, "R44_cross", label=r"$R_{44}^{cross}$")
    ax.plot(
        summary["n"].to_numpy(dtype=float),
        (summary["R44_self"] * summary["Q2"] ** 2).to_numpy(dtype=float),
        marker="x",
        linewidth=1.4,
        label=r"$R_{44}^{self} Q_2^2$",
    )
    ax.axhline(target_a, color="0.45", linestyle=":", linewidth=1.2, label=rf"$a_*={target_a:g}$")
    finish_log2_axis(ax, "p=4 ratio")
    ax.set_title("Self-normalized versus cross-normalized p=4 ratio")
    fig.tight_layout()
    fig.savefig(outdir / "cross_vs_self.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    errorbar(ax, summary, "Q2", label=r"$Q_2$")
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0, label="no inflation")
    finish_log2_axis(ax, r"$\bar M_2^{(4)} / \bar M_2^{(2)}$")
    ax.set_title("Quadratic inflation of the p=4 optimizer")
    fig.tight_layout()
    fig.savefig(outdir / "quadratic_inflation.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    errorbar(ax, summary, "I4", label=r"$I_4$")
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    finish_log2_axis(ax, r"$1 - \bar M_4^{(4)} / \bar M_4^{(2)}$")
    ax.set_title("Quartic improvement over the p=2 optimizer")
    fig.tight_layout()
    fig.savefig(outdir / "quartic_improvement.pdf")
    plt.close(fig)


def design_matrix(log_n: np.ndarray, model: str) -> np.ndarray:
    if model == "1/log(n)":
        return np.column_stack([np.ones_like(log_n), 1.0 / log_n])
    if model == "1/log(n)+1/log(n)^2":
        return np.column_stack([np.ones_like(log_n), 1.0 / log_n, 1.0 / log_n**2])
    if model == "1/sqrt(log(n))":
        return np.column_stack([np.ones_like(log_n), 1.0 / np.sqrt(log_n)])
    if model == "1/log(n)^2":
        return np.column_stack([np.ones_like(log_n), 1.0 / log_n**2])
    raise ValueError(model)


def fit_wls(x: np.ndarray, y: np.ndarray, se: np.ndarray, model: str) -> tuple[float, float]:
    mat = design_matrix(x, model)
    if len(y) < mat.shape[1]:
        return float("nan"), float("nan")
    weights = np.where(np.isfinite(se) & (se > 0.0), 1.0 / se**2, 1.0)
    root_w = np.sqrt(weights)
    xw = mat * root_w[:, None]
    yw = y * root_w
    beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    resid = yw - xw @ beta
    dof = max(len(y) - mat.shape[1], 1)
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.pinv(xw.T @ xw)
    return float(beta[0]), float(math.sqrt(max(cov[0, 0], 0.0)))


def finite_size_fits(summary: pd.DataFrame) -> list[FitResult]:
    results: list[FitResult] = []
    models = ("1/log(n)", "1/log(n)+1/log(n)^2", "1/sqrt(log(n))", "1/log(n)^2")
    for ratio in ("R22", "R44_self", "R44_cross"):
        for min_n in (256, 512):
            tail = summary[summary["n"] >= min_n].copy()
            if tail.empty:
                continue
            log_n = np.log(tail["n"].to_numpy(dtype=float))
            y = tail[ratio].to_numpy(dtype=float)
            se = tail[f"{ratio}_se"].to_numpy(dtype=float)
            for model in models:
                r_inf, stderr = fit_wls(log_n, y, se, model)
                if np.isfinite(r_inf):
                    results.append(
                        FitResult(
                            ratio=ratio,
                            model=model,
                            min_n=min_n,
                            points=len(tail),
                            r_inf=r_inf,
                            stderr=stderr,
                        )
                    )
    return results


def dataframe_to_markdown(df: pd.DataFrame, *, floatfmt: str = ".6g") -> str:
    headers = [str(col) for col in df.columns]
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        rendered: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append(format(float(value), floatfmt) if np.isfinite(value) else "nan")
            else:
                rendered.append(str(value))
        rows.append(rendered)

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(cells, widths)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rows)])


def save_finite_size_plot(summary: pd.DataFrame, outdir: Path, fits: list[FitResult]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=False)
    for ratio in ("R22", "R44_self", "R44_cross"):
        axes[0].errorbar(
            1.0 / np.log(summary["n"].to_numpy(dtype=float)),
            summary[ratio].to_numpy(dtype=float),
            yerr=summary[f"{ratio}_se"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
            capsize=3.0,
            label=ratio,
        )
        axes[1].errorbar(
            1.0 / np.log(summary["n"].to_numpy(dtype=float)) ** 2,
            summary[ratio].to_numpy(dtype=float),
            yerr=summary[f"{ratio}_se"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
            capsize=3.0,
            label=ratio,
        )
    axes[0].set_xlabel(r"$1/\log(n)$")
    axes[1].set_xlabel(r"$1/\log(n)^2$")
    for ax in axes:
        ax.set_ylabel("ratio")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Finite-size fit diagnostics; extrapolations are not conclusive")
    fig.tight_layout()
    fig.savefig(outdir / "finite_size_fits.pdf")
    plt.close(fig)

    fit_table = pd.DataFrame([result.__dict__ for result in fits])
    if not fit_table.empty:
        fit_table.to_csv(outdir / "finite_size_fits.csv", index=False)


def qc_notes(summary: pd.DataFrame, notes: list[str]) -> list[str]:
    out = list(notes)
    max_identity = float(summary["identity_error"].abs().max())
    if max_identity > IDENTITY_TOL:
        out.append(f"Maximum identity error {max_identity:.3e} exceeds tolerance {IDENTITY_TOL}.")
    if (summary["samples_p2"] < 2).any() or (summary["samples_p4"] < 2).any():
        out.append("At least one n has fewer than two samples for an optimizer.")
    if len(summary) >= 3:
        last = summary.tail(max(1, len(summary) // 3))
        r22_tail = float(last["R22"].mean())
        if abs(r22_tail - 2.0) > 0.35:
            out.append(f"Large-n average R22 is {r22_tail:.5g}, not close to 2.")
    for ratio in RATIO_NAMES:
        widths = summary[f"{ratio}_ci_high"] - summary[f"{ratio}_ci_low"]
        if widths.notna().sum() >= 3 and widths.iloc[-1] > 2.0 * widths.iloc[0]:
            out.append(f"{ratio} CI width grows substantially from first to last n.")
    return out


def write_raw_qc(df: pd.DataFrame, summary: pd.DataFrame, notes: list[str], outdir: Path) -> None:
    lines = [
        "# Raw QC",
        "",
        f"Usable rows: {len(df)}",
        f"n values: {', '.join(str(int(n)) for n in sorted(df['n'].unique()))}",
        "",
        "## Samples by n and optimizer",
        "",
        dataframe_to_markdown(df.groupby(["n", "p_label"]).size().unstack(fill_value=0).reset_index()),
        "",
        "## Paired samples",
        "",
        dataframe_to_markdown(summary[["n", "samples_p2", "samples_p4", "paired_samples"]]),
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {note}" for note in notes] or ["- None"])
    (outdir / "raw_qc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    summary: pd.DataFrame,
    fits: list[FitResult],
    notes: list[str],
    outdir: Path,
    *,
    paired: bool,
    bootstrap: int,
    alpha: float,
    target_a: float,
) -> None:
    latest = summary.iloc[-1]
    fit_table = pd.DataFrame([result.__dict__ for result in fits])
    max_identity = float(summary["identity_error"].abs().max())
    cross_gap = abs(float(latest["R44_cross"]) - target_a)
    self_gap = abs(float(latest["R44_self"]) - target_a)
    explained = cross_gap < self_gap

    table_cols = [
        "n",
        "analysis_samples",
        "R22",
        "R22_ci_low",
        "R22_ci_high",
        "R44_self",
        "R44_self_ci_low",
        "R44_self_ci_high",
        "R44_cross",
        "R44_cross_ci_low",
        "R44_cross_ci_high",
        "Q2",
        "Q2_ci_low",
        "Q2_ci_high",
        "I4",
        "I4_ci_low",
        "I4_ci_high",
    ]
    lines = [
        "# Matching ratio diagnostics",
        "",
        f"Input analysis mode: {'paired' if paired else 'unpaired'}",
        f"Bootstrap replicates per n: {bootstrap}",
        f"Confidence level: {(1.0 - alpha) * 100.0:.1f}%",
        f"Formal flow target a*: {target_a:.5f}",
        "",
        "## Data summary",
        "",
        dataframe_to_markdown(
            summary[["n", "samples_p2", "samples_p4", "paired_samples", "analysis_samples"]]
        ),
        "",
        "## Main estimates",
        "",
        dataframe_to_markdown(summary[table_cols], floatfmt=".6g"),
        "",
        "## Identity check",
        "",
        r"The diagnostic identity is $R_{44}^{cross}=R_{44}^{self}Q_2^2$.",
        f"Maximum absolute identity error: {max_identity:.3e}.",
        "",
        "## Formal target comparison",
        "",
        f"At the largest n={int(latest['n'])}, R22={latest['R22']:.5f} versus 2.00000.",
        (
            f"At the largest n={int(latest['n'])}, R44_cross={latest['R44_cross']:.5f} "
            f"and R44_self={latest['R44_self']:.5f} versus a*={target_a:.5f}."
        ),
        (
            "The self-normalized p=4 curve is a different normalization from the formal flow "
            "comparison; the cross-normalized curve is the relevant a* diagnostic."
        ),
        "",
        "## Finite-size fits",
        "",
        "These extrapolations are diagnostic only and should not be overinterpreted.",
        "",
    ]
    if fit_table.empty:
        lines.append("No finite-size fits were available.")
    else:
        lines.append(dataframe_to_markdown(fit_table, floatfmt=".6g"))

    lines.extend(
        [
            "",
            "## QC warnings",
            "",
            *([f"- {note}" for note in notes] if notes else ["- None"]),
            "",
            "## Conclusion",
            "",
            (
                f"The self-normalized p=4 ratio is near {latest['R44_self']:.5f}, while the "
                f"cross-normalized ratio is near {latest['R44_cross']:.5f}. "
                f"The quadratic inflation Q2 is {latest['Q2']:.5f}. "
                f"Therefore the apparent discrepancy with a* is "
                f"{'more consistent with self-normalization' if explained else 'not fully explained by self-normalization'} "
                "in the largest available size."
            ),
        ]
    )
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be nonnegative.")
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("--alpha must be between 0 and 1.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_input(args.input)
    df, prep_notes = prepare_data(raw)
    summary, boot, summary_notes = summarize(
        df,
        paired=bool(args.paired),
        bootstrap=int(args.bootstrap),
        alpha=float(args.alpha),
        rng_seed=int(args.rng_seed),
    )
    notes = qc_notes(summary, [*prep_notes, *summary_notes])

    summary.to_csv(outdir / "summary_by_n.csv", index=False)
    try:
        summary.to_parquet(outdir / "summary_by_n.parquet", index=False)
    except Exception as exc:  # pragma: no cover - optional dependency surface
        notes.append(f"Could not write summary_by_n.parquet: {exc}")
    if not boot.empty:
        try:
            boot.to_parquet(outdir / "bootstrap_samples.parquet", index=False)
        except Exception as exc:  # pragma: no cover - optional dependency surface
            notes.append(f"Could not write bootstrap_samples.parquet: {exc}")

    save_ratio_plots(summary, outdir, target_a=float(args.target_a))
    fits = finite_size_fits(summary)
    save_finite_size_plot(summary, outdir, fits)
    write_raw_qc(df, summary, notes, outdir)
    write_report(
        summary,
        fits,
        notes,
        outdir,
        paired=bool(args.paired),
        bootstrap=int(args.bootstrap),
        alpha=float(args.alpha),
        target_a=float(args.target_a),
    )

    if notes:
        for note in notes:
            warnings.warn(note, stacklevel=1)


if __name__ == "__main__":
    main()
