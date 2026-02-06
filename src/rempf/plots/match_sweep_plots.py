from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

from rempf.core.scaling import r_dp


def plot_sweep_normalized(
    *,
    n_list: np.ndarray,
    costs: np.ndarray,
    dim: int,
    q_eval: np.ndarray,
    title: str,
    out: str,
    quantiles=(0.1, 0.9),
) -> None:
    """
    costs: shape (Nn, trials, nq)
    Plot (for each q) mean(cost)/r(d,q)(n) with quantile band across trials.
    """
    n_list = np.asarray(n_list)
    q_eval = np.asarray(q_eval, dtype=float)

    if costs.ndim != 3:
        raise ValueError(f"Expected costs shape (Nn, trials, nq), got {costs.shape}")

    Nn, T, nq = costs.shape
    if Nn != len(n_list):
        raise ValueError(f"n_list has len {len(n_list)} but costs has Nn={Nn}")
    if nq != len(q_eval):
        raise ValueError(f"q_eval has len {len(q_eval)} but costs has nq={nq}")

    qlo, qhi = quantiles

    plt.figure()

    for j, q in enumerate(q_eval):
        # Normalize each trial at each n: shape (Nn, T)
        scales = np.array([r_dp(int(n), dim, float(q)) for n in n_list], dtype=float)  # (Nn,)
        norm = costs[:, :, j] / scales[:, None]  # (Nn, T)

        mean = norm.mean(axis=1)                    # (Nn,)
        lo = np.quantile(norm, qlo, axis=1)         # (Nn,)
        hi = np.quantile(norm, qhi, axis=1)         # (Nn,)

        plt.plot(n_list, mean, marker="o", label=f"q={q:g}")
        plt.fill_between(n_list, lo, hi, alpha=0.2)

    plt.xscale("log")
    plt.xlabel("n")
    plt.ylabel("cost / r(d,q)(n)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_sweep_concentration(
    *,
    n_list: np.ndarray,
    costs: np.ndarray,
    dim: int,
    q_eval: np.ndarray,
    title: str,
    out: str,
) -> None:
    """
    Plot relative fluctuations: std(cost)/mean(cost) as a function of n for each q.
    costs shape (Nn, trials, nq)
    """
    n_list = np.asarray(n_list)
    q_eval = np.asarray(q_eval, dtype=float)

    plt.figure()

    for j, q in enumerate(q_eval):
        rel = np.empty(len(n_list), dtype=float)
        for i, n in enumerate(n_list):
            x = costs[i, :, j]
            m = float(np.mean(x))
            s = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
            rel[i] = s / m if m > 0 else np.nan
        plt.plot(n_list, rel, marker="o", label=f"q={q:g}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("std/mean")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    
    
def plot_sweep_variance(
    *,
    n_list: np.ndarray,
    costs: np.ndarray,
    dim: int,
    q_eval: np.ndarray,
    title: str,
    out: str,
):
    import matplotlib.pyplot as plt
    import numpy as np
    from rempf.core.scaling import r_dp

    plt.figure()

    for j, q in enumerate(q_eval):
        vars_ = []
        for i, n in enumerate(n_list):
            norm = costs[i, :, j] / r_dp(int(n), dim, float(q))
            vars_.append(np.var(norm, ddof=1))
        plt.plot(n_list, vars_, marker="o", label=f"q={q:g}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("Var(cost / r(d,q)(n))")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

